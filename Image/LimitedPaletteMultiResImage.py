# LimitedPaletteMultiResImage.py

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image

from pytti import DEVICE, clamp_with_grad
from pytti.Image import DifferentiableImage

class LimitedPaletteMultiResImage(DifferentiableImage):
    """
    Limited Palette Multi-resolution Image.
    
    This model represents an image with a multi-resolution decomposition 
    (learnable residuals at several scales) and then applies a limited palette
    quantization step. The continuous output (after summing residuals and applying tanh)
    is mapped to discrete colors by computing a soft assignment over a learned palette,
    which is then replaced with a hard assignment using a straight-through estimator.
    
    Parameters:
      - width, height: Final output resolution.
      - palette_size: Number of palette entries.
      - scales: Tuple of downscale factors for the multi-resolution decomposition.
      - gamma: Gamma correction to apply when encoding an image.
      - temperature: Temperature for softmax quantization (lower makes the distribution sharper).
      - init: Initialization for the residuals ('random' or 'zeros').
    """
    def __init__(self, width, height, palette_size, scales=(1,2,4,8,16),
                 gamma=0.8, temperature=0.5, init='random', device=DEVICE):
        super().__init__(width, height, 'RGB')
        self.pixel_format = 'RGB'
        self.scales = scales
        self.gamma = gamma
        self.temperature = temperature
        n_channels = 3
        
        # Multi-resolution residuals: list of parameters for each scale.
        self.residuals = nn.ParameterList()
        for s in scales:
            h_down = max(1, height // s)
            w_down = max(1, width // s)
            if init == 'random':
                data = 2 * torch.rand(n_channels, h_down, w_down, device=device) - 1
            else:
                data = torch.zeros(n_channels, h_down, w_down, device=device)
            self.residuals.append(nn.Parameter(data))
        
        # Palette: learnable parameter of shape [palette_size, 3], values in [0,1]
        self.palette = nn.Parameter(torch.rand(palette_size, 3, device=device))
        # Buffer to store target palette.
        self.register_buffer('palette_target', torch.empty_like(self.palette))
        self.use_palette_target = False
        
        self.output_axes = ('n', 's', 'y', 'x')
        self.lr = 0.1

    def decode_tensor(self):
        """
        Decodes the image by:
          1. Upsampling each multi-resolution residual to (H,W) and summing.
          2. Applying tanh and mapping the result from [-1,1] to [0,1] to get a continuous image.
          3. For each pixel, computing squared distances to each palette color and
             using softmax with temperature to obtain soft assignments.
          4. Converting these soft assignments to a hard one-hot vector via a straight-through estimator.
          5. Producing the quantized image as the weighted sum of palette colors.
          
        Returns a tensor of shape [1, 3, H, W].
        """
        width, height = self.image_shape
        n_channels = 3
        device = self.residuals[0].device
        
        # Sum multi-resolution residuals.
        accum = torch.zeros(1, n_channels, height, width, device=device)
        for scale, param in zip(self.scales, self.residuals):
            up = F.interpolate(param.unsqueeze(0), size=(height, width),
                               mode='bilinear', align_corners=True)
            accum = accum + up
        continuous = (torch.tanh(accum) + 1) / 2  # continuous image in [0,1]
        
        # Palette quantization.
        # Expand palette: [1, palette_size, 3, 1, 1]
        palette = self.palette.view(1, -1, 3, 1, 1)
        # Expand continuous image: [1, 1, 3, H, W]
        continuous_exp = continuous.unsqueeze(1)
        # Compute squared L2 distances: [1, palette_size, H, W]
        diff = continuous_exp - palette
        dist2 = (diff ** 2).sum(dim=2)
        logits = -dist2 / self.temperature
        # Compute softmax weights.
        weights_soft = torch.softmax(logits, dim=1)  # [1, palette_size, H, W]
        # Obtain hard assignments: one-hot vectors for each pixel.
        _, max_idx = torch.max(weights_soft, dim=1, keepdim=True)
        weights_hard = torch.zeros_like(weights_soft).scatter_(1, max_idx, 1.0)
        # Straight-through estimator: use hard assignment in forward pass but soft for gradients.
        weights = weights_hard - weights_soft.detach() + weights_soft
        # Weighted sum of palette colors.
        quantized = (weights.unsqueeze(2) * palette).sum(dim=1)  # [1, 3, H, W]
        return clamp_with_grad(quantized, 0, 1)

    def get_image_tensor(self):
        """
        Returns the decoded image tensor with shape [3, H, W].
        """
        return self.decode_tensor().squeeze(0)

    def set_image_tensor(self, tensor):
        """
        Sets the multi-resolution residuals so that decoding yields the given continuous image.
        Inverts the tanh mapping and distributes evenly among scales.
        Expects tensor of shape [3, H, W] in [0,1].
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor with shape [3, H, W], got {tensor.shape}")
        eps = 1e-5
        tensor = tensor.clamp(eps, 1 - eps)
        pre = 0.5 * torch.log(tensor / (1 - tensor))
        N = len(self.scales)
        for i, s in enumerate(self.scales):
            h_down = self.residuals[i].shape[1]
            w_down = self.residuals[i].shape[2]
            down = F.interpolate(pre.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_(down.squeeze(0) / N)

    @torch.no_grad()
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """
        Initializes the multi-resolution residuals from the input image.
        Converts the image to RGB, applies gamma correction, downsamples to each scale,
        and maps the values to [-1,1].
        """
        pil_image = pil_image.convert('RGB')
        full_tensor = TF.to_tensor(pil_image).to(device)
        corrected = full_tensor.pow(self.gamma)
        width, height = self.image_shape
        N = len(self.scales)
        for i, s in enumerate(self.scales):
            h_down, w_down = self.residuals[i].shape[1], self.residuals[i].shape[2]
            down = F.interpolate(corrected.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_((down[0] * 2 - 1) / N)

    @torch.no_grad()
    def encode_random(self, random_palette=False):
        """
        Overwrites the multi-resolution residuals with random noise in [-1,1].
        If random_palette is True, reinitializes the palette to random values in [0,1].
        """
        for param in self.residuals:
            param.uniform_(-1, 1)
        if random_palette:
            self.palette.uniform_(0, 1)

    def update(self):
        """
        Clamps the residuals to [-1,1] and the palette to [0,1].
        """
        for param in self.residuals:
            param.data.clamp_(-1, 1)
        self.palette.data.clamp_(0, 1)

    @torch.no_grad()
    def set_palette_target(self, pil_image):
        """
        Sets the target palette by computing a palette from the input image.
        The image is converted to RGB, flattened, and k-means clustering (from sklearn) is
        used to extract 'palette_size' colors. The resulting palette is sorted by brightness and
        copied into self.palette. Also sets use_palette_target=True.
        """
        if pil_image is None:
            self.use_palette_target = False
            return
        pil_image = pil_image.convert('RGB')
        target_tensor = TF.to_tensor(pil_image).to(self.palette.device)  # [3, H, W]
        flat = target_tensor.view(3, -1).transpose(0, 1)  # [N, 3]
        from sklearn.cluster import KMeans
        n_clusters = self.palette.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(flat.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=self.palette.device, dtype=self.palette.dtype)
        # Sort palette by brightness.
        brightness = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
        sorted_indices = torch.argsort(brightness)
        sorted_palette = centers[sorted_indices]
        with torch.no_grad():
            self.palette.copy_(sorted_palette)
            self.use_palette_target = True

    def lock_palette(self, lock=True):
        """
        Locks or unlocks the palette.
        When locked (lock=True), the palette is fixed and won't change during optimization.
        """
        self.use_palette_target = lock

    def sort_palette(self):
        """
        Returns the palette sorted by brightness.
        """
        brightness = 0.299 * self.palette[:,0] + 0.587 * self.palette[:,1] + 0.114 * self.palette[:,2]
        sorted_indices = torch.argsort(brightness)
        return self.palette[sorted_indices]

def render_palette(palette, cell_size=32):
    """
    Renders the palette (a tensor of shape [N, 3]) as an image.
    Each palette color is displayed as a cell of size (cell_size x cell_size).
    """
    import numpy as np
    from PIL import Image
    palette_np = palette.detach().cpu().numpy()
    n = palette_np.shape[0]
    out = np.zeros((cell_size, cell_size * n, 3), dtype=np.uint8)
    for i, color in enumerate(palette_np):
        out[:, i*cell_size:(i+1)*cell_size, :] = (color * 255).astype(np.uint8)
    return Image.fromarray(out)
