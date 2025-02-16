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
    
    This model represents an image using a multi-resolution decomposition
    (learnable residuals at several scales) and then applies limited palette
    quantization. The continuous image (computed from the residuals via tanh)
    is converted to a discrete image by computing its luminance and then quantizing
    that luminance into indices into a sorted palette. Linear interpolation between
    adjacent palette entries is used to produce the final output.
    
    Parameters:
      - width, height: Final output resolution.
      - palette_size: Number of palette entries.
      - scales: Tuple of downscale factors for the multi-resolution decomposition.
      - gamma: Gamma correction to apply during encoding.
      - init: Initialization for the residuals ('random' or 'zeros').
    """
    def __init__(self, width, height, palette_size, scales=(1,2,4,8,16),
                 gamma=0.8, init='random', device=DEVICE):
        super().__init__(width, height, 'RGB')
        self.pixel_format = 'RGB'
        self.scales = scales
        self.gamma = gamma
        n_channels = 3
        
        # Multi-resolution residuals.
        self.residuals = nn.ParameterList()
        for s in scales:
            h_down = max(1, height // s)
            w_down = max(1, width // s)
            if init == 'random':
                data = 2 * torch.rand(n_channels, h_down, w_down, device=device) - 1
            else:
                data = torch.zeros(n_channels, h_down, w_down, device=device)
            self.residuals.append(nn.Parameter(data))
        
        # Palette: learnable parameter [palette_size, 3] in [0,1]
        self.palette = nn.Parameter(torch.rand(palette_size, 3, device=device))
        self.register_buffer('palette_target', torch.empty_like(self.palette))
        self.use_palette_target = False
        
        self.output_axes = ('n', 's', 'y', 'x')
        self.lr = 0.1

    def decode_tensor(self):
        """
        Decodes the image as follows:
         1. Upsample each multi-resolution residual to full resolution and sum,
            then apply tanh and map to [0,1] to obtain a continuous image.
         2. Compute the luminance (using 0.299, 0.587, 0.114 weights).
         3. Quantize the luminance into an index between 0 and palette_size-1.
         4. For each pixel, linearly interpolate between the palette colors at the floor
            and ceil indices.
        Returns a tensor of shape [1, 3, H, W] that more faithfully preserves the input colors.
        """
        width, height = self.image_shape
        device = self.residuals[0].device

        # 1. Compute continuous image.
        accum = torch.zeros(1, 3, height, width, device=device)
        for scale, param in zip(self.scales, self.residuals):
            up = F.interpolate(param.unsqueeze(0), size=(height, width),
                               mode='bilinear', align_corners=True)
            accum = accum + up
        continuous = (torch.tanh(accum) + 1) / 2  # [1,3,H,W]

        # 2. Compute luminance.
        R = continuous[:,0:1,:,:]
        G = continuous[:,1:2,:,:]
        B = continuous[:,2:3,:,:]
        lum = 0.299 * R + 0.587 * G + 0.114 * B  # [1,1,H,W]

        # 3. Quantize luminance.
        palette_size = self.palette.shape[0]
        index = lum * (palette_size - 1)  # scale lum to [0, palette_size-1]
        floor_index = index.floor()
        ceil_index = index.ceil()
        frac = index - floor_index  # [1,1,H,W], fraction between 0 and 1

        # 4. Get sorted palette.
        sorted_palette = self.sort_palette()  # [palette_size, 3]
        # Remove batch and channel dims.
        floor_index = floor_index.squeeze(0).squeeze(0).long()  # [H,W]
        ceil_index = ceil_index.squeeze(0).squeeze(0).long()      # [H,W]
        frac = frac.squeeze(0).squeeze(0)                         # [H,W]
        
        # Gather palette colors.
        # sorted_palette is [palette_size, 3]. For each pixel, get the floor and ceil colors.
        floor_color = sorted_palette[floor_index]  # [H,W,3]
        ceil_color = sorted_palette[ceil_index]    # [H,W,3]
        quantized = floor_color * (1 - frac.unsqueeze(-1)) + ceil_color * frac.unsqueeze(-1)  # [H,W,3]

        # Permute to [1,3,H,W]
        quantized = quantized.permute(2,0,1).unsqueeze(0)
        return clamp_with_grad(quantized, 0, 1)

    def get_image_tensor(self):
        """Returns decoded image tensor [3, H, W]."""
        return self.decode_tensor().squeeze(0)

    def set_image_tensor(self, tensor):
        """
        Inverts the tanh mapping and distributes evenly among scales,
        so that decoding yields the provided continuous image.
        Expects tensor of shape [3, H, W] in [0,1].
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor with shape [3, H, W], got {tensor.shape}")
        eps = 1e-5
        tensor = tensor.clamp(eps, 1 - eps)
        pre = 0.5 * torch.log(tensor / (1 - tensor))
        N = len(self.scales)
        for i in range(N):
            h_down = self.residuals[i].shape[1]
            w_down = self.residuals[i].shape[2]
            down = F.interpolate(pre.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_(down.squeeze(0) / N)

    @torch.no_grad()
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """
        Initializes the multi-resolution residuals from the input image.
        Converts the image to RGB, applies gamma correction, downsamples it to each scale,
        and maps the values to [-1,1]. If smart_encode is True, also updates the palette via
        k-means clustering on a random subsample of pixels.
        """
        pil_image = pil_image.convert('RGB')
        full_tensor = TF.to_tensor(pil_image).to(device)
        corrected = full_tensor.pow(self.gamma)
        width, height = self.image_shape
        N = len(self.scales)
        for i in range(N):
            h_down, w_down = self.residuals[i].shape[1], self.residuals[i].shape[2]
            down = F.interpolate(corrected.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_((down[0] * 2 - 1) / N)
        if smart_encode:
            sample_size = 4096
            flat = full_tensor.view(3, -1).transpose(0, 1)  # [num_pixels, 3]
            num_pixels = flat.shape[0]
            if num_pixels > sample_size:
                indices = torch.randperm(num_pixels)[:sample_size]
                flat_sample = flat[indices]
            else:
                flat_sample = flat
            from sklearn.cluster import KMeans
            n_clusters = self.palette.shape[0]
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
            labels = kmeans.fit_predict(flat_sample.cpu().numpy())
            centers = torch.tensor(kmeans.cluster_centers_, device=device, dtype=self.palette.dtype)
            brightness = 0.299 * centers[:,0] + 0.587 * centers[:,1] + 0.114 * centers[:,2]
            sorted_indices = torch.argsort(brightness)
            sorted_palette = centers[sorted_indices]
            self.palette.data.copy_(sorted_palette)

    @torch.no_grad()
    def encode_random(self, random_palette=False):
        """
        Overwrites the multi-resolution residuals with random noise in [-1,1].
        If random_palette is True, reinitializes the palette with random values in [0,1].
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
        Updates the palette using k-means clustering on the target image.
        A random subset of up to 4096 pixels is used to speed up clustering.
        The resulting palette is sorted by brightness and copied into self.palette.
        Also sets use_palette_target=True.
        """
        if pil_image is None:
            self.use_palette_target = False
            return
        pil_image = pil_image.convert('RGB')
        target_tensor = TF.to_tensor(pil_image).to(self.palette.device)  # [3, H, W]
        flat = target_tensor.view(3, -1).transpose(0, 1)  # [num_pixels, 3]
        sample_size = 4096
        num_pixels = flat.shape[0]
        if num_pixels > sample_size:
            indices = torch.randperm(num_pixels)[:sample_size]
            flat_sample = flat[indices]
        else:
            flat_sample = flat
        from sklearn.cluster import KMeans
        n_clusters = self.palette.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
        labels = kmeans.fit_predict(flat_sample.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=self.palette.device, dtype=self.palette.dtype)
        brightness = 0.299 * centers[:,0] + 0.587 * centers[:,1] + 0.114 * centers[:,2]
        sorted_indices = torch.argsort(brightness)
        sorted_palette = centers[sorted_indices]
        with torch.no_grad():
            self.palette.copy_(sorted_palette)
            self.use_palette_target = True

    def lock_palette(self, lock=True):
        """
        When locked (lock=True), the palette remains fixed during optimization.
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
    palette_np = palette.detach().cpu().numpy()
    n = palette_np.shape[0]
    out = np.zeros((cell_size, cell_size * n, 3), dtype=np.uint8)
    for i, color in enumerate(palette_np):
        out[:, i*cell_size:(i+1)*cell_size, :] = (color * 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(out)
