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
    is mapped to discrete colors by computing soft assignments over a learned palette.
    
    Parameters:
      - width, height: Final output resolution.
      - palette_size: Number of palette entries.
      - scales: Tuple of downscale factors used for the multi-resolution decomposition.
      - gamma: Gamma correction parameter applied during encoding (default 0.8).
      - temperature: Temperature for softmax quantization (default 0.5).
      - init: How to initialize the residuals ('random' or 'zeros').
    """
    def __init__(self, width, height, palette_size, scales=(1, 2, 4, 8, 16),
                 gamma=0.8, temperature=0.5, init='random', device=DEVICE):
        super().__init__(width, height, 'RGB')
        # Set pixel_format explicitly
        self.pixel_format = 'RGB'
        self.scales = scales
        self.gamma = gamma
        self.temperature = temperature
        n_channels = 3
        
        # Multi-resolution residuals: each is a parameter of shape [C, H_down, W_down]
        self.residuals = nn.ParameterList()
        for s in scales:
            h_down = max(1, height // s)
            w_down = max(1, width // s)
            if init == 'random':
                data = 2 * torch.rand(n_channels, h_down, w_down, device=device) - 1
            else:
                data = torch.zeros(n_channels, h_down, w_down, device=device)
            self.residuals.append(nn.Parameter(data))
        
        # Palette parameter: shape [palette_size, 3]; values in [0,1]
        self.palette = nn.Parameter(torch.rand(palette_size, 3, device=device))
        
        self.output_axes = ('n', 's', 'y', 'x')
        self.lr = 0.1

    def decode_tensor(self):
        """
        Decode the image by summing up the multi-resolution residuals, applying tanh,
        and then quantizing to the learned palette.
        
        Steps:
          1. For each scale, upsample to the full resolution and sum.
          2. Apply tanh and map to [0,1] to get a continuous image.
          3. For each pixel, compute squared distances to each palette color,
             scale by temperature, and softmax to get weights.
          4. Output the weighted sum of palette colors.
          
        Returns a tensor of shape [1, 3, height, width].
        """
        width, height = self.image_shape
        n_channels = 3
        device = self.residuals[0].device
        
        # Step 1: Sum multi-resolution residuals.
        accum = torch.zeros(1, n_channels, height, width, device=device)
        for scale, param in zip(self.scales, self.residuals):
            up = F.interpolate(param.unsqueeze(0), size=(height, width),
                               mode='bilinear', align_corners=True)
            accum = accum + up
        # Step 2: Map to [0,1]
        continuous = (torch.tanh(accum) + 1) / 2
        
        # Step 3: Limited palette quantization.
        # Expand palette: [1, palette_size, 3, 1, 1]
        palette = self.palette.view(1, -1, 3, 1, 1)
        # Expand continuous image: [1, 1, 3, H, W]
        continuous_exp = continuous.unsqueeze(1)
        # Compute squared L2 distances along channel dimension.
        diff = continuous_exp - palette  # shape: [1, palette_size, 3, H, W]
        dist2 = (diff ** 2).sum(dim=2)    # shape: [1, palette_size, H, W]
        # Compute softmax weights (using negative distances, scaled by temperature).
        logits = -dist2 / self.temperature
        weights = torch.softmax(logits, dim=1)  # shape: [1, palette_size, H, W]
        # Compute weighted sum of palette colors.
        quantized = (weights.unsqueeze(2) * palette).sum(dim=1)  # shape: [1, 3, H, W]
        return clamp_with_grad(quantized, 0, 1)

    def get_image_tensor(self):
        """
        Returns the decoded image tensor with shape [3, height, width].
        """
        return self.decode_tensor().squeeze(0)

    def set_image_tensor(self, tensor):
        """
        Sets the internal multi-resolution residuals so that decoding produces the given continuous image.
        Since quantization is many-to-one, we only update the residuals.
        
        Expects tensor of shape [3, H, W] in [0,1]. Inverts the tanh mapping and distributes
        evenly across the scales.
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor with shape [3, H, W], got {tensor.shape}")
        eps = 1e-5
        tensor = tensor.clamp(eps, 1 - eps)
        pre = 0.5 * torch.log(tensor / (1 - tensor))  # Inversion of tanh mapping.
        N = len(self.scales)
        for i, s in enumerate(self.scales):
            h_down = self.residuals[i].shape[1]
            w_down = self.residuals[i].shape[2]
            down = F.interpolate(pre.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_(down.squeeze(0) / N)
        # Note: We leave the palette unchanged.

    @torch.no_grad()
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """
        Initializes the multi-resolution residuals from the input image.
        Converts the image to RGB, applies gamma correction, then downsamples
        to each scale and maps the values to [-1,1].
        
        The 'smart_encode' flag is accepted for compatibility.
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
        # Optionally, one could initialize the palette from the image,
        # for example via clustering. For now, we leave the palette as is.

    @torch.no_grad()
    def encode_random(self, random_palette=False):
        """
        Overwrites the multi-resolution residuals with random noise in [-1,1].
        If random_palette is True, the palette is also reinitialized to random values in [0,1].
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
