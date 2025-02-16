# MultiResImage.py

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image

from pytti import DEVICE, clamp_with_grad, replace_grad
from pytti.Image import DifferentiableImage

class MultiResImage(DifferentiableImage):
    """
    Multi-resolution direct ascent approach for image representation.
    This follows the idea of summing multiple scales of learned parameters,
    as described in the 'Direct Ascent Synthesis' paper.
    """
    def __init__(self,
                 width,
                 height,
                 pixel_format='RGB',
                 scales=(1, 2, 4, 8, 16),
                 init='random',
                 device=DEVICE):
        """
        width, height    : final output resolution
        pixel_format     : PIL image mode, e.g. 'RGB', 'L', etc.
        scales           : list or tuple of integer downscale factors
        init             : how to initialize the parameters ('random' or 'zeros')
        device           : which torch device to use
        """
        super().__init__(width, height, pixel_format)
        self.scales = scales
        
        # Create one Parameter for each scale
        # Each is shape (3, H//scale, W//scale) if RGB, or shape (1, H//scale, W//scale) if L
        # We'll store them in an nn.ModuleList or nn.ParameterList:
        self.residuals = nn.ParameterList()
        
        n_channels = 3 if pixel_format == 'RGB' else 1
        
        for s in scales:
            # scaled-down size
            h_down = max(1, height // s)
            w_down = max(1, width  // s)
            
            if init == 'random':
                data = torch.rand(n_channels, h_down, w_down, device=device)
            else:
                data = torch.zeros(n_channels, h_down, w_down, device=device)
            
            # we wrap the data in an nn.Parameter so it’s learnable
            param = nn.Parameter(data)
            self.residuals.append(param)
        
        self.output_axes = ('n', 's', 'y', 'x')
        # you can adjust the default learning rate if you like
        self.lr = 0.1

    def decode_tensor(self):
        """
        Sums all learned multi-resolution components (upsampled to the
        final size) and clamps the result to [0,1].
        """
        # Start with zeros at the final resolution
        pixel_format = self.pixel_format
        n_channels = 3 if pixel_format == 'RGB' else 1
        
        # We'll accumulate in a single tensor
        accum = torch.zeros(1, n_channels, self.image_shape[1], self.image_shape[0],
                            device=self.residuals[0].device)
        
        for scale, param in zip(self.scales, self.residuals):
            # param is shape [C, h_down, w_down]
            # upsample to final size
            up = F.interpolate(
                param.unsqueeze(0),  # add batch dim
                size=(self.image_shape[1], self.image_shape[0]),
                mode='bilinear',  # or 'nearest' if you want chunkier style
                align_corners=True
            )
            accum = accum + up
        
        # Optionally apply something like (tanh(...) + 1)/2 if you prefer
        # for now, we just clamp
        image = clamp_with_grad(accum, 0, 1)
        return image  # shape [1, C, H, W]

    @torch.no_grad()
    def encode_image(self, pil_image):
        """
        Optional method for directly initializing
        the multi-resolution parameters from an existing image.
        A trivial approach: just downsample the given image for each scale
        and store in self.residuals.
        """
        pil_image = pil_image.convert(self.pixel_format)
        
        # convert to tensor, shape [C, H, W]
        full_tensor = TF.to_tensor(pil_image).to(self.residuals[0].device)
        
        for scale, param in zip(self.scales, self.residuals):
            h_down = param.shape[1]
            w_down = param.shape[2]
            
            # downsample the original image
            down = F.interpolate(
                full_tensor.unsqueeze(0),
                size=(h_down, w_down),
                mode='bilinear',
                align_corners=True
            )
            # Because each scale in this approach is effectively a residual,
            # you can either set it directly or treat it as an offset from zero
            self.residuals[self.scales.index(scale)].copy_(down[0])

    @torch.no_grad()
    def encode_random(self):
        """
        Overwrite each scale with random noise in [0,1].
        """
        for i, param in enumerate(self.residuals):
            param.uniform_(0,1)

    def update(self):
        """
        Hook called during training steps for clamping, etc.
        Here we can clamp each param in [0,1] if you like, or do nothing.
        """
        for param in self.residuals:
            param.data.clamp_(0, 1)