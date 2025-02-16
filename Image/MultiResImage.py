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
    This class stores learnable residuals at several downscaled resolutions.
    In decoding, each residual is upsampled to the final output size, summed,
    passed through a tanh, and then mapped to [0, 1].
    """
    def __init__(self,
                 width,
                 height,
                 pixel_format='RGB',
                 scales=(1, 2, 4, 8, 16),
                 init='random',
                 device=DEVICE):
        """
        width, height: Final output resolution (in pixels).
        pixel_format : PIL image mode (e.g., 'RGB', 'L', etc.).
        scales       : A tuple of downscale factors for multi-resolution components.
        init         : Initialization mode ('random' or 'zeros').
        device       : Torch device to use.
        """
        super().__init__(width, height, pixel_format)
        self.scales = scales
        n_channels = 3 if pixel_format == 'RGB' else 1
        
        # Create a learnable parameter for each scale.
        self.residuals = nn.ParameterList()
        for s in scales:
            h_down = max(1, height // s)
            w_down = max(1, width  // s)
            if init == 'random':
                # Uniform noise in [-1, 1]
                data = 2 * torch.rand(n_channels, h_down, w_down, device=device) - 1
            else:
                data = torch.zeros(n_channels, h_down, w_down, device=device)
            self.residuals.append(nn.Parameter(data))
        
        self.output_axes = ('n', 's', 'y', 'x')
        self.lr = 0.1

    def decode_tensor(self):
        """
        Upsamples each scale to the final resolution, sums them,
        applies tanh, and maps the result from [-1, 1] to [0, 1].
        Returns a tensor with shape [1, C, height, width],
        where height and width match self.image_shape.
        """
        # self.image_shape is (width, height)
        width, height = self.image_shape
        n_channels = 3 if self.pixel_format == 'RGB' else 1
        device = self.residuals[0].device
        
        # Accumulator with shape [1, C, height, width]
        accum = torch.zeros(1, n_channels, height, width, device=device)
        for scale, param in zip(self.scales, self.residuals):
            # Upsample each parameter tensor to (height, width)
            up = F.interpolate(
                param.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=True
            )
            accum = accum + up
        # Apply tanh then map from [-1,1] to [0,1]
        image = (torch.tanh(accum) + 1) / 2
        return clamp_with_grad(image, 0, 1)

    def get_image_tensor(self):
        """
        Returns the decoded tensor with shape [C, height, width].
        This ensures downstream transforms (like zoom_3d) receive the expected dimensions.
        """
        return self.decode_tensor().squeeze(0)

    def set_image_tensor(self, tensor):
        """
        Sets the internal multi-resolution parameters so that decoding yields the provided tensor.
        Expects `tensor` to have shape [C, H, W] in the [0, 1] range.
        We invert the tanh mapping:
            output = (tanh(sum) + 1)/2   =>   sum = atanh(2*output-1)
        Since atanh(2*output-1) = 0.5 * log(output/(1-output)),
        we distribute the pre-activation value evenly among the scales.
        """
        # Ensure tensor is in the proper shape and range
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor with shape [C, H, W], got {tensor.shape}")
        eps = 1e-5
        tensor = tensor.clamp(eps, 1 - eps)
        # Invert mapping: compute pre-activation values
        # Correct inversion: pre = atanh(2*output-1) = 0.5 * log(output/(1-output))
        pre = 0.5 * torch.log(tensor / (1 - tensor))
        # Distribute pre-activation evenly among scales
        N = len(self.scales)
        for i, s in enumerate(self.scales):
            # Downsample pre to the size of this scale
            h_down = self.residuals[i].shape[1]
            w_down = self.residuals[i].shape[2]
            down = F.interpolate(pre.unsqueeze(0), size=(h_down, w_down), mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_(down.squeeze(0) / N)

    @torch.no_grad()
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """
        Initializes each scale's parameters by downsampling the given image.
        The image is converted to the desired pixel_format, then each scale
        is computed and mapped from [0, 1] to [-1, 1] to match the internal range.
        The 'smart_encode' parameter is accepted for compatibility but is not used.
        """
        pil_image = pil_image.convert(self.pixel_format)
        full_tensor = TF.to_tensor(pil_image).to(device)
        width, height = self.image_shape
        for i, s in enumerate(self.scales):
            h_down, w_down = self.residuals[i].shape[1], self.residuals[i].shape[2]
            down = F.interpolate(
                full_tensor.unsqueeze(0),
                size=(h_down, w_down),
                mode='bilinear',
                align_corners=True
            )
            self.residuals[i].copy_(down[0] * 2 - 1)

    @torch.no_grad()
    def encode_random(self):
        """
        Overwrites each scale with random noise uniformly drawn from [-1, 1].
        """
        for param in self.residuals:
            param.uniform_(-1, 1)

    def update(self):
        """
        Optional hook called during training to clamp parameters to the range [-1, 1].
        """
        for param in self.residuals:
            param.data.clamp_(-1, 1)
