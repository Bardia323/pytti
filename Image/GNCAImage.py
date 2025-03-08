from pytti import *
from pytti.Image import DifferentiableImage
from pytti.Image.RGBImage import RGBImage  # Import the known working version
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

class GNCAImage(DifferentiableImage):
    """
    GNCA-inspired image class for Pytti
    """
    
    @vram_usage_mode('GNCA Image')
    def __init__(self, width, height, scale=1, **kwargs):
        super().__init__(width, height)
        self.scale = scale
        
        # Create tensor in RGB format
        self.tensor = nn.Parameter(torch.zeros(3, height, width, device=DEVICE))
        
        # CRITICAL: Match the exact axes format from DifferentiableImage
        self.output_axes = ('s', 'y', 'x')  # This matches what named_rearrange expects
        
        # Animation parameters
        self.steps_per_update = 1
        
        # Initialize with visible pattern
        self.reset_state()
    
    def reset_state(self):
        """Reset image state"""
        with torch.no_grad():
            # Clear tensor
            self.tensor.zero_()
            
            # Get dimensions
            c, h, w = self.tensor.shape
            
            # Create pattern - circular gradient
            y = torch.linspace(-1, 1, h).view(-1, 1).expand(-1, w)
            x = torch.linspace(-1, 1, w).view(1, -1).expand(h, -1)
            
            # Create circular distance from center
            dist = torch.sqrt(x.pow(2) + y.pow(2)).clamp(0, 1)
            
            # Create colorful pattern
            r = 0.5 + 0.5 * torch.cos(dist * 3.14159 * 3)
            g = 0.5 + 0.5 * torch.sin(dist * 3.14159 * 4)
            b = 0.5 + 0.5 * torch.cos(dist * 3.14159 * 5 + 3.14159/2)
            
            # Set tensor values
            self.tensor[0] = r  # Red
            self.tensor[1] = g  # Green
            self.tensor[2] = b  # Blue
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale)
        with torch.no_grad():
            clone.tensor.copy_(self.tensor)
            clone.steps_per_update = self.steps_per_update
        return clone
    
    def decode_tensor(self):
        """Returns tensor in the expected output format"""
        return self.tensor
    
    def get_image_tensor(self):
        """Return tensor for transformations"""
        return self.tensor
    
    def set_image_tensor(self, tensor):
        """Set from tensor"""
        with torch.no_grad():
            self.tensor.copy_(tensor)
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Set from target image"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize if needed
        c, h, w = self.tensor.shape
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Set tensor
        with torch.no_grad():
            self.tensor.copy_(img_tensor)
    
    def encode_random(self):
        """Fill with random data"""
        with torch.no_grad():
            self.tensor.uniform_().mul_(0.1).add_(0.5)
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    @torch.no_grad()
    def update(self):
        """Run animation effect"""
        for _ in range(self.steps_per_update):
            # Apply cellular automata-like update
            # 1. Get a blurred version of the image
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.tensor.device) / (kernel_size**2)
            
            # Apply convolution to each channel separately
            channels = []
            for i in range(3):
                channel = self.tensor[i:i+1].unsqueeze(0)  # Add batch dim
                blurred = F.conv2d(channel, kernel, padding=kernel_size//2)
                channels.append(blurred.squeeze(0))
            
            # Combine channels with slight variations to create movement
            r = channels[0] * 0.9 + channels[1] * 0.1
            g = channels[1] * 0.9 + channels[2] * 0.1
            b = channels[2] * 0.9 + channels[0] * 0.1
            
            # Add noise
            noise = torch.randn_like(self.tensor) * 0.01
            
            # Update tensor
            self.tensor[0:1] = (r + noise[0:1]).clamp(0, 1)
            self.tensor[1:2] = (g + noise[1:2]).clamp(0, 1)
            self.tensor[2:3] = (b + noise[2:3]).clamp(0, 1)
    
    # Required for compatibility
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 