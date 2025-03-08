from pytti import *
from pytti.Image import DifferentiableImage
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import torch

class GNCAImage(DifferentiableImage):
    """
    Simplified GNCA-based image for Pytti
    """
    
    @vram_usage_mode('Growing Neural CA Image')
    def __init__(self, width, height, scale=1, channel_n=16, device=DEVICE):
        super().__init__(width, height)
        self.scale = scale
        self.channel_n = channel_n
        
        # Create simple RGB image tensor instead of complex CA
        self.tensor = nn.Parameter(torch.zeros(3, height, width, device=device))
        
        # Match PixelImage's tensor structure (value + channels)
        self.value = nn.Parameter(torch.zeros(height, width, device=device))
        
        # Keep PixelImage's exact output_axes
        self.output_axes = ('n', 's', 'y', 'x')
        
        # Initialize with some visible content
        self.reset_state()
    
    def reset_state(self):
        """Initialize the image with a visible pattern"""
        with torch.no_grad():
            self.tensor.zero_()
            self.value.zero_()
            h, w = self.tensor.shape[1:]
            
            # Create a simple pattern - gradient background
            y_coords = torch.linspace(0, 1, h).view(-1, 1).repeat(1, w)
            x_coords = torch.linspace(0, 1, w).view(1, -1).repeat(h, 1)
            
            # Set RGB channels with different patterns
            self.tensor[0] = y_coords  # Red
            self.tensor[1] = x_coords  # Green
            self.tensor[2] = (y_coords + x_coords) / 2  # Blue
            
            # Set value for compatibility
            self.value.copy_(y_coords)
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale, self.channel_n)
        with torch.no_grad():
            clone.tensor.copy_(self.tensor)
            clone.value.copy_(self.value)
        return clone
    
    def get_image_tensor(self):
        """Return tensor in PixelImage format: [value, tensor]"""
        return torch.cat([self.value.unsqueeze(0), self.tensor])
    
    def set_image_tensor(self, tensor):
        """Set tensor from PixelImage format: [value, tensor]"""
        with torch.no_grad():
            self.value.copy_(tensor[0])
            self.tensor.copy_(tensor[1:])
    
    def decode_tensor(self):
        """Convert to RGB tensor"""
        return self.tensor
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Set from target image"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize to match our dimensions
        h, w = self.tensor.shape[1:]
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Set tensor directly
        with torch.no_grad():
            self.tensor.copy_(img_tensor)
            # Also set value tensor to grayscale for compatibility
            self.value.copy_((0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]))
    
    @torch.no_grad()
    def update(self):
        """Minimal update for animation"""
        # Apply a simple blur for some movement
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.tensor.device) / (kernel_size ** 2)
        
        # Apply blur separately to each channel
        for i in range(3):
            channel = self.tensor[i:i+1].unsqueeze(0)
            blurred = F.conv2d(channel, kernel, padding=kernel_size//2)
            self.tensor[i:i+1] = blurred.squeeze(0)
    
    @torch.no_grad()
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.tensor
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
    
    # PixelImage compatibility methods
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 