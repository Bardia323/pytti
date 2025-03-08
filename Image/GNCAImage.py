from pytti import *
from pytti.Image import DifferentiableImage
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

class GNCAImage(DifferentiableImage):
    """
    Very simple image class compatible with Pytti.
    Follows RGBImage's pattern which is known to work.
    """
    
    @vram_usage_mode('GNCA Image')
    def __init__(self, width, height, scale=1, channel_n=16, device=DEVICE):
        super().__init__(width, height)
        self.scale = scale
        
        # Simple tensor with batch dimension - crucial for channels_last format!
        # Using NCHW format: [1, 3, height, width]
        self.tensor = nn.Parameter(torch.zeros(1, 3, height, width, device=device))
        
        # Initialize with gradients
        self.reset_state()
        
        # Animation parameters
        self.steps_per_update = 1
    
    def set_steps_per_update(self, steps):
        """Configure animation speed"""
        self.steps_per_update = steps
    
    def reset_state(self):
        """Initialize with a gradient pattern"""
        with torch.no_grad():
            h, w = self.tensor.shape[2:]
            
            # Create a simple gradient pattern
            y_coords = torch.linspace(0, 1, h).view(1, 1, -1, 1).expand(1, 3, -1, w)
            x_coords = torch.linspace(0, 1, w).view(1, 1, 1, -1).expand(1, 3, h, -1)
            
            # Set RGB channels
            self.tensor[0, 0] = x_coords[0, 0]  # Red - horizontal gradient
            self.tensor[0, 1] = y_coords[0, 0]  # Green - vertical gradient
            self.tensor[0, 2] = 1 - ((x_coords[0, 0] + y_coords[0, 0])/2)  # Blue - diagonal gradient
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale)
        with torch.no_grad():
            clone.tensor.copy_(self.tensor)
            clone.steps_per_update = self.steps_per_update
        return clone
    
    def decode_tensor(self):
        """Convert to RGB tensor"""
        return self.tensor[0]  # Return [C, H, W] format
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Set from target image"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize to match our dimensions
        h, w = self.tensor.shape[2:]
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Set tensor directly - keeping the batch dimension
        with torch.no_grad():
            self.tensor[0].copy_(img_tensor)
    
    @torch.no_grad()
    def update(self):
        """Simple update for animation - just a subtle blur"""
        for _ in range(self.steps_per_update):
            # Apply a simple blur
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.tensor.device) / (kernel_size ** 2)
            
            # Process each channel
            for i in range(3):
                channel = self.tensor[:, i:i+1]
                blurred = F.conv2d(channel, kernel, padding=kernel_size//2)
                # Add a small perturbation
                noise = torch.randn_like(blurred) * 0.01
                self.tensor[:, i:i+1] = (blurred + noise).clamp(0, 1)
    
    @torch.no_grad()
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.tensor[0]  # Remove batch dimension
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
    
    # Methods needed for compatibility
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 