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
        
        # Create tensor with batch dimension [batch, channels, height, width]
        self.tensor = nn.Parameter(torch.zeros(1, 3, height, width, device=DEVICE))
        
        # CRITICAL: Set output_axes to match what the embedder expects
        self.output_axes = ('n', 'c', 'y', 'x')
        
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
            _, _, h, w = self.tensor.shape
            
            # Create pattern - circular gradient
            y = torch.linspace(-1, 1, h).view(-1, 1).expand(-1, w)
            x = torch.linspace(-1, 1, w).view(1, -1).expand(h, -1)
            
            # Create circular distance from center
            dist = torch.sqrt(x.pow(2) + y.pow(2))
            
            # Create interesting patterns
            r = torch.sin(dist * 6.28) * 0.5 + 0.5  # Red channel
            g = torch.sin(dist * 6.28) * 0.5 + 0.5  # Green channel
            b = torch.sin(dist * 6.28) * 0.5 + 0.5  # Blue channel
            
            # Assign to tensor (keeping NCHW format)
            self.tensor[0, 0] = r
            self.tensor[0, 1] = g
            self.tensor[0, 2] = b
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    @torch.no_grad()
    def update(self):
        """Add subtle animation"""
        # Additional animation effects
        for _ in range(self.steps_per_update):
            # Apply a simple blur for movement
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.tensor.device) / (kernel_size ** 2)
            
            # Process each channel
            for i in range(3):
                # Ensure correct shape with batch dimension
                channel = self.tensor[:, i:i+1]
                # Apply blur
                blurred = F.conv2d(channel, kernel, padding=kernel_size//2)
                # Add noise
                noise = torch.randn_like(blurred) * 0.01
                # Update tensor
                self.tensor[:, i:i+1] = (blurred + noise).clamp(0, 1)
    
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