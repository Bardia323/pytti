from pytti import *
from pytti.Image import DifferentiableImage
from pytti.Image.RGBImage import RGBImage  # Import the known working version
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

class GNCAImage(RGBImage):
    """
    GNCA image based on RGBImage
    Uses simple blur-based animation as a placeholder for full CA
    """
    
    @vram_usage_mode('GNCA Image')
    def __init__(self, width, height, scale=1, device=DEVICE, **kwargs):
        # Call parent exactly as in RGBImage
        super().__init__(width, height, scale, device)
        
        # Animation speed parameter
        self.steps_per_update = 1
        
        # Create initial pattern
        self.init_pattern()
    
    def init_pattern(self):
        """Create initial pattern"""
        with torch.no_grad():
            # Get actual tensor dimensions (NCHW format)
            _, _, h, w = self.tensor.shape
            
            # Create simple pattern
            for i in range(w):
                for j in range(h):
                    x, y = i/w, j/h
                    self.tensor[0, 0, j, i] = ((x + y)/2) % 1.0  # Red
                    self.tensor[0, 1, j, i] = x % 1.0  # Green
                    self.tensor[0, 2, j, i] = y % 1.0  # Blue
    
    def reset_state(self):
        """Compatibility with initialization code"""
        self.init_pattern()
    
    def set_steps_per_update(self, steps):
        """Animation speed setting"""
        self.steps_per_update = steps
    
    @torch.no_grad()
    def update(self):
        """Simple animation - blur and add noise"""
        for _ in range(self.steps_per_update):
            # Make a copy for the update
            updated = self.tensor.clone()
            
            # Apply simple 3x3 blur
            for c in range(3):  # For each RGB channel
                for i in range(1, self.tensor.shape[3]-1):
                    for j in range(1, self.tensor.shape[2]-1):
                        # Simple 3x3 average
                        neighbors_sum = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                neighbors_sum += self.tensor[0, c, j+dj, i+di]
                        
                        # Update with blur + small random change
                        updated[0, c, j, i] = (neighbors_sum / 9.0) + torch.rand(1).item() * 0.02 - 0.01
            
            # Update tensor with clipping
            self.tensor.copy_(updated.clamp(0, 1))
    
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