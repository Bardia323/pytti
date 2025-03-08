from pytti import *
from pytti.Image import DifferentiableImage
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

class GNCAImage(DifferentiableImage):
    """
    Differentiable image format with GNCA-inspired animation.
    """
    @vram_usage_mode('GNCA Image')
    def __init__(self, width, height, scale=1, channel_n=16, device=DEVICE):
        super().__init__(width, height)
        self.scale = scale
        self.channel_n = channel_n
        
        # EXACTLY match PixelImage's structure
        self.value = nn.Parameter(torch.zeros(height, width, dtype=torch.float32, device=device))
        self.tensor = nn.Parameter(torch.zeros(channel_n, height, width, dtype=torch.float32, device=device))
        
        # CRITICAL: This is the exact output_axes from PixelImage
        self.output_axes = ('n', 's', 'y', 'x')
        
        # Animation parameters
        self.steps_per_update = 3
        
        # Initialize with seed
        self.reset_state()
    
    def reset_state(self):
        """Initialize state with patterns"""
        with torch.no_grad():
            h, w = self.value.shape
            
            # Initialize value
            y = torch.linspace(0, 1, h).view(-1, 1).expand(-1, w)
            x = torch.linspace(0, 1, w).view(1, -1).expand(h, -1)
            self.value.copy_((x + y) / 2)
            
            # Initialize "life" (higher values in center)
            cx, cy = w // 2, h // 2
            for i in range(4):  # First 4 channels as RGBA
                radius = min(w, h) // 4
                dist = ((torch.arange(h)[:, None] - cy) ** 2 + 
                        (torch.arange(w)[None, :] - cx) ** 2)
                mask = (dist < radius**2).float()
                
                if i < 3:  # RGB channels
                    self.tensor[i].copy_(mask * (i+1)/3)
                else:  # Alpha channel - life
                    self.tensor[i].copy_(mask)
    
    def clone(self):
        """Create a clone of this image"""
        clone = GNCAImage(
            self.image_shape[0] // self.scale, 
            self.image_shape[1] // self.scale, 
            self.scale, 
            self.channel_n
        )
        with torch.no_grad():
            clone.value.copy_(self.value)
            clone.tensor.copy_(self.tensor)
            clone.steps_per_update = self.steps_per_update
        return clone
    
    def get_image_tensor(self):
        """CRITICAL: Exactly match PixelImage's return format"""
        return torch.cat([self.value.unsqueeze(0), self.tensor])
    
    def set_image_tensor(self, tensor):
        """CRITICAL: Exactly match PixelImage's parameter format"""
        with torch.no_grad():
            self.value.copy_(tensor[0])
            self.tensor.copy_(tensor[1:])
    
    def decode_tensor(self):
        """Convert to RGB tensor"""
        # Extract RGBA from first 4 channels
        rgb = self.tensor[:3]
        alpha = self.tensor[3:4]
        
        # Combine for output (match PixelImage's logic)
        return rgb * alpha + (1.0 - alpha)
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Set from target image"""
        img_tensor = TF.to_tensor(pil_image).to(device)
        h, w = self.tensor.shape[1:]
        
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        with torch.no_grad():
            # Set RGB channels
            self.tensor[:3].copy_(img_tensor)
            
            # Create alpha mask based on brightness
            brightness = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
            self.tensor[3:4].copy_((brightness > 0.2).float())
            
            # Set value
            self.value.copy_(brightness)
    
    def set_steps_per_update(self, steps):
        """Animation speed"""
        self.steps_per_update = steps
    
    @torch.no_grad()
    def update(self):
        """Apply cellular automaton-inspired update"""
        for _ in range(self.steps_per_update):
            # Life mask based on alpha channel
            life_mask = self.tensor[3:4] > 0.1
            
            # Apply diffusion to RGB and alpha
            for i in range(4):
                # Simple diffusion via blur
                kernel_size = 3
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.tensor.device) / (kernel_size ** 2)
                channel = self.tensor[i:i+1].unsqueeze(0)
                blurred = F.conv2d(channel, kernel, padding=kernel_size//2).squeeze(0)
                
                # Add small noise for variation
                noise = torch.randn_like(blurred) * 0.01
                
                # Update with diffusion + noise
                self.tensor[i:i+1] = (blurred + noise).clamp(0, 1)
            
            # Apply life mask - only keep effects where cells are alive
            self.tensor[:3] *= life_mask
            
            # Update value parameter too
            brightness = 0.299 * self.tensor[0] + 0.587 * self.tensor[1] + 0.114 * self.tensor[2]
            self.value.copy_(brightness)
    
    @torch.no_grad()
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.decode_tensor()
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
    
    # Methods needed for compatibility with PixelImage
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 