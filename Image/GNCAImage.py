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
    GNCA-inspired image class for Pytti with sharper output
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
        self.update_mode = 'none'  # 'none', 'ca', 'edge'
        
        # Initialize with visible pattern
        self.reset_state()
    
    def reset_state(self):
        """Reset image state with a sharper pattern"""
        with torch.no_grad():
            # Clear tensor
            self.tensor.zero_()
            
            # Get dimensions
            c, h, w = self.tensor.shape
            
            # Create pattern with sharper edges
            y = torch.linspace(-1, 1, h).view(-1, 1).expand(-1, w)
            x = torch.linspace(-1, 1, w).view(1, -1).expand(h, -1)
            
            # Create circular distance from center
            dist = torch.sqrt(x.pow(2) + y.pow(2)).clamp(0, 1)
            
            # Create patterns with sharp edges (using step functions)
            stripes_x = (torch.sin(x * 10 * np.pi) > 0).float()
            stripes_y = (torch.sin(y * 10 * np.pi) > 0).float()
            circles = ((dist * 10) % 1.0 > 0.5).float()
            
            # Combine for interesting sharp pattern
            r = stripes_x * 0.8 + 0.2
            g = stripes_y * 0.8 + 0.2
            b = circles * 0.8 + 0.2
            
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
            clone.update_mode = self.update_mode
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
        """Set from target image with enhanced sharpness"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize if needed - use NEAREST for sharper resizing
        c, h, w = self.tensor.shape
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='nearest'
            ).squeeze(0)
        
        # Optional: Enhance contrast for even sharper look
        if smart_encode:
            # Simple contrast enhancement
            mean = img_tensor.mean()
            img_tensor = (img_tensor - mean) * 1.2 + mean
            img_tensor = img_tensor.clamp(0, 1)
        
        # Set tensor
        with torch.no_grad():
            self.tensor.copy_(img_tensor)
    
    def encode_random(self):
        """Fill with random data - make it high contrast"""
        with torch.no_grad():
            # Binary noise for sharper appearance
            self.tensor.bernoulli_(0.5)
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    def set_update_mode(self, mode):
        """Set animation style"""
        if mode in ['none', 'ca', 'edge']:
            self.update_mode = mode
    
    @torch.no_grad()
    def update(self):
        """Run animation effect with enhanced sharpness"""
        if self.update_mode == 'none':
            return  # Do nothing for static images
            
        for _ in range(self.steps_per_update):
            if self.update_mode == 'ca':
                # Apply cellular automata-like update
                kernel_size = 3
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.tensor.device) / (kernel_size**2)
                
                # Apply convolution to each channel separately
                channels = []
                for i in range(3):
                    channel = self.tensor[i:i+1].unsqueeze(0)  # Add batch dim
                    blurred = F.conv2d(channel, kernel, padding=kernel_size//2)
                    channels.append(blurred.squeeze(0))
                
                # Make sharper edges by applying a step function
                for i in range(3):
                    # Threshold the blurred values for sharp transitions
                    threshold = channels[i].mean()
                    # Add a small random noise to prevent static patterns
                    channels[i] = ((channels[i] + torch.randn_like(channels[i]) * 0.05) > threshold).float()
                
                # Update tensor
                self.tensor[0:1] = channels[0]
                self.tensor[1:2] = channels[1]
                self.tensor[2:3] = channels[2]
                
            elif self.update_mode == 'edge':
                # Edge detection for sharp transitions
                # Sobel filters for edge detection - ensure proper type
                sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], 
                                      device=self.tensor.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], 
                                      device=self.tensor.device).view(1, 1, 3, 3)
                
                for i in range(3):
                    channel = self.tensor[i:i+1].unsqueeze(0)
                    edges_x = F.conv2d(channel, sobel_x, padding=1)
                    edges_y = F.conv2d(channel, sobel_y, padding=1)
                    edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2))
                    
                    # Threshold edges for binary edge map
                    edge_threshold = edges.mean() * 2
                    edge_mask = (edges > edge_threshold).float()
                    
                    # Update based on edges - invert regions with edges
                    self.tensor[i:i+1] = self.tensor[i:i+1] * (1 - edge_mask.squeeze(0)) + \
                                        (1 - self.tensor[i:i+1]) * edge_mask.squeeze(0)
    
    # Required for compatibility
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 