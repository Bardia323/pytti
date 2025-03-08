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
    Improved GNCA image class with better animation modes
    """
    
    @vram_usage_mode('GNCA Image')
    def __init__(self, width, height, scale=1, **kwargs):
        super().__init__(width, height)
        self.scale = scale
        
        # Create tensor in RGB format
        self.tensor = nn.Parameter(torch.zeros(3, height, width, device=DEVICE))
        
        # Original tensor for reference (used in better CA)
        self.register_buffer('original', torch.zeros_like(self.tensor))
        
        # CRITICAL: Match the exact axes format
        self.output_axes = ('s', 'y', 'x')
        
        # Animation parameters
        self.steps_per_update = 1
        self.update_mode = 'none'  # 'none', 'ca', 'enhance', 'sharpen'
        self.ca_strength = 0.1  # How much CA affects the image (0-1)
        
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
            
            # Create patterns with sharp edges
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
            
            # Store original
            self.original.copy_(self.tensor)
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale)
        with torch.no_grad():
            clone.tensor.copy_(self.tensor)
            clone.original.copy_(self.original)
            clone.steps_per_update = self.steps_per_update
            clone.update_mode = self.update_mode
            clone.ca_strength = self.ca_strength
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
            # Enhance sharpness
            img_tensor = self._enhance_sharpness(img_tensor)
        
        # Set tensor
        with torch.no_grad():
            self.tensor.copy_(img_tensor)
            self.original.copy_(img_tensor)  # Store original for reference
    
    def _enhance_sharpness(self, img):
        """Enhance the sharpness of an image tensor"""
        # 1. Increase contrast
        mean = img.mean()
        img = (img - mean) * 1.3 + mean
        
        # 2. Apply sharpening kernel
        kernel = torch.tensor([[-1, -1, -1], 
                              [-1,  9, -1], 
                              [-1, -1, -1]], dtype=torch.float32, device=img.device) / 9.0
        kernel = kernel.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        
        channels = []
        for i in range(img.shape[0]):
            ch = img[i:i+1].unsqueeze(0)
            ch = F.conv2d(ch, kernel, padding=1)
            channels.append(ch.squeeze(0))
        
        img = torch.cat(channels, dim=0).clamp(0, 1)
        return img
    
    def encode_random(self):
        """Fill with random data - make it high contrast"""
        with torch.no_grad():
            # Generate random noise
            self.tensor.uniform_()
            
            # Apply quantization for sharper appearance
            self.tensor = (self.tensor > 0.5).float()
            
            # Store as original
            self.original.copy_(self.tensor)
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    def set_update_mode(self, mode):
        """Set animation style"""
        if mode in ['none', 'ca', 'enhance', 'sharpen']:
            self.update_mode = mode
    
    def set_ca_strength(self, strength):
        """Set how strongly the CA affects the image (0-1)"""
        self.ca_strength = max(0.0, min(1.0, strength))
    
    def _apply_conway_rules(self, channel):
        """Apply Conway's Game of Life rules"""
        # Count neighbors (including diagonals)
        kernel = torch.ones(1, 1, 3, 3, device=channel.device)
        kernel[0, 0, 1, 1] = 0  # Don't count the cell itself
        
        # Count alive neighbors
        neighbors = F.conv2d(channel, kernel, padding=1)
        
        # Conway's rules
        # 1. Any live cell with 2 or 3 live neighbors survives
        # 2. Any dead cell with exactly 3 live neighbors becomes alive
        # 3. All other cells die or stay dead
        
        # We threshold the channel to get binary live/dead state
        alive = (channel > 0.5).float()
        
        # Apply rules
        new_state = ((alive == 1) & ((neighbors == 2) | (neighbors == 3))) | ((alive == 0) & (neighbors == 3))
        return new_state.float()
    
    @torch.no_grad()
    def update(self):
        """Run animation effect with improved modes"""
        if self.update_mode == 'none':
            return  # Do nothing for static images
            
        for _ in range(self.steps_per_update):
            if self.update_mode == 'ca':
                # Better cellular automata that preserves image structure
                
                # Work with thresholded version for CA
                thresholded = (self.tensor > self.tensor.mean(dim=(1, 2), keepdim=True)).float()
                
                # Apply CA rules to each channel
                new_channels = []
                for i in range(3):
                    channel = thresholded[i:i+1].unsqueeze(0)  # Add batch dim
                    new_state = self._apply_conway_rules(channel)
                    new_channels.append(new_state.squeeze(0))
                
                # Blend with original based on strength
                for i in range(3):
                    # Use original colors but new patterns
                    self.tensor[i:i+1] = self.tensor[i:i+1] * (1 - self.ca_strength) + \
                                         new_channels[i] * self.tensor[i:i+1].mean() * self.ca_strength
                
            elif self.update_mode == 'enhance':
                # Subtle enhancement that improves the image over time
                
                # 1. Apply subtle sharpening
                kernel = torch.tensor([[-0.1, -0.1, -0.1], 
                                       [-0.1,  1.8, -0.1], 
                                       [-0.1, -0.1, -0.1]], dtype=torch.float32, device=self.tensor.device) / 1.0
                kernel = kernel.view(1, 1, 3, 3)
                
                channels = []
                for i in range(3):
                    channel = self.tensor[i:i+1].unsqueeze(0)
                    enhanced = F.conv2d(channel, kernel, padding=1)
                    channels.append(enhanced.squeeze(0))
                
                # Blend with current image
                for i in range(3):
                    self.tensor[i:i+1] = self.tensor[i:i+1] * 0.9 + channels[i] * 0.1
                
                # 2. Increase local contrast
                for i in range(3):
                    # Calculate local mean
                    local_mean = F.avg_pool2d(
                        self.tensor[i:i+1].unsqueeze(0), 
                        kernel_size=7, 
                        stride=1, 
                        padding=3
                    ).squeeze(0)
                    
                    # Increase contrast relative to local mean
                    self.tensor[i:i+1] = (self.tensor[i:i+1] - local_mean) * 1.1 + local_mean
                
                # Clamp values
                self.tensor.clamp_(0, 1)
                
            elif self.update_mode == 'sharpen':
                # Progressive sharpening that maintains structure
                
                # Create a version with increased edge contrast
                sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                       [-2.0, 0.0, 2.0], 
                                       [-1.0, 0.0, 1.0]], device=self.tensor.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1.0, -2.0, -1.0], 
                                       [0.0, 0.0, 0.0], 
                                       [1.0, 2.0, 1.0]], device=self.tensor.device).view(1, 1, 3, 3)
                
                edges = torch.zeros_like(self.tensor)
                
                for i in range(3):
                    channel = self.tensor[i:i+1].unsqueeze(0)
                    edges_x = F.conv2d(channel, sobel_x, padding=1)
                    edges_y = F.conv2d(channel, sobel_y, padding=1)
                    edge_mag = torch.sqrt(edges_x.pow(2) + edges_y.pow(2)).squeeze(0)
                    edges[i:i+1] = edge_mag
                
                # Create an edge-enhanced version
                edge_enhanced = self.tensor + edges * 0.2
                
                # Blend with original
                self.tensor.copy_(torch.lerp(self.tensor, edge_enhanced.clamp(0, 1), 0.1))
    
    # Required for compatibility
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 