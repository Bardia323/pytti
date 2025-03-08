from pytti import *
from pytti.Image import DifferentiableImage
from pytti.Image.RGBImage import RGBImage  # Import the known working version
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

class CAModel(nn.Module):
    """Neural Cellular Automata model that's fully differentiable"""
    
    def __init__(self, channel_n=16, hidden_n=128, device=DEVICE):
        super().__init__()
        self.channel_n = channel_n
        
        # Perception kernels for edge detection (3×3×channel_n)
        self.register_buffer('identity', torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                                                    dtype=torch.float32).reshape(1, 1, 3, 3))
        self.register_buffer('sobel_x', torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0], 
                                              dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0)
        self.register_buffer('sobel_y', torch.tensor([1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0], 
                                              dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0)
        
        # Neural update network (fully differentiable)
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, 1)
        )
        
        # Initialize last layer to zeros for stability
        with torch.no_grad():
            self.update_net[-1].weight.zero_()
            self.update_net[-1].bias.zero_()
    
    def perceive(self, x):
        """Apply perception kernels to state"""
        batch, c, h, w = x.shape
        
        # Split into separate channels
        identity_out = []
        sobel_x_out = []
        sobel_y_out = []
        
        # Apply convolutions to each channel
        for i in range(c):
            # Extract one channel
            channel = x[:, i:i+1]
            
            # Apply the three perception kernels
            identity_out.append(F.conv2d(channel, self.identity, padding=1))
            sobel_x_out.append(F.conv2d(channel, self.sobel_x, padding=1))
            sobel_y_out.append(F.conv2d(channel, self.sobel_y, padding=1))
        
        # Concatenate results for all channels
        identity_out = torch.cat(identity_out, dim=1)
        sobel_x_out = torch.cat(sobel_x_out, dim=1)
        sobel_y_out = torch.cat(sobel_y_out, dim=1)
        
        # Stack all perception features
        perception = torch.cat([identity_out, sobel_x_out, sobel_y_out], dim=1)
        return perception
    
    def forward(self, x, step_size=0.1):
        """Perform one CA update step"""
        # Get perception features
        perception = self.perceive(x)
        
        # Apply update network
        update = self.update_net(perception) * step_size
        
        # Apply update (residual connection)
        x = x + update
        
        # Get alive mask (alpha channel if available, otherwise use mean)
        if x.shape[1] > 3:
            alive_mask = torch.sigmoid(x[:, 3:4]) > 0.1
        else:
            # For RGB-only, use mean brightness
            alive_mask = x.mean(dim=1, keepdim=True) > 0.1
            
        return x * alive_mask.float()

class GNCAImage(DifferentiableImage):
    """
    Growing Neural Cellular Automata image for Pytti
    """
    
    @vram_usage_mode('GNCA Image')
    def __init__(self, width, height, scale=1, channel_n=16, device=DEVICE):
        super().__init__(width, height)
        self.scale = scale
        self.channel_n = channel_n
        
        # Create tensor for RGB + hidden channels
        # First 4 channels are RGBA, rest are hidden state
        self.state = nn.Parameter(torch.zeros(1, channel_n, height, width, device=device))
        
        # Create the CA model
        self.ca_model = CAModel(channel_n, hidden_n=128, device=device)
        
        # CRITICAL: Match expected axes format
        self.output_axes = ('s', 'y', 'x')
        
        # Parameters
        self.update_mode = 'hybrid'  # 'none', 'ca', 'hybrid', 'adaptive'
        self.steps_per_update = 1
        self.ca_vs_clip = 0.5  # Balance between CA and CLIP (0=all CLIP, 1=all CA)
        self.step_size = 0.1
        
        # Initialize with seed
        self.reset_state()
    
    def reset_state(self):
        """Reset to initial seed state"""
        with torch.no_grad():
            # Clear state
            self.state.zero_()
            
            # Create a seed at the center
            h, w = self.state.shape[2:]
            cx, cy = w // 2, h // 2
            
            # Seed size
            seed_size = min(h, w) // 10
            seed_size = max(4, seed_size)  # At least 4 pixels
            
            # Set RGB to white
            self.state[0, 0:3, cy-seed_size//2:cy+seed_size//2, cx-seed_size//2:cx+seed_size//2] = 1.0
            
            # Set alpha to alive
            self.state[0, 3:4, cy-seed_size//2:cy+seed_size//2, cx-seed_size//2:cx+seed_size//2] = 1.0
            
            # Set some hidden state
            if self.channel_n > 4:
                self.state[0, 4:, cy-seed_size//2:cy+seed_size//2, cx-seed_size//2:cx+seed_size//2] = 0.1
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale, self.channel_n)
        with torch.no_grad():
            clone.state.copy_(self.state)
            clone.ca_model.load_state_dict(self.ca_model.state_dict())
            clone.update_mode = self.update_mode
            clone.steps_per_update = self.steps_per_update
            clone.ca_vs_clip = self.ca_vs_clip
            clone.step_size = self.step_size
        return clone
    
    def decode_tensor(self):
        """Returns RGB tensor for display"""
        # Extract RGB from state and apply alpha premultiplication
        rgb = self.state[0, 0:3]
        alpha = torch.sigmoid(self.state[0, 3:4])
        
        # Background color (white)
        bg_color = torch.ones_like(rgb)
        
        # Composite with white background
        composite = alpha * rgb + (1 - alpha) * bg_color
        
        return composite
    
    def get_image_tensor(self):
        """Return tensor for transformations"""
        # Just return RGB+hidden state without batch dimension
        return self.state[0]
    
    def set_image_tensor(self, tensor):
        """Set from tensor"""
        with torch.no_grad():
            self.state[0].copy_(tensor)
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Set from target image"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize if needed
        h, w = self.state.shape[2:]
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Set RGB channels
        with torch.no_grad():
            self.state[0, 0:3].copy_(img_tensor)
            
            # Set alpha based on brightness (bright areas = alive)
            brightness = img_tensor.mean(dim=0, keepdim=True)
            alpha = (brightness > 0.2).float() * 2.0 - 1.0  # Convert to logits
            self.state[0, 3:4].copy_(alpha)
            
            # Initialize hidden state in alive areas
            if self.channel_n > 4 and smart_encode:
                # Make sure alive_mask has correct shape: [h, w] not [1, h, w]
                alive_mask = (brightness > 0.2).float().squeeze(0)  # Remove extra dim
                target_device = self.state.device
                
                for i in range(4, self.channel_n):
                    # Different frequencies for different channels
                    freq = i / 2.0
                    # Create tensors on the correct device
                    y = torch.linspace(0, h-1, h, device=target_device).view(-1, 1).expand(-1, w) / h
                    x = torch.linspace(0, w-1, w, device=target_device).view(1, -1).expand(h, -1) / w
                    pattern = torch.sin(x * freq * np.pi) * torch.sin(y * freq * np.pi) * 0.5
                    # Make sure pattern and alive_mask have same shape before multiplying
                    self.state[0, i].copy_(pattern * alive_mask)
    
    def encode_random(self):
        """Initialize with random values"""
        with torch.no_grad():
            # Random RGB
            self.state[0, 0:3].uniform_(0, 1)
            
            # Random alive areas (sparse)
            alpha = torch.zeros_like(self.state[0, 3:4])
            alpha.bernoulli_(0.1)  # 10% alive cells
            self.state[0, 3:4] = alpha * 2.0 - 1.0  # Convert to logits
            
            # Random hidden state
            if self.channel_n > 4:
                self.state[0, 4:].uniform_(-0.1, 0.1)
    
    def set_update_mode(self, mode):
        """Set the update mode"""
        valid_modes = ['none', 'ca', 'hybrid', 'adaptive']
        if mode in valid_modes:
            self.update_mode = mode
    
    def set_steps_per_update(self, steps):
        """Set number of CA steps per update"""
        self.steps_per_update = max(1, steps)
    
    def set_ca_strength(self, strength):
        """Set balance between CA and CLIP (0-1)"""
        self.ca_vs_clip = max(0.0, min(1.0, strength))
    
    def set_step_size(self, step_size):
        """Set CA update step size"""
        self.step_size = max(0.01, min(1.0, step_size))
    
    def update(self):
        """Update image state"""
        # If no update, do nothing
        if self.update_mode == 'none':
            return
            
        # Save original state for hybrid mode
        if self.update_mode == 'hybrid':
            # Remember gradient contributions from CLIP
            original_state = self.state.clone()
            
        # Apply CA updates
        if self.update_mode in ['ca', 'hybrid', 'adaptive']:
            for _ in range(self.steps_per_update):
                self.state.copy_(self.ca_model(self.state, self.step_size))
        
        # For hybrid mode, blend CA result with original CLIP gradients
        if self.update_mode == 'hybrid':
            # Blend based on ca_vs_clip ratio
            with torch.no_grad():
                self.state.copy_(
                    self.ca_vs_clip * self.state + 
                    (1.0 - self.ca_vs_clip) * original_state
                )
        
        # For adaptive mode, adjust based on current state
        if self.update_mode == 'adaptive':
            # Adaptive behavior based on alive cell ratio
            with torch.no_grad():
                alive_ratio = torch.sigmoid(self.state[0, 3:4]).mean()
                
                # If too few alive cells, reduce CA influence
                if alive_ratio < 0.1:
                    self.ca_vs_clip = max(0.1, self.ca_vs_clip * 0.9)
                
                # If too many alive cells, increase CA influence
                elif alive_ratio > 0.5:
                    self.ca_vs_clip = min(0.9, self.ca_vs_clip * 1.1)
    
    # Required for compatibility
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 