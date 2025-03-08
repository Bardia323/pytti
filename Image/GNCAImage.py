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
    """Neural network for CA updates with stronger visual effects"""
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Perception network
        self.perception = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
        )
        
        # Update network - stronger updates
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels + 3 + 3, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
            nn.Tanh(),
        )
    
    def forward(self, current_state, clip_gradients, step_size=0.3):
        """
        current_state: [B, 3, H, W] tensor of current RGB values
        clip_gradients: [B, 3, H, W] tensor of gradients from CLIP loss
        step_size: How strong the updates should be (higher = more visible)
        """
        # Perceive local patterns
        perception_features = self.perception(current_state)
        
        # Normalize clip gradients
        norm_grad = clip_gradients / (clip_gradients.std() + 1e-8) * 0.2
        
        # Concatenate features, current state, and gradients
        combined = torch.cat([perception_features, current_state, norm_grad], dim=1)
        
        # Compute update with stronger effect
        update_values = self.update(combined)
        
        # Apply update with larger step size for visibility
        new_state = current_state + update_values * step_size
        
        return new_state.clamp(0, 1)

class GNCAImage(DifferentiableImage):
    """
    GNCA image with visibly different neural CA effects
    """
    
    @vram_usage_mode('Neural CA Image')
    def __init__(self, width, height, scale=1, **kwargs):
        super().__init__(width, height)
        self.scale = scale
        
        # Create tensor in RGB format
        self.tensor = nn.Parameter(torch.zeros(3, height, width, device=DEVICE))
        
        # Create the neural CA model
        self.ca_model = CAModel().to(DEVICE)
        
        # Match the expected output axes format
        self.output_axes = ('s', 'y', 'x')
        
        # Animation parameters
        self.steps_per_update = 1
        self.update_mode = 'neural'  # 'neural', 'none'
        self.debug_mode = False
        
        # Register buffers
        self.register_buffer('grad_buffer', torch.zeros_like(self.tensor))
        self.register_buffer('alive_mask', torch.zeros(1, height, width, device=DEVICE))
        self.register_buffer('update_strength', torch.tensor(0.3, device=DEVICE))
        
        # Initialize CA params
        self.ca_optimizer = torch.optim.Adam(self.ca_model.parameters(), lr=5e-4)
        
        # Initialize with visible pattern
        self.reset_state()
    
    def reset_state(self):
        """Reset image state with more distinct pattern"""
        with torch.no_grad():
            # Clear tensor and buffers
            self.tensor.zero_()
            self.grad_buffer.zero_()
            self.alive_mask.zero_()
            
            # Get dimensions
            c, h, w = self.tensor.shape
            
            # Create coordinate grid
            y = torch.linspace(-1, 1, h).view(-1, 1).expand(-1, w)
            x = torch.linspace(-1, 1, w).view(1, -1).expand(h, -1)
            
            # Create distance from center
            dist = torch.sqrt(x.pow(2) + y.pow(2)).clamp(0, 1)
            
            # Create pattern with more colors
            r = 0.5 + 0.4 * torch.sin(dist * 3.14159 * 3)
            g = 0.5 + 0.4 * torch.sin(dist * 3.14159 * 4 + 1.0)
            b = 0.5 + 0.4 * torch.sin(dist * 3.14159 * 5 + 2.0)
            
            # Set tensor values - colorful patterns
            self.tensor[0] = r
            self.tensor[1] = g
            self.tensor[2] = b
            
            # Initialize alive mask in the center with a larger area
            center_size = min(h, w) // 4
            cy, cx = h//2, w//2
            self.alive_mask[0, cy-center_size:cy+center_size, cx-center_size:cx+center_size] = 1.0
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale)
        with torch.no_grad():
            clone.tensor.copy_(self.tensor)
            clone.grad_buffer.copy_(self.grad_buffer)
            clone.alive_mask.copy_(self.alive_mask)
            clone.ca_model.load_state_dict(self.ca_model.state_dict())
            clone.steps_per_update = self.steps_per_update
            clone.update_mode = self.update_mode
            clone.debug_mode = self.debug_mode
            clone.update_strength.copy_(self.update_strength)
        return clone
    
    def decode_tensor(self):
        """Returns tensor in the expected output format"""
        if self.debug_mode:
            # In debug mode, show the alive mask as a green overlay
            debug_tensor = self.tensor.clone()
            debug_tensor[1] = torch.max(debug_tensor[1], self.alive_mask.squeeze(0) * 0.7)
            return debug_tensor
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
            
            # Initialize alive mask based on image content - more active areas
            brightness = img_tensor.mean(dim=0, keepdim=True)
            self.alive_mask[0] = (brightness > brightness.mean() * 0.8).float()
            
            # Grow the initial mask to cover more area
            self.alive_mask = F.max_pool2d(self.alive_mask, kernel_size=7, stride=1, padding=3)
    
    def encode_random(self):
        """Fill with random data with more distinct patterns"""
        with torch.no_grad():
            # Create pattern with visible structure
            for i in range(3):
                # Low-frequency noise
                noise = torch.randn(self.tensor.shape[1]//8, self.tensor.shape[2]//8, device=self.tensor.device)
                noise = F.interpolate(
                    noise.unsqueeze(0).unsqueeze(0), 
                    size=self.tensor.shape[1:], 
                    mode='bicubic'
                ).squeeze(0)
                self.tensor[i] = (noise * 0.3 + 0.5).clamp(0, 1)
            
            # Create alive mask with larger connected regions
            noise = torch.randn(self.tensor.shape[1]//4, self.tensor.shape[2]//4, device=self.tensor.device)
            noise = F.interpolate(noise.unsqueeze(0).unsqueeze(0), size=self.tensor.shape[1:], mode='bicubic').squeeze(0)
            self.alive_mask[0] = (noise > noise.mean() * 0.5).float()
            
            # Grow the mask for more coverage
            self.alive_mask = F.max_pool2d(self.alive_mask, kernel_size=5, stride=1, padding=2)
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    def set_update_mode(self, mode):
        """Set animation style"""
        if mode in ['neural', 'none']:
            self.update_mode = mode
    
    def set_debug_mode(self, debug):
        """Toggle debug visualization"""
        self.debug_mode = debug
    
    def set_update_strength(self, strength):
        """Set how strong the neural updates should be"""
        with torch.no_grad():
            self.update_strength.copy_(torch.tensor(strength, device=self.update_strength.device))
    
    @torch.no_grad()
    def _grow_alive_mask(self):
        """Grow the alive mask more aggressively"""
        # Get current alive pixels
        current_alive = self.alive_mask > 0.5
        
        # Find edges of alive regions (where growth happens)
        dilated = F.max_pool2d(self.alive_mask, kernel_size=5, stride=1, padding=2)
        edge_mask = (dilated > 0.5) & ~current_alive
        
        # Add probability of growth at edges
        growth_prob = torch.rand_like(self.alive_mask) < 0.3
        new_alive = edge_mask & growth_prob
        
        # Update mask with new growth
        self.alive_mask = torch.maximum(
            self.alive_mask,
            new_alive.float() * 0.7  # New areas start at 70% alive
        )
    
    @torch.no_grad()
    def update(self):
        """Neural CA update with more visible effects"""
        if self.update_mode == 'none':
            return
            
        # Store gradients from CLIP in the buffer
        if self.tensor.grad is not None:
            self.grad_buffer.copy_(self.tensor.grad)
            self.tensor.grad.zero_()
        
        # Ensure we have some gradient signal (use random if none)
        if self.grad_buffer.abs().sum() < 1e-6:
            self.grad_buffer.copy_(torch.randn_like(self.grad_buffer) * 0.01)
        
        for _ in range(self.steps_per_update):
            if self.update_mode == 'neural':
                # Run the neural CA model with current strength
                next_state = self.ca_model(
                    self.tensor.unsqueeze(0),
                    self.grad_buffer.unsqueeze(0),
                    float(self.update_strength)
                )
                
                # Apply updated state only in alive areas
                alive_mask_expanded = self.alive_mask.expand_as(self.tensor)
                self.tensor.copy_(
                    self.tensor * (1 - alive_mask_expanded) + 
                    next_state.squeeze(0) * alive_mask_expanded
                )
                
                # Grow the alive mask for more visible spreading
                self._grow_alive_mask()
                
                # Train the CA model to maintain coherence (occasionally)
                if torch.rand(1).item() < 0.1:
                    self._train_ca_step()
    
    def _train_ca_step(self):
        """Train CA model to create more interesting patterns"""
        with torch.enable_grad():
            # Sample input state
            x = self.tensor.unsqueeze(0).detach().clone().requires_grad_(True)
            
            # Run model
            next_state = self.ca_model(x, self.grad_buffer.unsqueeze(0), float(self.update_strength))
            
            # Calculate change
            diff = next_state - x
            
            # Create target that follows CLIP gradients but maintains structure
            target_diff = self.grad_buffer.unsqueeze(0) * 0.1
            
            # Loss to follow CLIP guidance
            direction_loss = F.mse_loss(diff, target_diff)
            
            # Loss to create visually interesting patterns
            pattern_loss = 0.1 * (
                # Encourage some local contrast (but not too much)
                - F.mse_loss(next_state[:, :, 1:, :], next_state[:, :, :-1, :]) * 0.5 +
                # But discourage harsh transitions
                F.smooth_l1_loss(next_state[:, :, 1:, :], next_state[:, :, :-1, :]) * 2.0
            )
            
            # Loss to prevent color extremes
            color_balance_loss = 0.2 * torch.abs(next_state.mean(dim=(2, 3)) - 0.5).mean()
            
            # Combined loss
            loss = direction_loss + pattern_loss + color_balance_loss
            
            # Update model
            self.ca_optimizer.zero_grad()
            loss.backward()
            self.ca_optimizer.step()
    
    # Required for compatibility
    def image_loss(self):
        return []
    
    def set_pallet_target(self, pil_image):
        if pil_image is not None:
            self.encode_image(pil_image)
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        pass 