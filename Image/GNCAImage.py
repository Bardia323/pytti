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
    """Neural network for CA updates, guided by CLIP gradients"""
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Perception network - analyzes neighborhood
        self.perception = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        
        # Update network - produces changes guided by perception and gradients
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels + 3 + 3, hidden_channels, kernel_size=1),  # perception + state + gradients
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
            nn.Tanh(),  # Use tanh to avoid extreme values
        )
        
        # Initialize last layer weights to near zero for stability
        self.update[-2].weight.data.mul_(0.1)
        self.update[-2].bias.data.zero_()
        
    def forward(self, current_state, clip_gradients):
        """
        current_state: [B, 3, H, W] tensor of current RGB values
        clip_gradients: [B, 3, H, W] tensor of gradients from CLIP loss
        """
        # Perceive local patterns
        perception_features = self.perception(current_state)
        
        # Normalize clip gradients to avoid extreme values
        norm_grad = clip_gradients / (clip_gradients.std() + 1e-8) * 0.1
        
        # Concatenate features, current state, and gradients
        combined = torch.cat([perception_features, current_state, norm_grad], dim=1)
        
        # Compute update
        update_values = self.update(combined)
        
        # Apply soft update to current state
        new_state = current_state + update_values * 0.1
        
        return new_state.clamp(0, 1)

class GNCAImage(DifferentiableImage):
    """
    GNCA image with neural CA conditioned by CLIP gradients
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
        
        # Register buffers for optimization and CA state
        self.register_buffer('grad_buffer', torch.zeros_like(self.tensor))
        self.register_buffer('alive_mask', torch.zeros(1, height, width, device=DEVICE))
        
        # Initialize CA learnable params
        self.ca_optimizer = torch.optim.Adam(self.ca_model.parameters(), lr=1e-4)
        
        # Initialize with visible pattern
        self.reset_state()
    
    def reset_state(self):
        """Reset image state with a soft pattern"""
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
            
            # Create circular distance from center
            dist = torch.sqrt(x.pow(2) + y.pow(2)).clamp(0, 1)
            
            # Create a smooth pattern with gradients (not binary)
            r = torch.sin(dist * 3.14159 * 2) * 0.25 + 0.5
            g = torch.sin(dist * 3.14159 * 3) * 0.25 + 0.5  
            b = torch.sin(dist * 3.14159 * 4) * 0.25 + 0.5
            
            # Set tensor values - smooth gradients, not high contrast
            self.tensor[0] = r
            self.tensor[1] = g
            self.tensor[2] = b
            
            # Initialize alive mask in the center
            center_size = min(h, w) // 8
            cy, cx = h//2, w//2
            self.alive_mask[0, cy-center_size:cy+center_size, cx-center_size:cx+center_size] = 1.0
            
            # Smooth the alive mask
            self.alive_mask = F.avg_pool2d(
                F.interpolate(self.alive_mask.unsqueeze(0), scale_factor=0.25, mode='bilinear'),
                kernel_size=3, stride=1, padding=1
            ).squeeze(0)
            self.alive_mask = F.interpolate(self.alive_mask.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
    
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
        """Set from target image with natural colors"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize if needed
        c, h, w = self.tensor.shape
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',  # Use bilinear for smoother results
                align_corners=False
            ).squeeze(0)
        
        # Set tensor
        with torch.no_grad():
            self.tensor.copy_(img_tensor)
            
            # Create alive mask based on image content (non-binary)
            brightness = img_tensor.mean(dim=0, keepdim=True)
            
            # Compute edges
            edges = F.max_pool2d(brightness.unsqueeze(0), 3, stride=1, padding=1) - \
                    F.avg_pool2d(brightness.unsqueeze(0), 3, stride=1, padding=1)
            
            # Create smooth alive mask based on content
            content_mask = brightness + edges.squeeze(0) * 2
            content_mask = content_mask / content_mask.max()
            
            # Smooth mask
            self.alive_mask = F.avg_pool2d(
                F.max_pool2d(content_mask.unsqueeze(0), 3, stride=1, padding=1), 
                3, stride=1, padding=1
            ).squeeze(0)
    
    def encode_random(self):
        """Fill with random data - smooth gradient noise, not binary"""
        with torch.no_grad():
            # Create smooth noise
            for i in range(3):
                noise = torch.randn(self.tensor.shape[1]//4, self.tensor.shape[2]//4, device=self.tensor.device)
                noise = F.interpolate(
                    noise.unsqueeze(0).unsqueeze(0), 
                    size=self.tensor.shape[1:], 
                    mode='bicubic'
                ).squeeze(0)
                # Scale to [0.3, 0.7] range to avoid extremes
                self.tensor[i] = (noise * 0.2 + 0.5).clamp(0.3, 0.7)
            
            # Create smooth alive mask with connected regions
            noise = torch.randn(self.tensor.shape[1]//8, self.tensor.shape[2]//8, device=self.tensor.device)
            noise = F.interpolate(noise.unsqueeze(0).unsqueeze(0), size=self.tensor.shape[1:], mode='bicubic').squeeze(0)
            self.alive_mask = (noise > 0).float()
            # Smooth the mask
            self.alive_mask = F.avg_pool2d(
                F.max_pool2d(self.alive_mask.unsqueeze(0), 5, stride=1, padding=2),
                3, stride=1, padding=1
            ).squeeze(0)
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    def set_update_mode(self, mode):
        """Set animation style"""
        if mode in ['neural', 'none']:
            self.update_mode = mode
    
    @torch.no_grad()
    def _update_alive_mask(self):
        """Update the alive mask based on current state"""
        # Compute image-based alive signal
        image_signal = self.tensor.mean(dim=0, keepdim=True)
        
        # Compute image changes (movement)
        change_signal = (image_signal - image_signal.mean()).abs()
        
        # Create a combined signal where there's activity
        activity = (image_signal > 0.1) & (change_signal > change_signal.mean())
        
        # Grow the alive mask through diffusion
        next_alive = F.max_pool2d(
            self.alive_mask.unsqueeze(0), 
            kernel_size=3, stride=1, padding=1
        ).squeeze(0)
        
        # Combine current alive mask, diffusion, and activity
        self.alive_mask = (self.alive_mask * 0.8 + next_alive * 0.2 + activity.float() * 0.1).clamp(0, 1)
    
    @torch.no_grad()
    def update(self):
        """Neural CA update conditioned on CLIP gradients"""
        if self.update_mode == 'none':
            return
            
        # Store gradients from CLIP in the buffer if available
        if self.tensor.grad is not None:
            self.grad_buffer.copy_(self.tensor.grad)
            self.tensor.grad.zero_()
        
        for _ in range(self.steps_per_update):
            if self.update_mode == 'neural':
                # Run the neural CA model
                next_state = self.ca_model(
                    self.tensor.unsqueeze(0),
                    self.grad_buffer.unsqueeze(0)
                )
                
                # Apply alive mask - changes only happen in alive areas
                self.tensor.copy_(
                    self.tensor * (1 - self.alive_mask) + 
                    next_state.squeeze(0) * self.alive_mask
                )
                
                # Update alive mask based on activity
                self._update_alive_mask()
                
                # Train the CA model to follow gradients and maintain coherence
                # This happens occasionally to avoid slowing things down
                if torch.rand(1).item() < 0.1:  # 10% of updates
                    self._train_ca_step()
    
    def _train_ca_step(self):
        """Train the CA model to better follow CLIP guidance"""
        with torch.enable_grad():
            # Create a training sample
            x = self.tensor.unsqueeze(0).detach().clone().requires_grad_(True)
            
            # Run the model
            next_state = self.ca_model(x, self.grad_buffer.unsqueeze(0))
            
            # Compute losses:
            
            # 1. Follow CLIP gradients (align with text prompts)
            # We want the change to be in the direction of the CLIP gradients
            direction_loss = -F.cosine_similarity(
                next_state - x,
                self.grad_buffer.unsqueeze(0),
                dim=1
            ).mean()
            
            # 2. Maintain smoothness (avoid high contrast black/white)
            smoothness_loss = F.mse_loss(
                next_state[:, :, 1:, :],
                next_state[:, :, :-1, :]
            ) + F.mse_loss(
                next_state[:, :, :, 1:],
                next_state[:, :, :, :-1]
            )
            
            # 3. Maintain natural color distribution (avoid black/white goo)
            color_range_loss = torch.abs(
                next_state.mean(dim=(2, 3)) - torch.tensor([0.5, 0.5, 0.5], device=next_state.device)
            ).mean()
            
            # Combined loss
            loss = direction_loss + 0.2 * smoothness_loss + 0.1 * color_range_loss
            
            # Update model parameters
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