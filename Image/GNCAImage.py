from pytti import *
from pytti.Image import DifferentiableImage
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import torch

class CAModel(nn.Module):
    """PyTorch implementation of the Growing Neural Cellular Automata model"""
    
    def __init__(self, channel_n=16, fire_rate=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        
        # Perception kernels
        self.register_buffer('identity', torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32).reshape(3, 3))
        self.register_buffer('dx', torch.tensor([1, 2, 1, 0, 0, 0, -1, -2, -1], dtype=torch.float32).reshape(3, 3) / 8.0)
        self.register_buffer('dy', torch.tensor([1, 0, -1, 2, 0, -2, 1, 0, -1], dtype=torch.float32).reshape(3, 3) / 8.0)
        
        # Update network
        self.dmodel = nn.Sequential(
            nn.Conv2d(channel_n * 3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, channel_n, 1, bias=False)
        )
        # Initialize last layer to zeros for stability
        self.dmodel[-1].weight.data.zero_()
    
    def perceive(self, x, angle=0.0):
        """Apply perception kernels to the input tensor"""
        batch_size, c, h, w = x.shape
        
        # Prepare kernels
        identity = self.identity
        dx, dy = self.dx, self.dy
        
        # Apply rotation if needed
        if angle != 0.0:
            c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            new_dx = c * dx - s * dy
            new_dy = s * dx + c * dy
            dx, dy = new_dx, new_dy
        
        # Stack kernels for all channels
        kernel = torch.stack([identity, dx, dy], dim=0)  # [3, 3, 3]
        kernel = kernel.reshape(3, 1, 3, 3).repeat(1, self.channel_n, 1, 1)  # [3, channel_n, 3, 3]
        kernel = kernel.reshape(3 * self.channel_n, 1, 3, 3)
        
        # Apply convolution
        x = x.reshape(batch_size * self.channel_n, 1, h, w)
        y = F.conv2d(x, kernel, padding=1, groups=self.channel_n)
        y = y.reshape(batch_size, 3 * self.channel_n, h, w)
        return y
    
    def get_living_mask(self, x):
        """Determine which cells are alive based on alpha channel"""
        alpha = x[:, 3:4, :, :]
        return F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
    
    def forward(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        """Run one step of the CA model"""
        pre_life_mask = self.get_living_mask(x)
        
        # Perceive neighborhood
        y = self.perceive(x, angle)
        
        # Compute update
        dx = self.dmodel(y) * step_size
        
        # Apply stochastic update
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = (torch.rand_like(x[:, :1]) <= fire_rate).float()
        x = x + dx * update_mask
        
        # Apply life mask
        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        
        return x * life_mask.float()

class GNCAImage(DifferentiableImage):
    """
    Differentiable image powered by Growing Neural Cellular Automata.
    Provides the same interface as PixelImage for compatibility.
    """
    
    @vram_usage_mode('Growing Neural CA Image')
    def __init__(self, width, height, scale=1, channel_n=16, device=DEVICE):
        super().__init__(width, height)
        self.scale = scale
        self.channel_n = channel_n
        
        # Create the CA model
        self.ca_model = CAModel(channel_n=channel_n).to(device)
        
        # CA state (RGBA + hidden channels)
        self.state = nn.Parameter(torch.zeros(1, channel_n, height, width, device=device))
        
        # For targeting specific images
        self.target_image = None
        self.use_target = False
        self.output_axes = ('n', 'c', 'y', 'x')
        self.steps_per_update = 1
        
        # Set the seed (center pixel)
        self.reset_state()
    
    def reset_state(self):
        """Reset the CA state to a single seed in the center"""
        with torch.no_grad():
            self.state.zero_()
            h, w = self.state.shape[2:]
            cx, cy = w // 2, h // 2
            self.state[0, 3:, cy-1:cy+1, cx-1:cx+1] = 1.0  # Set alive state for seed
    
    def clone(self):
        """Create a clone of this image"""
        width, height = self.image_shape
        clone = GNCAImage(width, height, self.scale, self.channel_n)
        with torch.no_grad():
            clone.state.copy_(self.state)
            clone.ca_model.load_state_dict(self.ca_model.state_dict())
            clone.steps_per_update = self.steps_per_update
            clone.use_target = self.use_target
            if self.target_image is not None:
                clone.target_image = self.target_image.clone()
        return clone
    
    def get_image_tensor(self):
        """Return the inner state tensor - required for transformations"""
        return self.state.squeeze(0)
    
    def set_image_tensor(self, tensor):
        """Set the inner state tensor - required for transformations"""
        with torch.no_grad():
            self.state.copy_(tensor.unsqueeze(0))
    
    def decode_tensor(self):
        """Convert the CA state to an RGB image tensor"""
        # Extract RGBA channels and convert to RGB
        rgba = self.state[0, :4]
        rgb = rgba[:3]
        alpha = rgba[3:4]
        
        # Premultiply RGB by alpha for compositing
        rgb_out = (1.0 - alpha) + rgb
        
        return rgb_out
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Set a target image for the CA to grow towards"""
        # Convert PIL image to tensor
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Resize to match our dimensions
        h, w = self.state.shape[2:]
        if img_tensor.shape[1] != h or img_tensor.shape[2] != w:
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Store as target
        self.target_image = img_tensor
        self.use_target = True
        
        # Reset CA state to a seed
        self.reset_state()
        
        # If smart_encode, pre-train the CA to approach the target
        if smart_encode:
            self.pretrain_ca(steps=100)
    
    def pretrain_ca(self, steps=100):
        """Pre-train the CA to grow towards the target image"""
        if not self.use_target or self.target_image is None:
            return
            
        # Create optimizer for CA parameters
        optimizer = optim.Adam(self.ca_model.parameters(), lr=2e-3)
        
        # Pre-training loop
        for _ in range(steps):
            # Run multiple CA steps
            x = self.state.clone()
            for _ in range(8):  # Run multiple steps for each optimization step
                x = self.ca_model(x)
            
            # Calculate loss against target
            rgb = x[0, :3]
            target_rgb = self.target_image[:3]
            loss = F.mse_loss(rgb, target_rgb)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update state
            with torch.no_grad():
                self.state.copy_(x.detach())
    
    @torch.no_grad()
    def update(self):
        """Run CA steps to update the state"""
        for _ in range(self.steps_per_update):
            self.state.copy_(self.ca_model(self.state))
    
    @torch.no_grad()
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.decode_tensor()
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
    
    def set_steps_per_update(self, steps):
        """Set how many CA steps to run per update"""
        self.steps_per_update = steps
    
    # Additional methods to maintain compatibility with PixelImage
    def image_loss(self):
        """Return empty list for compatibility"""
        return []
    
    def set_pallet_target(self, pil_image):
        """Compatibility method"""
        if pil_image is not None:
            self.encode_image(pil_image)
        else:
            self.use_target = False
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        """Compatibility method"""
        pass 