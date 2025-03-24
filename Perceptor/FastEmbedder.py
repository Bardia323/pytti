from pytti import *
import pytti
import torch
from torch import nn

from .Embedder import HDMultiClipEmbedder
from ..fast_clip import fast_generate_cutouts, fast_spherical_dist_loss

class FastHDMultiClipEmbedder(HDMultiClipEmbedder):
    """
    Optimized version of HDMultiClipEmbedder using custom CUDA kernels.
    """
    
    def __init__(self, perceptors=None, cutn=40, cut_pow=1.5, padding=0.25, border_mode='clamp', noise_fac=0.1):
        """Initialize with the same parameters as the original embedder."""
        super().__init__(perceptors, cutn, cut_pow, padding, border_mode, noise_fac)
        
    def make_cutouts(self, input, side_x, side_y, cut_size, device=DEVICE):
        """
        Fast version of make_cutouts using CUDA kernel.
        
        Returns:
            cutouts: Tensor of cutouts [cutn, C, cut_size, cut_size]
            offsets: Tensor of offsets [cutn, 2]
            sizes: Tensor of sizes [cutn, 2]
        """
        max_size = min(side_x, side_y)
        
        # Generate random sizes using CUDA
        sizes = torch.zeros(self.cutn, device=device).normal_(mean=0.8, std=0.3)
        sizes = sizes.clamp(cut_size/max_size, 1.) ** self.cut_pow
        
        # Generate random offsets using CUDA
        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)
        
        # Calculate valid offset ranges
        offsetx_max = side_x - int(max_size * sizes.min()) + 1
        offsety_max = side_y - int(max_size * sizes.min()) + 1
        
        if self.border_mode == 'clamp':
            # Generate random offsets
            offsetx = torch.rand(self.cutn, device=device) * (offsetx_max + 2*paddingx) - paddingx
            offsety = torch.rand(self.cutn, device=device) * (offsety_max + 2*paddingy) - paddingy
            
            # Clamp offsets to valid range
            offsetx = torch.clamp(offsetx.floor().int(), 0, offsetx_max).float() / side_x
            offsety = torch.clamp(offsety.floor().int(), 0, offsety_max).float() / side_y
        else:
            # For other border modes
            px = torch.min(sizes * max_size, torch.tensor(paddingx, device=device)).floor().int()
            py = torch.min(sizes * max_size, torch.tensor(paddingy, device=device)).floor().int()
            
            offsetx = (torch.rand(self.cutn, device=device) * (offsetx_max + 2*px) - px).floor() / side_x
            offsety = (torch.rand(self.cutn, device=device) * (offsety_max + 2*py) - py).floor() / side_y
            
        # Stack offsets
        offsets = torch.stack([offsetx, offsety], dim=1)
        
        # Generate cutouts
        cutouts = fast_generate_cutouts(
            input, 
            sizes, 
            offsets, 
            cut_size, 
            self.cutn
        )
        
        # Add noise if needed
        if self.noise_fac:
            facs = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
            
        # Run augmentations
        cutouts = self.augs(cutouts)
        
        # Create size tensor in expected format
        sizes = torch.stack([sizes, sizes], dim=1).unsqueeze(1)
        offsets = offsets.unsqueeze(1)
        
        return cutouts, offsets, sizes
        
    def forward(self, diff_image, input=None, device=DEVICE):
        """
        Optimized forward pass using CUDA kernels.
        
        Args:
            diff_image: Image representation
            input: Optional pre-computed image tensor
            device: Device to run on
            
        Returns:
            image_embeds: Tensor of image embeddings
            all_offsets: Tensor of offsets
            all_sizes: Tensor of sizes
        """
        perceptors = self.perceptors
        side_x, side_y = diff_image.image_shape
        
        # Get input tensor
        if input is None:
            input = format_module(diff_image, self).to(device=device, memory_format=torch.channels_last)
        else:
            input = format_input(input, diff_image, self).to(device=device, memory_format=torch.channels_last)
        
        # Apply padding for non-clamp border modes
        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)
        if self.border_mode != 'clamp':
            input = F.pad(input, (paddingx, paddingx, paddingy, paddingy), 
                         mode=PADDING_MODES[self.border_mode])
        
        # Process each CLIP model
        image_embeds = []
        all_offsets = []
        all_sizes = []
        
        for cut_size, perceptor in zip(self.cut_sizes, perceptors):
            # Generate cutouts with CUDA
            cutouts, offsets, sizes = self.make_cutouts(input, side_x, side_y, cut_size, device)
            
            # Normalize and encode
            clip_in = normalize(cutouts)
            
            # Use torch.cuda.amp for faster encoding
            with torch.cuda.amp.autocast(enabled=True):
                embeddings = perceptor.encode_image(clip_in).float()
                
            image_embeds.append(embeddings.unsqueeze(0))
            all_offsets.append(offsets)
            all_sizes.append(sizes)
            
        return cat_with_pad(image_embeds), torch.stack(all_offsets), torch.stack(all_sizes)

# Create a faster version of the spherical distance loss
def fast_spherical_dist_loss_function(x, y):
    """
    Drop-in replacement for the spherical_dist_loss function.
    
    Args:
        x: First embedding tensor
        y: Second embedding tensor
        
    Returns:
        Spherical distance loss tensor
    """
    return fast_spherical_dist_loss(
        F.normalize(x, dim=-1),
        F.normalize(y, dim=-1)
    ) 