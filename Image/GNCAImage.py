                for j in range(self.pallet_size):
                    brightness = j / max(1, self.pallet_size - 1)
                    # Convert HSV to RGB - avoiding extreme saturation
                    h = hue
                    s = 0.6  # moderate saturation
                    v = 0.3 + brightness * 0.7  # avoid pure black
                    
                    # Simple HSV to RGB
                    c = v * s
                    x = c * (1 - abs((h * 6) % 2 - 1))
                    m = v - c
from pytti import *
            
            # Create a simple pattern
            h, w = self.value.shape
            y = torch.linspace(0, 1, h).view(-1, 1).expand(-1, w)
            x = torch.linspace(0, 1, w).view(1, -1).expand(h, -1)
            
            # Initialize value with distance pattern
            dist = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2).clamp(0, 1)
            self.value.copy_(dist)
            
            # Initialize palette weights
            self.tensor.zero_()
            center_size = min(h, w) // 4
            cy, cx = h//2, w//2
            
            # Assign different palette indices to different regions
            for i in range(min(4, self.n_pallets)):
                quadrant_y, quadrant_x = i // 2, i % 2
                y_start = cy - center_size + quadrant_y * center_size
                y_end = cy - center_size + (quadrant_y + 1) * center_size
                x_start = cx - center_size + quadrant_x * center_size
                x_end = cx - center_size + (quadrant_x + 1) * center_size
                
                y_start, y_end = max(0, y_start), min(h, y_end)
                x_start, x_end = max(0, x_start), min(w, x_end)
                
                if y_end > y_start and x_end > x_start:
                    self.tensor[i, y_start:y_end, x_start:x_end] = 2.0  # Less extreme value
            
            # Initialize alive mask in center
            self.alive_mask.zero_()
            self.alive_mask[0, cy-center_size:cy+center_size, cx-center_size:cx+center_size] = 1.0
            
            # Clear grad buffer
            self.grad_buffer.zero_()
    
    def set_pallet_target(self, pil_image):
        """Set target palette from image (like PixelImage)"""
        if pil_image is None:
            self.use_pallet_target = False
            return
        dummy = self.clone()
        dummy.use_pallet_target = False
        dummy.encode_image(pil_image)
        with torch.no_grad():
            self.pallet_target.copy_(dummy.sort_pallet())
            self.pallet.copy_(self.pallet_target)
            self.use_pallet_target = True
    
    @torch.no_grad()
    def lock_pallet(self, lock=True):
        """Lock the palette (like PixelImage)"""
        if lock:
            self.pallet_target.copy_(self.sort_pallet())
        self.use_pallet_target = lock
    
    def image_loss(self):
        """Return image-specific losses (like PixelImage)"""
        return [x for x in [self.hdr_loss, self.loss] if x is not None]
    
    def sort_pallet(self):
        """Sort the palette by brightness (like PixelImage)"""
        if self.use_pallet_target:
            return self.pallet_target
        pallet = (self.pallet / self.pallet_inertia).clamp(0, 1)
        # Calculate color norms for sorting
        magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
        color_norms = pallet.square().mul(magic_color).sum(dim=-1)
        # Optimized sorting using torch.gather
        pallet_indices = color_norms.argsort(dim=0)
        sorted_pallet = torch.gather(pallet, 0, pallet_indices.unsqueeze(-1).expand(-1, self.n_pallets, 3))
        return sorted_pallet
    
    def get_image_tensor(self):
        """Return tensor for transformations (like PixelImage)"""
        return torch.cat([self.value.unsqueeze(0), self.tensor])
    
    @torch.no_grad()
    def set_image_tensor(self, tensor):
        """Set tensor from external source (like PixelImage)"""
        self.value.copy_(tensor[0])
        self.tensor.copy_(tensor[1:])
    
    def decode_tensor(self):
        """Convert to RGB tensor in the expected format for CLIP"""
        width, height = self.image_shape
        pallet = self.sort_pallet()
        
        # Brightness values of pixels
        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)
        
        # Get palette weights using softmax for smooth blending
        pallet_weights = self.tensor.movedim(0, 2)
        pallet_weights = F.softmax(pallet_weights, dim=2).unsqueeze(-1)
        
        # Get colors based on brightness values
        colors_cont = pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        
        # Resize to final dimensions if needed
        if self.scale > 1:
            colors_cont = F.interpolate(
                colors_cont.permute(2, 0, 1).unsqueeze(0),
                size=(height, width),
                mode='nearest'
            ).squeeze(0)
        else:
            colors_cont = colors_cont.permute(2, 0, 1)
        
        # Ensure tensor is in the correct format (C,H,W) and properly clamped
        return colors_cont.clamp(0, 1)
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Encode from PIL image with palette (like PixelImage)"""
        width, height = self.image_shape
        
        # Resize and convert to tensor
        scale = self.scale
        color_ref = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        color_ref = TF.to_tensor(color_ref).to(device)
        
        # Calculate grayscale values for brightness
        with torch.no_grad():
            magic_color = torch.tensor([0.299, 0.587, 0.114], device=device).view(3, 1, 1)
            value_ref = (color_ref * magic_color).sum(dim=0)
            self.value.copy_(value_ref)
            
            # Initialize alive mask based on brightness
            self.alive_mask[0] = (value_ref > value_ref.mean() * 0.8).float()
        
        if smart_encode:
            # Skip optimization for simplicity and just assign random palette weights
            with torch.no_grad():
                # Find areas with different colors
                mean_color = color_ref.mean(dim=(1, 2), keepdim=True)
                diff_color = (color_ref - mean_color).abs().sum(dim=0)
                
                # Create different regions based on color differences
                regions = (diff_color > diff_color.mean()).float()
                
                # Assign palette weights based on regions
                k_means = min(4, self.n_pallets)
                h, w = self.tensor.shape[1:]
                for i in range(k_means):
                    # Create mask for this region (simple grid-based division)
                    region_y, region_x = i // 2, i % 2
                    y_start = region_y * (h // 2)
                    y_end = (region_y + 1) * (h // 2)
                    x_start = region_x * (w // 2)
                    x_end = (region_x + 1) * (w // 2)
                    
                    # Set weights higher in this region
                    self.tensor[i, y_start:y_end, x_start:x_end] = 5.0
    
    def encode_random(self, random_pallet=False):
        """Initialize with random values (like PixelImage)"""
        with torch.no_grad():
            self.value.uniform_()
            self.tensor.uniform_()
            if random_pallet:
                self.pallet.uniform_(to=self.pallet_inertia)
            
            # Initialize alive mask with random connected regions
            h, w = self.value.shape
            noise = torch.randn(h//4, w//4, device=self.value.device)
            noise = F.interpolate(noise.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic').squeeze(0)
            self.alive_mask[0] = (noise > noise.mean()).float()
            self.alive_mask = F.max_pool2d(self.alive_mask, 5, stride=1, padding=2)
    
    def set_steps_per_update(self, steps):
        """Set animation speed"""
        self.steps_per_update = steps
    
    def set_update_mode(self, mode):
        """Set animation style"""
        if mode in ['grow', 'none']:
            self.update_mode = mode
    
    @torch.no_grad()
    def update(self):
        """Update the image with CA growth"""
        # First update palette parameters like PixelImage
        self.pallet.copy_(self.pallet.clamp(0, self.pallet_inertia))
        self.value.copy_(self.value.clamp(0, 1))
        self.tensor.copy_(self.tensor.clamp(0, float('inf')))
        
        if self.update_mode == 'none':
            return
            
        # Store gradients if available
        if self.tensor.grad is not None:
            self.grad_buffer.copy_(self.tensor.grad)
            self.tensor.grad.zero_()
            
        # Run CA growth steps
        for _ in range(self.steps_per_update):
            if self.update_mode == 'grow':
                # 1. Save current alive mask
                prev_alive = self.alive_mask.clone()
                
                # 2. Calculate growth probability
                neighbors = F.avg_pool2d(self.alive_mask, kernel_size=3, stride=1, padding=1)
                growth_prob = neighbors * (1 - self.alive_mask)
                
                # 3. Grow mask with randomness
                rand_mask = torch.rand_like(growth_prob) < growth_prob * 0.3
                new_alive = self.alive_mask + rand_mask.float() * 0.5
                
                # 4. Never decrease alive mask
                self.alive_mask = torch.maximum(new_alive, prev_alive)
                
                # 5. Update palette weights (tensor) in alive regions
                # Get the current palette distribution
                weights = F.softmax(self.tensor, dim=0)
                
                # Apply diffusion to weights
                kernel = torch.ones(1, 1, 3, 3, device=self.tensor.device) / 9.0
                diffused_weights = torch.zeros_like(weights)
                
                for i in range(self.n_pallets):
                    channel = weights[i:i+1].unsqueeze(0)
                    diffused = F.conv2d(channel, kernel, padding=1)
                    diffused_weights[i] = diffused.squeeze(0)
                
                # Convert back to logits for the tensor
                diffused_tensor = torch.log(diffused_weights.clamp(min=1e-6))
                
                # Only update in alive regions
                alive_expanded = self.alive_mask.expand_as(self.tensor)
                self.tensor.copy_(
                    self.tensor * (1 - alive_expanded) + 
                    diffused_tensor * alive_expanded
                )
                
                # 6. Update brightness values with smoother gradients
                value_kernel = torch.ones(3, 3, device=self.value.device) / 9.0
                diffused_value = F.conv2d(
                    self.value.unsqueeze(0).unsqueeze(0), 
                    value_kernel.unsqueeze(0).unsqueeze(0), 
                    padding=1
                ).squeeze(0).squeeze(0)
                
                # Apply to alive regions
                alive_mask_2d = self.alive_mask.squeeze(0)
                self.value.copy_(
                    self.value * (1 - alive_mask_2d) + 
                    diffused_value * alive_mask_2d
                )
    
    @torch.no_grad()
    def render_value_image(self):
        """Render value image for visualization (like PixelImage)"""
        width, height = self.image_shape
        values = self.value.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 3)
        array = (values.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array).resize((width, height), Image.NEAREST)
    
    @torch.no_grad()
    def render_pallet(self):
        """Render palette for visualization (like PixelImage)"""
        pallet = self.sort_pallet()
        width, height = self.n_pallets * 16, self.pallet_size * 32
        array = (pallet.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array).resize((width, height), Image.NEAREST)
    
    @torch.no_grad()
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.decode_tensor()
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.decode_tensor()
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)
        pallet = self.sort_pallet()
        width, height = self.n_pallets * 16, self.pallet_size * 32
        array = (pallet.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array).resize((width, height), Image.NEAREST)
    
    @torch.no_grad()
    def decode_image(self):
        """Convert to PIL image for display"""
        tensor = self.decode_tensor()
        array = (tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)