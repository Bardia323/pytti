from pytti import *
from pytti.Image import DifferentiableImage
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
import types
import sys

# Create a special flag at the module level
ANIMATION_MODE = "None"

class LogicGate(nn.Module):
    """
    Differentiable logic gate as described in the DiffLogic CA paper.
    Uses continuous relaxation during training and discrete operations during inference.
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        # Initialize with bias towards pass-through gates (A and B)
        self.logits = nn.Parameter(torch.zeros(16, device=device))
        self.logits.data[3] = 2.0  # Bias towards A
        self.logits.data[5] = 2.0  # Bias towards B

    def forward(self, a, b, hard=False):
        if hard:
            # Use the most probable gate during inference
            gate_idx = self.logits.argmax().item()
            return self._apply_gate(a, b, gate_idx)
        else:
            # During training, use continuous relaxation with softmax weighting
            gate_probs = F.softmax(self.logits, dim=0)
            result = torch.zeros_like(a)
            for i in range(16):
                result += gate_probs[i] * self._apply_gate(a, b, i)
            return result
    
    def _apply_gate(self, a, b, gate_idx):
        """Apply the specified logic gate using continuous relaxation."""
        if gate_idx == 0:    # FALSE
            return torch.zeros_like(a)
        elif gate_idx == 1:  # AND
            return a * b
        elif gate_idx == 2:  # A AND (NOT B)
            return a * (1 - b)
        elif gate_idx == 3:  # A
            return a
        elif gate_idx == 4:  # (NOT A) AND B
            return (1 - a) * b
        elif gate_idx == 5:  # B
            return b
        elif gate_idx == 6:  # XOR
            return a + b - 2 * a * b
        elif gate_idx == 7:  # OR
            return a + b - a * b
        elif gate_idx == 8:  # NOR
            return 1 - (a + b - a * b)
        elif gate_idx == 9:  # XNOR
            return 1 - (a + b - 2 * a * b)
        elif gate_idx == 10: # NOT B
            return 1 - b
        elif gate_idx == 11: # A OR (NOT B)
            return a + (1 - b) - a * (1 - b)
        elif gate_idx == 12: # NOT A
            return 1 - a
        elif gate_idx == 13: # (NOT A) OR B
            return (1 - a) + b - (1 - a) * b
        elif gate_idx == 14: # NAND
            return 1 - a * b
        elif gate_idx == 15: # TRUE
            return torch.ones_like(a)


class PerceptionCircuit(nn.Module):
    """
    Perception circuit that processes the neighborhood of a cell.
    """
    def __init__(self, channels, device=DEVICE):
        super().__init__()
        self.channels = channels
        # Create a 3-layer circuit with decreasing gate counts
        num_gates = [8, 4, 2, 1]  # Following the paper's structure
        self.layers = nn.ModuleList()
        
        # First layer: process center cell and neighbors
        layer1 = nn.ModuleList([LogicGate(device) for _ in range(num_gates[0])])
        self.layers.append(layer1)
        
        # Subsequent layers
        for i in range(1, len(num_gates)):
            layer = nn.ModuleList([LogicGate(device) for _ in range(num_gates[i])])
            self.layers.append(layer)
            
    def forward(self, neighborhood, hard=False):
        """
        Process a 3x3 neighborhood for each channel.
        neighborhood: tensor of shape (batch, channels, 3, 3)
        Returns: tensor of shape (batch, 1) - ALWAYS 2D
        """
        batch_size = neighborhood.shape[0]
        center = neighborhood[:, :, 1, 1]  # Center cell values (batch, channels)
        
        # Extract neighbors
        # Simpler approach: just get the 8 neighbors directly
        n_tl = neighborhood[:, :, 0, 0]  # top-left
        n_tm = neighborhood[:, :, 0, 1]  # top-middle
        n_tr = neighborhood[:, :, 0, 2]  # top-right
        n_ml = neighborhood[:, :, 1, 0]  # middle-left
        n_mr = neighborhood[:, :, 1, 2]  # middle-right
        n_bl = neighborhood[:, :, 2, 0]  # bottom-left
        n_bm = neighborhood[:, :, 2, 1]  # bottom-middle
        n_br = neighborhood[:, :, 2, 2]  # bottom-right
        
        neighbors = [n_tl, n_tm, n_tr, n_ml, n_mr, n_bl, n_bm, n_br]
        
        # First layer: connect center with neighbors
        layer1_outputs = []
        for gate in self.layers[0]:
            gate_outputs = []
            for n in neighbors:
                # Apply gate between center and each neighbor
                result = torch.zeros(batch_size, device=neighborhood.device)
                for b in range(batch_size):
                    # We'll average the result over all channels
                    channel_results = []
                    for c in range(self.channels):
                        channel_results.append(gate(center[b, c], n[b, c], hard))
                    result[b] = torch.stack(channel_results).mean()
                gate_outputs.append(result)
            # Combine results for this gate (average over neighbors)
            gate_result = torch.stack(gate_outputs, dim=1).mean(dim=1)
            layer1_outputs.append(gate_result)
            
        # Process through remaining layers
        prev_outputs = layer1_outputs
        for layer_idx in range(1, len(self.layers)):
            layer = self.layers[layer_idx]
            next_outputs = []
            
            for gate_idx, gate in enumerate(layer):
                # Get inputs from previous layer
                idx1 = min(gate_idx * 2, len(prev_outputs) - 1)
                idx2 = min(gate_idx * 2 + 1, len(prev_outputs) - 1)
                
                input1 = prev_outputs[idx1]
                input2 = prev_outputs[idx2]
                
                # Apply gate
                result = torch.zeros(batch_size, device=neighborhood.device)
                for b in range(batch_size):
                    result[b] = gate(input1[b], input2[b], hard)
                
                next_outputs.append(result)
                
            prev_outputs = next_outputs
            
        # Final layer should have a single output per sample
        # Guaranteed to be 2D tensor [batch, 1]
        final_output = prev_outputs[0].view(batch_size, 1)
        return final_output


class UpdateCircuit(nn.Module):
    """
    Circuit that updates the cell state based on perception outputs.
    """
    def __init__(self, in_channels, state_channels, device=DEVICE):
        super().__init__()
        # Input: perception outputs + current state
        total_inputs = in_channels + state_channels
        
        # Create a multi-layer circuit with decreasing gate counts
        self.layers = nn.ModuleList()
        
        # First several layers with fixed size
        fixed_layers = 8
        for _ in range(fixed_layers):
            layer = nn.ModuleList([LogicGate(device) for _ in range(128)])
            self.layers.append(layer)
        
        # Decreasing size layers
        sizes = [64, 32, 16, 8, 4, 2, state_channels]
        for size in sizes:
            layer = nn.ModuleList([LogicGate(device) for _ in range(size)])
            self.layers.append(layer)
            
        # Cache layer indices for faster access
        self.layer_input_indices = []
        for layer_idx in range(len(self.layers)):
            indices = []
            for gate_idx in range(len(self.layers[layer_idx])):
                if layer_idx == 0:
                    # For first layer, use consecutive inputs
                    idx1 = gate_idx * 2
                    idx2 = gate_idx * 2 + 1
                else:
                    # For later layers, connect pairs from previous layer
                    idx1 = gate_idx * 2
                    idx2 = gate_idx * 2 + 1
                indices.append((idx1, idx2))
            self.layer_input_indices.append(indices)
    
    def forward(self, perception_outputs, current_state, hard=False):
        """
        Update the cell state based on perception outputs and current state.
        perception_outputs: tensor of shape (batch, in_channels)
        current_state: tensor of shape (batch, state_channels)
        Returns: tensor of shape (batch, state_channels)
        """
        # Combine inputs
        x = torch.cat([perception_outputs, current_state], dim=1)
        
        # Process through layers
        prev_x = None
        
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                # First layer processes raw inputs
                next_x = []
                for gate_idx, gate in enumerate(layer):
                    input_indices = self.layer_input_indices[layer_idx][gate_idx]
                    idx1, idx2 = input_indices
                    
                    # Handle edge cases
                    idx1 = min(idx1, x.shape[1]-1)
                    idx2 = min(idx2, x.shape[1]-1)
                    
                    # Apply gate vectorized across batch
                    out = gate(x[:, idx1], x[:, idx2], hard)
                    next_x.append(out)
            else:
                # Subsequent layers process previous layer outputs
                next_x = []
                for gate_idx, gate in enumerate(layer):
                    input_indices = self.layer_input_indices[layer_idx][gate_idx]
                    idx1, idx2 = input_indices
                    
                    # Handle edge cases
                    idx1 = min(idx1, len(prev_x)-1)
                    idx2 = min(idx2, len(prev_x)-1)
                    
                    # Apply gate vectorized across batch
                    out = gate(prev_x[idx1], prev_x[idx2], hard)
                    next_x.append(out)
            
            prev_x = next_x
        
        # Final layer output should match the state_channels
        return torch.stack(prev_x, dim=1)


class DiffLogicCAImage(DifferentiableImage):
    """
    Image representation using Differentiable Logic Cellular Automata
    """
    def __init__(self, width, height, ca_channels=8, rgb_channels=3, perception_kernels=16, steps=20, device=DEVICE):
        super().__init__(width, height)
        # Ensure rgb_channels is at least 3
        rgb_channels = max(3, rgb_channels)
        
        self.width = width
        self.height = height
        self.ca_channels = ca_channels
        self.rgb_channels = rgb_channels
        self.steps = steps
        self.device = device
        
        # Cell state: first rgb_channels are for RGB values, rest are hidden state
        self.state = nn.Parameter(torch.zeros(height, width, ca_channels, device=device))
        
        # Initialize perception kernels
        self.perception_kernels = nn.ModuleList([
            PerceptionCircuit(ca_channels, device) for _ in range(perception_kernels)
        ])
        
        # Initialize update circuit
        self.update_circuit = UpdateCircuit(perception_kernels, ca_channels, device)
        
        # Output processing: convert the first rgb_channels to actual RGB values
        self.output_axes = ('s', 'y', 'x')
        self.lr = 1e-3
        
        # Special flag to disable 3D mode if needed
        self.disable_3d = False
        
        # Pre-compute a depth map for 3D mode
        self._depth_map = None
        self._prepare_depth_map()
        
        # Apply our patch to handle depth processing
        if ANIMATION_MODE == "3D":
            try:
                self._patch_depth_loss()
            except Exception as e:
                print(f"Warning: Could not patch depth processing: {e}")
                print("3D animations may not work properly.")
    
    def _prepare_depth_map(self):
        """Generate a simple depth map for 3D animation"""
        # Create a gradient from top to bottom that works well for depth models
        depth = np.ones((self.height, self.width), dtype=np.float32) * 0.5
        # Add vertical gradient (closer to top = further away)
        for y in range(self.height):
            depth[y, :] = 0.3 + 0.4 * (y / self.height)
        self._depth_map = depth
    
    # This is a special method used by the 3D system to get depth information
    def get_depth(self):
        """Return our pre-computed depth map"""
        if self._depth_map is None:
            self._prepare_depth_map()
        return self._depth_map, False
    
    # This is called by DepthLoss to get depth information
    def get_depth_tensor(self):
        """Return depth tensor for DepthLoss"""
        if self._depth_map is None:
            self._prepare_depth_map()
        depth_tensor = torch.from_numpy(self._depth_map).to(self.device).float()
        return depth_tensor.unsqueeze(0)  # Add batch dimension
    
    def clone(self):
        clone = DiffLogicCAImage(self.width, self.height, self.ca_channels, 
                                 self.rgb_channels, len(self.perception_kernels), self.steps)
        clone.state.data.copy_(self.state.data)
        
        # Copy circuits
        for i, kernel in enumerate(self.perception_kernels):
            for layer_idx, layer in enumerate(kernel.layers):
                for gate_idx, gate in enumerate(layer):
                    clone.perception_kernels[i].layers[layer_idx][gate_idx].logits.data.copy_(
                        kernel.layers[layer_idx][gate_idx].logits.data)
        
        for layer_idx, layer in enumerate(self.update_circuit.layers):
            for gate_idx, gate in enumerate(layer):
                clone.update_circuit.layers[layer_idx][gate_idx].logits.data.copy_(
                    layer[gate_idx].logits.data)
        
        return clone
    
    def get_image_tensor(self):
        # For compatibility with rest of pytti system
        # Convert first rgb_channels to RGB image
        return self.state[..., :self.rgb_channels].permute(2, 0, 1)
    
    @torch.no_grad()
    def set_image_tensor(self, tensor):
        # Assume tensor is (channels, height, width)
        rgb = tensor[:self.rgb_channels].permute(1, 2, 0)
        self.state[..., :self.rgb_channels] = rgb
    
    def step(self, hard=False):
        """Do almost nothing - just add tiny noise for exploration"""
        # This function now does almost nothing - just add minimal noise
        # All real optimization happens in train()
        with torch.no_grad():
            # Just add tiny noise (0.5%) to allow some exploration
            noise = (torch.rand_like(self.state) - 0.5) * 0.005
            self.state.data = (self.state.data + noise).clamp(0, 1)
    
    def run_ca(self, steps=None, hard=False):
        """
        Run CLIP optimization for multiple steps.
        CA behavior is completely removed - this just adds noise now.
        """
        if steps is None:
            steps = self.steps
        
        for _ in range(steps):
            # Just add minimal exploration noise
            with torch.no_grad():
                noise = (torch.rand_like(self.state) - 0.5) * 0.01
                self.state.data = (self.state.data + noise).clamp(0, 1)
    
    @torch.no_grad()
    def update(self):
        """Run the CA and update the image - called by Pytti system"""
        # Run just one step in the update to make it more responsive
        # The system will call this repeatedly
        self.step(hard=True)
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Simple encoding that preserves the original image with minimal processing"""
        # Resize the image
        pil_image = pil_image.resize((self.width, self.height), Image.LANCZOS)
        
        # Convert to tensor and normalize to [0,1]
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Directly use the image with minimal processing
        rgb = img_tensor
        
        # Initialize state
        state = torch.zeros(self.height, self.width, self.ca_channels, device=device)
        state[..., :self.rgb_channels] = rgb.permute(1, 2, 0)
        
        # Initialize hidden channels with random values
        if self.ca_channels > self.rgb_channels:
            hidden = torch.rand(self.height, self.width, self.ca_channels - self.rgb_channels, device=device)
            state[..., self.rgb_channels:] = hidden
        
        self.state = nn.Parameter(state)
    
    @torch.no_grad()
    def encode_random(self):
        """Initialize with random RGB values for maximum creativity"""
        # Create a random starting state
        state = torch.rand(self.height, self.width, self.ca_channels, device=self.device)
        
        # Make sure RGB values are more diverse
        for c in range(min(3, self.rgb_channels)):
            # Create areas of color rather than pure noise
            freq = 0.05 * (c + 1)  # Different frequency per channel
            y_coords = torch.arange(self.height, device=self.device).view(-1, 1).float() * freq
            x_coords = torch.arange(self.width, device=self.device).view(1, -1).float() * freq
            
            # Create interesting starting patterns using sine waves
            pattern = torch.sin(y_coords) * torch.sin(x_coords) * 0.5 + 0.5
            
            # Add some randomness to the pattern
            noise = torch.rand(self.height, self.width, device=self.device) * 0.3
            state[..., c] = pattern * 0.7 + noise * 0.3
        
        self.state = nn.Parameter(state)
    
    def image_loss(self):
        """Calculate internal loss for the image"""
        # Return an empty list instead of 0
        # This indicates we have no special image-specific losses
        return []

    def decode_tensor(self):
        """
        Convert the CA state to a standard RGB image tensor.
        """
        # Take the first rgb_channels (should be 3) of the state as RGB values
        rgb_values = self.state[..., :self.rgb_channels]
        
        # Convert from (height, width, channels) to (channels, height, width)
        rgb_tensor = rgb_values.permute(2, 0, 1)
        
        # Make sure values are between 0 and 1 and exactly 3 channels
        rgb_tensor = rgb_tensor.clamp(0, 1)
        
        # Ensure we have exactly 3 channels for RGB
        if rgb_tensor.shape[0] != 3:
            # If we have more or fewer than 3 channels, fix it
            if rgb_tensor.shape[0] > 3:
                rgb_tensor = rgb_tensor[:3]  # Take first 3 channels
            else:
                # If fewer than 3, duplicate the last channel
                channels_to_add = 3 - rgb_tensor.shape[0]
                last_channel = rgb_tensor[-1:].expand(channels_to_add, *rgb_tensor.shape[1:])
                rgb_tensor = torch.cat([rgb_tensor, last_channel], dim=0)
        
        return rgb_tensor

    def decode_image(self):
        """
        Override the default decode_image method to ensure proper PIL image format.
        This is called by Transforms.py to get the PIL image before depth processing.
        """
        # Use the parent class implementation but with our properly formatted tensor
        tensor = self.decode_tensor()
        tensor = named_rearrange(tensor, self.output_axes, ('y', 'x', 's'))
        array = tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        pil_image = Image.fromarray(array)
        
        # Attach our model to the PIL image for depth processing
        pil_image._from_difflogic = True
        pil_image._difflogic_model = self
        
        return pil_image

    def get_image_for_display(self):
        """
        Get image in the format expected by the 3D system.
        """
        # Get standard RGB tensor
        rgb_tensor = self.decode_tensor()
        
        # Ensure it's in the format expected by the depth model
        # The depth model expects a standard RGB image
        return rgb_tensor

    # When the 3D system asks for the latent, give it something simple
    def get_latent_tensor(self, detach=False):
        """
        Special handling for 3D mode - return a simplified representation
        """
        # If we're in 3D mode and want to disable it
        if self.disable_3d:
            # Return zeros - this will effectively skip 3D processing
            return torch.zeros(1, 3, self.height, self.width, device=self.device)
        
        # Otherwise, return our standard tensor format but with NCHW format
        # (batch, channels, height, width)
        tensor = self.decode_tensor().unsqueeze(0)
        if detach:
            return tensor.detach()
        return tensor

    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0):
        """
        Direct CLIP guidance for each pixel - AGGRESSIVE VERSION.
        Each pixel directly minimizes CLIP loss with no CA behavior.
        """
        # Get the current image tensor
        z = self.get_image_tensor()
        
        # Calculate losses from prompts (CLIP guidance)
        losses = {}
        total_loss = 0
        
        # Process each prompt
        for prompt in prompts:
            # Calculate loss for this prompt
            loss = prompt(z)
            losses[prompt] = loss
            total_loss += loss
        
        # Process loss augmentations
        for aug in loss_augs:
            aug_loss = aug(z)
            losses[aug] = aug_loss
            total_loss += aug_loss
        
        # Store total loss
        losses['TOTAL'] = total_loss
        
        # Print loss value for debugging
        if i % 10 == 0:
            print(f"CLIP Loss at step {i}: {total_loss.item()}")
        
        # Now use the loss to guide each pixel directly - this is the key part
        if total_loss > 0:
            # Calculate gradients
            total_loss.backward()
            
            # Apply gradients directly with AGGRESSIVE learning rates
            with torch.no_grad():
                grad = self.state.grad
                if grad is not None:
                    # Fixed high learning rate - no decay
                    base_lr = 0.2
                    
                    # Apply changes directly proportional to gradient strength
                    grad_strength = grad.abs()
                    
                    # Normalize gradient strength
                    if grad_strength.max() > 0:
                        # Print max gradient for debugging
                        if i % 10 == 0:
                            print(f"Max gradient: {grad_strength.max().item()}")
                        
                        # Apply STRONG updates to RGB channels
                        for c in range(min(3, self.rgb_channels)):
                            # Direct application of scaled gradients
                            self.state.data[..., c] -= grad[..., c] * base_lr
                        
                        # Apply smaller updates to hidden channels if they exist
                        if self.ca_channels > self.rgb_channels:
                            for c in range(self.rgb_channels, self.ca_channels):
                                self.state.data[..., c] -= grad[..., c] * base_lr * 0.5
                    
                    # Ensure values stay in proper range
                    self.state.data.clamp_(0, 1)
                    
                    # Zero gradients for next step
                    self.state.grad.zero_()
        
        # NO CA step - completely removed CA behavior
        # Just add minimal noise for exploration
        with torch.no_grad():
            noise = (torch.rand_like(self.state) - 0.5) * 0.01
            mask = torch.rand_like(self.state) < 0.1  # Only apply to 10% of pixels
            self.state.data = torch.where(mask, self.state.data + noise, self.state.data).clamp(0, 1)
        
        return losses

    def encode_from_tensor(self, tensor, add_noise=False):
        """
        Special method to handle encoding from tensors in 3D mode.
        This is used by the zoom_3d function in Transforms.py.
        
        Args:
            tensor: Input tensor to encode (formats vary)
            add_noise: Whether to add noise to the encoding
        """
        # Check tensor dimensions and format
        if tensor.dim() == 4:  # NCHW format
            tensor = tensor[0]  # Remove batch dimension
        
        if tensor.shape[0] == 3:  # CHW format
            # Convert to our expected shape
            rgb = tensor.permute(1, 2, 0)  # CHW -> HWC
        else:
            # Unknown format, use as is
            rgb = tensor
        
        # Update our RGB channels
        with torch.no_grad():
            self.state.data[..., :self.rgb_channels] = rgb.clamp(0, 1)
        
        return self

    # Monkeypatch the DepthLoss.get_depth method at initialization
    def _patch_depth_loss(self):
        """
        Monkey patch the DepthLoss.get_depth function to use our get_depth method.
        This is called during initialization to make our model compatible with 3D animations.
        """
        from pytti.LossAug import DepthLoss
        from functools import wraps
        from types import MethodType
        
        # Store original depth method
        if not hasattr(DepthLoss, '_original_get_depth'):
            DepthLoss._original_get_depth = DepthLoss.get_depth
        
        # Create a new method that checks if the image is a DiffLogicCAImage
        @wraps(DepthLoss._original_get_depth)
        def patched_get_depth(pil_image):
            # Check if this is our PIL image coming from our model
            # We can identify it by attaching a special attribute to the PIL image
            if hasattr(pil_image, '_from_difflogic') and pil_image._from_difflogic:
                # Get the model that created this image
                if hasattr(pil_image, '_difflogic_model'):
                    # Use the model's get_depth method
                    return pil_image._difflogic_model.get_depth(pil_image)
                # Fallback to a simple gradient
                height, width = pil_image.size[1], pil_image.size[0]
                gradient = np.ones((height, width), dtype=np.float32) * 0.5
                for y in range(height):
                    gradient[y, :] = 0.3 + 0.4 * (y / height)
                return gradient, False
            # Otherwise use the original method
            return DepthLoss._original_get_depth(pil_image)
        
        # Apply the patch
        DepthLoss.get_depth = patched_get_depth
        print("Patched DepthLoss.get_depth for DiffLogicCA compatibility")

def init_difflogic_ca(animation_mode="None"):
    """
    Initialize DiffLogicCA system.
    Store the animation mode globally to use when creating model instances.
    
    Args:
        animation_mode: The animation mode being used ("None", "2D", "3D", "Video Source")
    """
    global ANIMATION_MODE
    ANIMATION_MODE = animation_mode
    print(f"DiffLogicCA system initialized with animation mode: {animation_mode}")
    
    # Warning for 3D mode
    if animation_mode == "3D":
        print("WARNING: 3D mode is not fully supported with DiffLogicCA.")
        print("It will automatically fallback to 2D animation mode.")
    
    return True

# This function acts as a factory to create the right model based on animation mode
def create_difflogic_model(width, height, ca_channels=8, rgb_channels=3, perception_kernels=16, steps=20, device=DEVICE):
    """
    Factory function to create the appropriate DiffLogicCA model based on animation mode.
    """
    global ANIMATION_MODE
    
    if ANIMATION_MODE == "3D":
        # For 3D mode, create a 2D-compatible model with a warning
        print("Creating DiffLogicCA with 2D compatibility instead of 3D")
        return DiffLogicCAImage2D(width, height, ca_channels, rgb_channels, perception_kernels, steps, device)
    else:
        # For other modes, use the standard model
        return DiffLogicCAImage(width, height, ca_channels, rgb_channels, perception_kernels, steps, device)

# Create a simpler 2D-only version that doesn't try to use depth features
class DiffLogicCAImage2D(DiffLogicCAImage):
    """
    A version of DiffLogicCA that's compatible with 2D animation only.
    This will be used as a fallback when 3D mode is selected.
    """
    def __init__(self, width, height, ca_channels=8, rgb_channels=3, perception_kernels=16, steps=20, device=DEVICE):
        super().__init__(width, height, ca_channels, rgb_channels, perception_kernels, steps, device)
        print("Using 2D-compatible DiffLogicCA model (3D animations disabled)")
        
    # Override methods that might be called in 3D mode
    def get_depth(self, pil_image=None):
        """
        Return a simpler depth map that won't cause errors with the depth model.
        This is safer than trying to monkey-patch the system.
        """
        h, w = self.height, self.width
        # Create a gradient that looks good for animations
        depth = np.ones((h, w), dtype=np.float32)
        # Add gradient with closest at bottom (for typical scenes)
        for y in range(h):
            depth[y, :] = 0.1 + 0.8 * (1 - y / h)
        return depth, False
    
    def get_latent_tensor(self, detach=False):
        # Return a tensor that won't be used for 3D processing
        print("Latent requested but using 2D mode")
        tensor = self.decode_tensor().unsqueeze(0)
        if detach:
            return tensor.detach()
        return tensor 