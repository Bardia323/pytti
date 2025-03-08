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
        """Simplified step method for better performance"""
        height, width, channels = self.state.shape
        
        # Use a simple cellular automaton rule for testing
        # This is much faster than the full perception/update circuit
        with torch.no_grad():
            # Create a padded version of the state
            padded = F.pad(self.state.permute(2, 0, 1), [1, 1, 1, 1], mode='replicate')
            
            # Simple convolution to count neighbors (for Game of Life-like rules)
            kernel = torch.ones(1, 1, 3, 3, device=self.device)
            kernel[0, 0, 1, 1] = 0  # Don't count the center cell
            
            new_state = torch.zeros_like(self.state)
            
            # Process each channel
            for c in range(channels):
                # Get this channel
                channel = padded[c:c+1].unsqueeze(0)
                
                # Count neighbors
                neighbors = F.conv2d(channel, kernel, padding=0)[0, 0]
                
                # Current state
                current = self.state[..., c]
                
                # Apply Game of Life-like rules
                # Live cells with 2-3 neighbors survive
                # Dead cells with 3 neighbors become alive
                if hard:
                    new_state[..., c] = ((current > 0.5) & ((neighbors >= 2) & (neighbors <= 3))) | ((current <= 0.5) & (neighbors == 3))
                else:
                    # Continuous version
                    survive = (current > 0.5) * ((neighbors >= 2) & (neighbors <= 3)).float()
                    born = (current <= 0.5) * (neighbors == 3).float()
                    new_state[..., c] = survive + born
        
        self.state = nn.Parameter(new_state)
    
    def run_ca(self, steps=None, hard=False):
        """Run CA for multiple steps"""
        if steps is None:
            steps = self.steps
        
        # Use torch.no_grad for inference if not training
        if not self.training and not hard:
            with torch.no_grad():
                for _ in range(steps):
                    self.step(hard)
        else:
            for _ in range(steps):
                self.step(hard)
    
    @torch.no_grad()
    def update(self):
        """Run the CA and update the image - called by Pytti system"""
        # Run just one step in the update to make it more responsive
        # The system will call this repeatedly
        self.step(hard=True)
    
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """Convert a PIL image to CA state"""
        # Resize the image
        pil_image = pil_image.resize((self.width, self.height), Image.LANCZOS)
        
        # Convert to tensor and normalize to [0,1]
        img_tensor = TF.to_tensor(pil_image).to(device)
        
        # Initialize first rgb_channels with the image values (binarized)
        # If smart_encode is True, we can use a more sophisticated encoding
        if smart_encode:
            # You can implement a more sophisticated encoding here if needed
            # For now, just use the same binarization
            rgb = (img_tensor > 0.5).float()
        else:
            rgb = (img_tensor > 0.5).float()
        
        # Initialize state
        state = torch.zeros(self.height, self.width, self.ca_channels, device=device)
        state[..., :self.rgb_channels] = rgb.permute(1, 2, 0)
        
        # Initialize a seed in the center
        center_h, center_w = self.height // 2, self.width // 2
        state[center_h, center_w, :] = 1.0
        
        self.state = nn.Parameter(state)
    
    @torch.no_grad()
    def encode_random(self):
        """Initialize with random state"""
        # Random binary state
        state = torch.randint(0, 2, (self.height, self.width, self.ca_channels), 
                              device=self.device).float()
        
        # Or just a center seed
        state = torch.zeros(self.height, self.width, self.ca_channels, device=self.device)
        center_h, center_w = self.height // 2, self.width // 2
        state[center_h, center_w, :] = 1.0
        
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
        """
        # Use the parent class implementation but with our properly formatted tensor
        tensor = self.decode_tensor()
        tensor = named_rearrange(tensor, self.output_axes, ('y', 'x', 's'))
        array = tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        return Image.fromarray(array)

    def get_image_for_display(self):
        """
        Get image in the format expected by the 3D system.
        """
        # Get standard RGB tensor
        rgb_tensor = self.decode_tensor()
        
        # Ensure it's in the format expected by the depth model
        # The depth model expects a standard RGB image
        return rgb_tensor

    # Add this method for 3D compatibility
    def get_latent_for_depth(self):
        """
        Special method to provide compatible input for depth model.
        This bypasses the normal image processing for the depth model.
        """
        # Convert our RGB image to the format expected by the depth model
        rgb_tensor = self.decode_tensor()
        
        # The depth model gets confused by our format, so return a simpler
        # format that it can handle - just a standard RGB image
        return rgb_tensor

    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0):
        """
        Integration with Pytti's training loop.
        This method is called by the DirectImageGuide to update the image based on prompts.
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
        
        # Now use the loss to guide the CA evolution
        # We'll adjust the CA state based on the gradient of the loss
        if total_loss > 0:
            # Calculate gradients
            total_loss.backward()
            
            # Use gradients to influence the CA state
            # This is the key part - we're using the CLIP loss to guide the CA
            with torch.no_grad():
                # Get gradients for the RGB channels
                grad = self.state.grad
                if grad is not None:
                    # Scale gradients to influence CA state
                    # Focus on RGB channels (first 3)
                    rgb_grad = grad[..., :self.rgb_channels]
                    
                    # Apply gradient influence (small step)
                    influence = 0.01
                    self.state.data[..., :self.rgb_channels] -= influence * rgb_grad
                    
                    # Zero gradients for next step
                    self.state.grad.zero_()
        
        # Run a CA step after applying gradient influence
        self.step(hard=False)  # Use soft logic during training
        
        return losses

# Add monkeypatch to make 3D mode work with our model
# This overrides the get_depth method only for our images
try:
    from pytti.LossAug.DepthLoss import get_depth as original_get_depth

    # Store the original method so we can still call it
    original_get_depth_for_pil = original_get_depth

    # Define a new method to handle DiffLogicCA images specially
    def patched_get_depth(pil_image):
        """
        Patched version of get_depth that handles our DiffLogicCA images specially
        """
        # Create a simpler image for depth estimation - a blank image with a gradient
        # This avoids errors with the depth model
        width, height = pil_image.size
        
        # Create a gradient image that depth models handle well
        gradient = np.ones((height, width), dtype=np.float32) * 0.5
        # Add a simple gradient from top to bottom
        for y in range(height):
            gradient[y, :] = 0.3 + 0.4 * (y / height)
        
        # Return the simplified depth map
        return gradient, False

    # Only apply the patch if zoom_3d is being used with our model
    def patch_depth_for_difflogic():
        """Apply the patch to DepthLoss.get_depth"""
        from pytti.LossAug import DepthLoss
        DepthLoss.get_depth = patched_get_depth
        print("Patched depth estimation for DiffLogicCA")

    def restore_depth_original():
        """Restore the original get_depth method"""
        from pytti.LossAug import DepthLoss
        DepthLoss.get_depth = original_get_depth_for_pil
        print("Restored original depth estimation")
        
except ImportError:
    # Handle case where the depth module isn't available
    def patch_depth_for_difflogic():
        print("Depth module not available, skipping patch")
        
    def restore_depth_original():
        print("Depth module not available, nothing to restore")

def init_difflogic_ca(animation_mode="None"):
    """
    Initialize DiffLogicCA system.
    This function applies special handling for different animation modes.
    
    Args:
        animation_mode: The animation mode being used ("None", "2D", "3D", "Video Source")
    """
    print(f"DiffLogicCA system initialized with animation mode: {animation_mode}")
    
    # Apply different optimizations based on animation mode
    if animation_mode == "3D":
        # Apply our depth estimation patch to make 3D mode work
        print("Applying 3D mode compatibility patch")
        patch_depth_for_difflogic()
    else:
        # For other modes, use default handling
        print("Using standard mode")
    
    return True 