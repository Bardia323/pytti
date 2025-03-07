from pytti import *
from pytti.Image import DifferentiableImage
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

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
        Returns: tensor of shape (batch, 1)
        """
        batch_size = neighborhood.shape[0]
        center = neighborhood[:, :, 1, 1]  # Center cell values
        
        # First layer: connect center with neighbors
        outputs = []
        for gate in self.layers[0]:
            results = []
            for b in range(batch_size):
                # For each sample, connect center with each neighbor
                center_val = center[b]
                neighbors = neighborhood[b].view(-1)  # Flatten the 3x3 grid
                # Remove center (which is at index 4 in the flattened grid)
                neighbors = torch.cat([neighbors[:4], neighbors[5:]])
                
                # Apply gate between center and each neighbor
                gate_results = []
                for i in range(8):  # 8 neighbors
                    gate_results.append(gate(center_val, neighbors[i], hard))
                
                # Combine results (could be adjusted based on task)
                results.append(torch.stack(gate_results))
            
            outputs.append(torch.stack(results))
        
        # Process through remaining layers
        for layer_idx in range(1, len(self.layers)):
            layer = self.layers[layer_idx]
            prev_outputs = outputs
            outputs = []
            
            for gate_idx, gate in enumerate(layer):
                inputs1 = prev_outputs[gate_idx * 2] if gate_idx * 2 < len(prev_outputs) else prev_outputs[-1]
                inputs2 = prev_outputs[gate_idx * 2 + 1] if gate_idx * 2 + 1 < len(prev_outputs) else prev_outputs[-1]
                
                results = []
                for b in range(batch_size):
                    # Apply gate to pairs of previous layer outputs
                    gate_results = []
                    for i in range(inputs1[b].shape[0]):
                        idx2 = min(i, inputs2[b].shape[0]-1)
                        gate_results.append(gate(inputs1[b][i], inputs2[b][idx2], hard))
                    results.append(torch.stack(gate_results))
                
                outputs.append(torch.stack(results))
        
        # Final layer should have a single output per sample
        return outputs[0].squeeze()


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
        for layer_idx, layer in enumerate(self.layers):
            next_x = []
            
            for gate_idx, gate in enumerate(layer):
                # Each gate takes two inputs from the previous layer
                if layer_idx == 0:
                    # For the first layer, use consecutive inputs
                    if gate_idx * 2 + 1 < x.shape[1]:
                        out = gate(x[:, gate_idx * 2], x[:, gate_idx * 2 + 1], hard)
                    else:
                        # If odd number of inputs, duplicate the last one
                        out = gate(x[:, -1], x[:, -1], hard)
                else:
                    # For subsequent layers, connect pairs from previous layer
                    if gate_idx * 2 + 1 < len(prev_x):
                        out = gate(prev_x[gate_idx * 2], prev_x[gate_idx * 2 + 1], hard)
                    elif gate_idx * 2 < len(prev_x):
                        # If odd number of gates, duplicate the last one
                        out = gate(prev_x[gate_idx * 2], prev_x[gate_idx * 2], hard)
                    else:
                        # If we need more outputs than previous layer had, reuse the last one
                        out = gate(prev_x[-1], prev_x[-1], hard)
                
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
        """Run one CA update step"""
        batch_size = self.height * self.width
        height, width, channels = self.state.shape
        
        # Create neighborhood for each cell
        padded = F.pad(self.state.permute(2, 0, 1), [1, 1, 1, 1], mode='replicate')
        neighborhoods = []
        
        for i in range(height):
            for j in range(width):
                neighborhood = padded[:, i:i+3, j:j+3]
                neighborhoods.append(neighborhood)
        
        neighborhoods = torch.stack(neighborhoods)  # (batch, channels, 3, 3)
        
        # Apply perception kernels
        perception_outputs = []
        for kernel in self.perception_kernels:
            perception_outputs.append(kernel(neighborhoods, hard))
        
        perception_outputs = torch.stack(perception_outputs, dim=1)  # (batch, num_kernels)
        
        # Current cell states
        current_states = self.state.reshape(batch_size, channels)
        
        # Update states
        new_states = self.update_circuit(perception_outputs, current_states, hard)
        
        # Reshape back to grid
        self.state = nn.Parameter(new_states.reshape(height, width, channels))
    
    def run_ca(self, steps=None, hard=False):
        """Run CA for multiple steps"""
        if steps is None:
            steps = self.steps
        
        for _ in range(steps):
            self.step(hard)
    
    @torch.no_grad()
    def update(self):
        """Run the CA and update the image"""
        self.run_ca(hard=True)  # Use hard (discrete) inference
    
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
        # No internal loss for now
        return 0 

    def decode_tensor(self):
        """
        Convert the CA state to an RGB image tensor.
        Returns a decoded tensor of this image.
        """
        # Take the first rgb_channels of the state as RGB values
        rgb_values = self.state[..., :self.rgb_channels]
        
        # Convert from (height, width, channels) to (channels, height, width)
        rgb_tensor = rgb_values.permute(2, 0, 1)
        
        # Make sure values are between 0 and 1
        rgb_tensor = rgb_tensor.clamp(0, 1)
        
        return rgb_tensor

def init_difflogic_ca():
    """
    Initialize DiffLogicCA system.
    This is a placeholder function to match the pattern of other image models.
    """
    print("DiffLogicCA system initialized.")
    return True 