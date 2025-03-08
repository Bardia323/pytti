from torch import optim, nn
from pytti.Notebook import tqdm
from pytti import *
import pandas as pd
import math
import inspect
import torch

from labellines import labelLines
from scipy.signal import savgol_filter
import torch.optim as optim
import matplotlib.pyplot as plt
from pytti import format_input

def unpack_dict(D, n=2):
    ds = [{k: V[i] for k, V in D.items()} for i in range(n)]
    return tuple(ds)

def smooth_dataframe(df, window_size):
    """Applies a moving average filter to the columns of df."""
    smoothed_df = pd.DataFrame(index=df.index, columns=df.columns)
    for key in df.columns:
        smoothed_df[key] = savgol_filter(df[key], window_size, 2, mode='nearest')
    return smoothed_df

class DirectImageGuide():
    """
    Image guide that uses an optimizer and torch autograd to optimize an image representation.
    Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun).
    """
    def __init__(self, image_rep, embedder, optimizer=None, lr=None, **optimizer_params):
        self.image_rep = image_rep
        self.embedder = embedder
        if lr is None:
            lr = image_rep.lr
        optimizer_params['lr'] = lr
        self.optimizer_params = optimizer_params
        if optimizer is None:
            self.optimizer = optim.Adam(image_rep.parameters(), **optimizer_params)
        else:
            self.optimizer = optimizer
        self.dataframe = []

    def run_steps(self, n_steps, prompts, interp_prompts, loss_augs, stop=-math.inf, interp_steps=0, i_offset=0, skipped_steps=0):
        """Runs the optimizer."""
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            losses = self.train(i + skipped_steps, prompts, interp_prompts, loss_augs, interp_steps=interp_steps)
            if losses['TOTAL'] <= stop:
                break
        return i + 1

    def set_optim(self, opt=None):
        if opt is not None:
            self.optimizer = opt
        else:
            self.optimizer = optim.Adam(self.image_rep.parameters(), **self.optimizer_params)

    def clear_dataframe(self):
        self.dataframe = []

    def plot_losses(self, axs):
        def plot_dataframe(df, ax, legend=False):
            keys = df.columns.tolist()
            keys.sort(reverse=True, key=lambda k: df[k].iloc[-1])
            ax.clear()
            df[keys].plot(ax=ax, legend=legend)
            if legend:
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                           bottom=True, top=False, left=True, right=False)
            labelLines(ax.get_lines(), align=False)
            return

        dfs = self.dataframe[:]
        if dfs:
            dfs[0] = smooth_dataframe(dfs[0], 17)
        for i, (df, ax) in enumerate(zip(dfs, axs)):
            if len(df.index) < 2:
                return False
            if not df.empty:
                plot_dataframe(df, ax, legend=(i == 0))
            ax.set_ylabel('Loss')
            ax.set_xlabel('Step')
        return True

    def update(self, i, stage_i):
        """Update hook called every step."""
        pass

    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
        """Performs a training step."""
        self.optimizer.zero_grad()
        z = self.image_rep.decode_training_tensor()

        # Precompute formatted inputs to avoid redundant computations
        if self.embedder is not None:
            image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)

            # Cache formatted inputs for prompts and interpolation prompts
            all_prompts = prompts + interp_prompts
            formatted_inputs = {}
            for prompt in set(all_prompts):
                formatted_inputs[prompt] = {
                    'embeds': format_input(image_embeds, self.embedder, prompt),
                    'offsets': format_input(offsets, self.embedder, prompt),
                    'sizes': format_input(sizes, self.embedder, prompt)
                }
        else:
            formatted_inputs = {}

        # Cache formatted inputs for loss augmentations
        formatted_z = {}
        for aug in loss_augs:
            formatted_z[aug] = format_input(z, self.image_rep, aug)

        # Compute interpolation factor
        if i < interp_steps:
            t = i / interp_steps
            interp_losses = [prompt(formatted_inputs[prompt]['embeds'],
                                    formatted_inputs[prompt]['offsets'],
                                    formatted_inputs[prompt]['sizes'])[0] * (1 - t) for prompt in interp_prompts]
        else:
            t = 1
            interp_losses = [0]

        # Compute prompt losses
        prompt_losses = {}
        for prompt in prompts:
            loss = prompt(formatted_inputs[prompt]['embeds'],
                          formatted_inputs[prompt]['offsets'],
                          formatted_inputs[prompt]['sizes'])
            # Scale loss by interpolation factor
            loss[0].mul_(t)
            prompt_losses[prompt] = loss

        # Compute augmentation losses
        aug_losses = {}
        for aug in loss_augs:
            loss = aug(formatted_z[aug], self.image_rep)
            aug_losses[aug] = loss

        # Compute image losses
        image_augs = self.image_rep.image_loss()
        image_losses = {}
        for aug in image_augs:
            loss = aug(self.image_rep)
            image_losses[aug] = loss

        # Aggregate losses
        total_loss = sum(loss[0] for loss in prompt_losses.values()) + \
                     sum(loss[0] for loss in aug_losses.values()) + \
                     sum(loss[0] for loss in image_losses.values()) + \
                     sum(interp_losses)

        # Prepare loss tracking
        if save_loss:
            loss_dict = {'TOTAL': float(total_loss)}
            loss_dict.update({str(k): float(v[0]) for k, v in prompt_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in aug_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in image_losses.items()})
            if not self.dataframe:
                self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
            else:
                self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])

        # First do the backward pass
        total_loss.backward()
        
        # Apply gradient clipping if enabled
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.image_rep.parameters(), self.grad_clip)
        
        # Apply the optimizer step
        self.optimizer.step()
        self.image_rep.update()

        return {'TOTAL': float(total_loss)}

class EnhancedImageGuide(DirectImageGuide):
    """
    Enhanced version of DirectImageGuide with support for:
    1. Multiple optimizer types (Adam, AdamW, RAdam, Lookahead)
    2. Adaptive loss weighting
    3. Gradient clipping to prevent exploding gradients
    """
    def __init__(self, image_rep, embedder, optimizer_name='adam', lr=None, adaptive_weights=False, 
                 weight_update_freq=10, weight_scale_factor=0.5, grad_clip=1.0, **optimizer_params):
        """
        image_rep: The image representation to optimize
        embedder: The embedder to use for image-text comparison
        optimizer_name: One of 'adam', 'adamw', 'radam', or 'lookahead'
        lr: Learning rate (if None, uses image_rep.lr)
        adaptive_weights: Whether to use adaptive loss weighting
        weight_update_freq: How often to update loss weights (in steps)
        weight_scale_factor: How strongly to adjust weights (0-1)
        grad_clip: Maximum norm for gradient clipping (None to disable)
        """
        self.image_rep = image_rep
        self.embedder = embedder
        
        # Set up learning rate
        if lr is None:
            lr = image_rep.lr
        optimizer_params['lr'] = lr
        self.optimizer_params = optimizer_params
        
        # Store adaptive weighting parameters
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        self.weight_scale_factor = weight_scale_factor
        self.grad_clip = grad_clip
        self.loss_history = {}
        self.initial_weights = {}
        
        # Create optimizer based on name
        self.optimizer_name = optimizer_name.lower()
        self._create_optimizer()
        
        self.dataframe = []
    
    def _create_optimizer(self):
        """Create the specified optimizer type"""
        params = self.image_rep.parameters()
        
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(params, **self.optimizer_params)
        elif self.optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(params, **self.optimizer_params)
        elif self.optimizer_name == 'radam':
            try:
                from torch_optimizer import RAdam
                self.optimizer = RAdam(params, **self.optimizer_params)
            except ImportError:
                print("RAdam optimizer not available, falling back to Adam")
                self.optimizer = optim.Adam(params, **self.optimizer_params)
        elif self.optimizer_name == 'lookahead':
            try:
                from torch_optimizer import Lookahead
                base_opt = optim.Adam(params, **self.optimizer_params)
                self.optimizer = Lookahead(base_opt)
            except ImportError:
                print("Lookahead optimizer not available, falling back to Adam")
                self.optimizer = optim.Adam(params, **self.optimizer_params)
        else:
            print(f"Unknown optimizer {self.optimizer_name}, using Adam")
            self.optimizer = optim.Adam(params, **self.optimizer_params)
    
    def set_optim(self, opt=None, optimizer_name=None):
        """Set a new optimizer"""
        if opt is not None:
            self.optimizer = opt
        elif optimizer_name is not None:
            self.optimizer_name = optimizer_name.lower()
            self._create_optimizer()
        else:
            self._create_optimizer()
    
    def _store_initial_weights(self, prompt_losses, aug_losses, image_losses):
        """Store initial weights of loss components"""
        if not self.initial_weights:
            for prompt, loss in prompt_losses.items():
                if hasattr(prompt, 'weight'):
                    # Handle both tensor and non-tensor weights
                    if hasattr(prompt.weight, 'clone'):
                        self.initial_weights[str(prompt)] = prompt.weight.clone()
                    else:
                        self.initial_weights[str(prompt)] = prompt.weight
            
            for aug, loss in aug_losses.items():
                if hasattr(aug, 'weight'):
                    # Handle both tensor and non-tensor weights
                    if hasattr(aug.weight, 'clone'):
                        self.initial_weights[str(aug)] = aug.weight.clone()
                    else:
                        self.initial_weights[str(aug)] = aug.weight
            
            for aug, loss in image_losses.items():
                if hasattr(aug, 'weight'):
                    # Handle both tensor and non-tensor weights
                    if hasattr(aug.weight, 'clone'):
                        self.initial_weights[str(aug)] = aug.weight.clone()
                    else:
                        self.initial_weights[str(aug)] = aug.weight
    
    def _update_loss_weights(self, i, prompt_losses, aug_losses, image_losses):
        """Adaptively adjust loss weights based on their magnitudes"""
        if not self.adaptive_weights or i % self.weight_update_freq != 0:
            return
        
        # Calculate mean loss for each component
        all_losses = {}
        for prompt, loss in prompt_losses.items():
            all_losses[str(prompt)] = float(loss[0])
        
        for aug, loss in aug_losses.items():
            all_losses[str(aug)] = float(loss[0])
        
        for aug, loss in image_losses.items():
            all_losses[str(aug)] = float(loss[0])
        
        # Skip if we don't have enough history
        for key, value in all_losses.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
        
        if i < self.weight_update_freq * 2:
            return
        
        # Calculate mean and std for each loss
        loss_stats = {}
        for key, history in self.loss_history.items():
            if len(history) >= 5:  # Need enough samples
                recent = history[-5:]
                loss_stats[key] = (sum(recent) / len(recent), max(recent) - min(recent))
        
        # Skip if not enough components have stats
        if len(loss_stats) < 2:
            return

        # We need to delay weight updates until after backward() to avoid autograd errors
        # Store the weights to update after backward
        self.weights_to_update = []
        
        # Adjust weights - increase weight for smaller losses, decrease for larger ones
        for prompt, loss in prompt_losses.items():
            if str(prompt) in loss_stats and hasattr(prompt, 'weight') and hasattr(prompt, 'set_weight'):
                stat = loss_stats[str(prompt)]
                mean_loss = stat[0]
                
                # Calculate adjustment factor based on relative magnitude
                all_means = [s[0] for s in loss_stats.values()]
                mean_of_means = sum(all_means) / len(all_means)
                
                if mean_loss > 0 and mean_of_means > 0:
                    # If this loss is smaller than average, increase its weight
                    # If it's larger, decrease its weight
                    ratio = mean_of_means / mean_loss
                    adjust = (ratio - 1.0) * self.weight_scale_factor
                    
                    # Apply adjustment with limit
                    initial_weight = self.initial_weights.get(str(prompt), prompt.weight)
                    
                    # Handle both tensor and non-tensor weights
                    current_weight = float(prompt.weight) if not hasattr(prompt.weight, 'item') else prompt.weight.item()
                    new_weight = current_weight * (1.0 + min(max(adjust, -0.2), 0.2))
                    
                    # Get initial weight as float
                    init_weight_val = float(initial_weight) if not hasattr(initial_weight, 'item') else initial_weight.item()
                    
                    # Don't let weight go below 20% or above 500% of initial
                    new_weight = max(min(new_weight, init_weight_val * 5.0), init_weight_val * 0.2)
                    
                    # Save for later application after backward pass
                    self.weights_to_update.append((prompt, new_weight))
        
        # Same for augmentation losses
        for aug, loss in {**aug_losses, **image_losses}.items():
            if str(aug) in loss_stats and hasattr(aug, 'weight') and hasattr(aug, 'set_weight'):
                stat = loss_stats[str(aug)]
                mean_loss = stat[0]
                
                all_means = [s[0] for s in loss_stats.values()]
                mean_of_means = sum(all_means) / len(all_means)
                
                if mean_loss > 0 and mean_of_means > 0:
                    ratio = mean_of_means / mean_loss
                    adjust = (ratio - 1.0) * self.weight_scale_factor
                    
                    # Apply adjustment with limit
                    initial_weight = self.initial_weights.get(str(aug), aug.weight)
                    
                    # Handle both tensor and non-tensor weights
                    current_weight = float(aug.weight) if not hasattr(aug.weight, 'item') else aug.weight.item()
                    new_weight = current_weight * (1.0 + min(max(adjust, -0.2), 0.2))
                    
                    # Get initial weight as float
                    init_weight_val = float(initial_weight) if not hasattr(initial_weight, 'item') else initial_weight.item()
                    
                    # Don't let weight go below 20% or above 500% of initial
                    new_weight = max(min(new_weight, init_weight_val * 5.0), init_weight_val * 0.2)
                    
                    # Save for later application after backward pass
                    self.weights_to_update.append((aug, new_weight))

    def _apply_weight_updates(self):
        """Apply weight updates after backward pass to avoid autograd errors"""
        if not hasattr(self, 'weights_to_update') or not self.weights_to_update:
            return
            
        for obj, new_weight in self.weights_to_update:
            try:
                # Check if the weight needs to be an integer
                if hasattr(obj.weight, 'dtype') and obj.weight.dtype == torch.int64:
                    new_weight = int(new_weight)
                
                # Get the number of arguments expected by set_weight
                sig = inspect.signature(obj.set_weight)
                num_params = len(sig.parameters)
                
                # Only call with the right number of arguments
                if num_params == 1:
                    obj.set_weight(new_weight)
                elif num_params == 2 and 'device' in str(sig):
                    obj.set_weight(new_weight, DEVICE)
                else:
                    print(f"Warning: Cannot set weight for {obj}, unexpected number of parameters ({num_params})")
            except Exception as e:
                print(f"Warning: Could not set weight for {obj}: {e}")
                
        # Clear the list
        self.weights_to_update = []
    
    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
        """Performs a training step with adaptive weight adjustment."""
        self.optimizer.zero_grad()
        z = self.image_rep.decode_training_tensor()

        # Precompute formatted inputs to avoid redundant computations
        if self.embedder is not None:
            image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)

            # Cache formatted inputs for prompts and interpolation prompts
            all_prompts = prompts + interp_prompts
            formatted_inputs = {}
            for prompt in set(all_prompts):
                formatted_inputs[prompt] = {
                    'embeds': format_input(image_embeds, self.embedder, prompt),
                    'offsets': format_input(offsets, self.embedder, prompt),
                    'sizes': format_input(sizes, self.embedder, prompt)
                }
        else:
            formatted_inputs = {}

        # Cache formatted inputs for loss augmentations
        formatted_z = {}
        for aug in loss_augs:
            formatted_z[aug] = format_input(z, self.image_rep, aug)

        # Compute interpolation factor
        if i < interp_steps:
            t = i / interp_steps
            interp_losses = [prompt(formatted_inputs[prompt]['embeds'],
                                    formatted_inputs[prompt]['offsets'],
                                    formatted_inputs[prompt]['sizes'])[0] * (1 - t) for prompt in interp_prompts]
        else:
            t = 1
            interp_losses = [0]

        # Compute prompt losses
        prompt_losses = {}
        for prompt in prompts:
            loss = prompt(formatted_inputs[prompt]['embeds'],
                          formatted_inputs[prompt]['offsets'],
                          formatted_inputs[prompt]['sizes'])
            # Scale loss by interpolation factor
            loss[0].mul_(t)
            prompt_losses[prompt] = loss

        # Compute augmentation losses
        aug_losses = {}
        for aug in loss_augs:
            loss = aug(formatted_z[aug], self.image_rep)
            aug_losses[aug] = loss

        # Compute image losses
        image_augs = self.image_rep.image_loss()
        image_losses = {}
        for aug in image_augs:
            loss = aug(self.image_rep)
            image_losses[aug] = loss
            
        # Store initial weights if using adaptive weights
        if self.adaptive_weights:
            self._store_initial_weights(prompt_losses, aug_losses, image_losses)
            
        # Update weights adaptively if enabled - this now just calculates new weights
        # but doesn't apply them yet to avoid autograd errors
        self._update_loss_weights(i, prompt_losses, aug_losses, image_losses)

        # Aggregate losses
        total_loss = sum(loss[0] for loss in prompt_losses.values()) + \
                     sum(loss[0] for loss in aug_losses.values()) + \
                     sum(loss[0] for loss in image_losses.values()) + \
                     sum(interp_losses)

        # Prepare loss tracking
        if save_loss:
            loss_dict = {'TOTAL': float(total_loss)}
            loss_dict.update({str(k): float(v[0]) for k, v in prompt_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in aug_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in image_losses.items()})
            if not self.dataframe:
                self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
            else:
                self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])

        # First do the backward pass
        total_loss.backward()
        
        # Apply gradient clipping if enabled
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.image_rep.parameters(), self.grad_clip)
        
        # Apply the optimizer step
        self.optimizer.step()
        
        # Now it's safe to apply weight updates after backward and optimizer steps
        if self.adaptive_weights:
            self._apply_weight_updates()
            
        self.image_rep.update()

        return {'TOTAL': float(total_loss)}
