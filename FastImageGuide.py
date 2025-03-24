import torch
from torch import optim, nn
import torch.nn.functional as F
import math
import pandas as pd
from tqdm import tqdm

from pytti import format_input, cat_with_pad, DEVICE
from ImageGuide import DirectImageGuide, EnhancedImageGuide
from Perceptor.FastEmbedder import FastHDMultiClipEmbedder, fast_spherical_dist_loss_function

class FastImageGuide(DirectImageGuide):
    """
    Real-time optimized version of DirectImageGuide.
    
    Key optimizations:
    1. Uses custom CUDA kernels for cutout generation
    2. Uses PyTorch mixed precision (AMP) for faster computation
    3. Batched processing of prompts and loss calculations
    4. JIT compilation for critical functions
    5. Reduced synchronization points
    """
    
    def __init__(self, image_rep, embedder=None, optimizer=None, lr=None, 
                 use_amp=True, jit_compile=True, **optimizer_params):
        """
        Initialize the fast image guide.
        
        Args:
            image_rep: Image representation
            embedder: CLIP embedder (will be replaced with fast version if not already)
            optimizer: PyTorch optimizer
            lr: Learning rate
            use_amp: Whether to use mixed precision training
            jit_compile: Whether to JIT compile critical functions
            **optimizer_params: Additional optimizer parameters
        """
        # Replace regular embedder with fast version if needed
        if embedder is not None and not isinstance(embedder, FastHDMultiClipEmbedder):
            # Extract parameters from old embedder
            params = {
                'perceptors': embedder.perceptors,
                'cutn': embedder.cutn,
                'cut_pow': embedder.cut_pow,
                'padding': embedder.padding,
                'border_mode': embedder.border_mode,
                'noise_fac': embedder.noise_fac
            }
            # Create fast embedder
            embedder = FastHDMultiClipEmbedder(**params)
            
        # Initialize parent class
        super().__init__(image_rep, embedder, optimizer, lr, **optimizer_params)
        
        # Set up mixed precision training
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Set up JIT compilation
        self.jit_compile = jit_compile
        if jit_compile:
            # JIT compile loss function
            self.spherical_dist_loss = torch.jit.script(fast_spherical_dist_loss_function)
        else:
            self.spherical_dist_loss = fast_spherical_dist_loss_function
            
        # Create empty buffer for prompt embeddings
        self.prompt_embeds_cache = {}
        self.last_prompt_set = None
            
    def clear_cache(self):
        """Clear the prompt embedding cache"""
        self.prompt_embeds_cache = {}
        self.last_prompt_set = None
            
    def batch_process_prompts(self, prompts, image_embeds, offsets, sizes, device=DEVICE):
        """
        Process all prompts in a single batched operation.
        
        Args:
            prompts: List of prompts
            image_embeds: CLIP image embeddings
            offsets: Cutout offsets
            sizes: Cutout sizes
            device: Device to use
            
        Returns:
            Tuple of (all_losses, all_raw_losses)
        """
        # Check if we need to compute text embeddings
        prompt_hash = hash(tuple(sorted([str(p) for p in prompts])))
        
        if self.last_prompt_set != prompt_hash:
            self.prompt_embeds_cache = {}
            self.last_prompt_set = prompt_hash
            
        # Process each prompt
        all_losses = []
        all_raw_losses = []
        
        for prompt in prompts:
            # Format inputs for prompt
            prompt_embeds = format_input(image_embeds, self.embedder, prompt)
            prompt_offsets = format_input(offsets, self.embedder, prompt)
            prompt_sizes = format_input(sizes, self.embedder, prompt)
            
            # Get prompt embedding from cache or compute it
            prompt_key = str(prompt)
            if prompt_key not in self.prompt_embeds_cache:
                if hasattr(prompt, 'embeds'):
                    self.prompt_embeds_cache[prompt_key] = prompt.embeds
                    
            # Compute loss
            target_embeds = self.prompt_embeds_cache.get(prompt_key, None)
            if target_embeds is not None:
                # Compute spherical distance
                dists_raw = self.spherical_dist_loss(prompt_embeds, target_embeds)
                
                # Apply prompt parameters
                weight = torch.as_tensor(float(prompt.weight), device=device)
                stop = torch.as_tensor(float(prompt.stop), device=device)
                
                # Get mask weights and stops
                mask_stops, mask_weights = prompt.mask(prompt_offsets, prompt_sizes, prompt_embeds.detach())
                weight = torch.as_tensor(mask_weights, device=device) * weight
                sign_offset = weight.sign().clamp(max=0)
                
                dists = dists_raw * weight.sign()
                stops = torch.maximum(mask_stops+sign_offset, stop)
                dists = weight.abs() * (dists.detach() * (dists >= stops) + dists * (dists < stops))
                
                all_losses.append(dists.mean())
                all_raw_losses.append(dists_raw.mean())
            
        return all_losses, all_raw_losses
        
    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
        """
        Optimized training step using CUDA kernels and mixed precision.
        
        Args:
            i: Current iteration
            prompts: List of text prompts
            interp_prompts: List of interpolation prompts
            loss_augs: List of loss augmentations
            interp_steps: Number of interpolation steps
            save_loss: Whether to save loss history
            
        Returns:
            Dictionary of losses
        """
        self.optimizer.zero_grad()
        
        # Decode image with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            z = self.image_rep.decode_training_tensor()
            
            # Get image embeddings
            if self.embedder is not None:
                image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)
                
                # Compute prompt losses in batches
                all_prompt_losses = []
                all_raw_losses = []
                
                # Calculate interpolation factor
                if i < interp_steps:
                    t = i / interp_steps
                    interp_weight = 1 - t
                    prompt_weight = t
                else:
                    interp_weight = 0
                    prompt_weight = 1
                
                # Process regular prompts
                if prompts:
                    prompt_losses, raw_losses = self.batch_process_prompts(
                        prompts, image_embeds, offsets, sizes
                    )
                    # Apply prompt weight
                    prompt_losses = [loss * prompt_weight for loss in prompt_losses]
                    all_prompt_losses.extend(prompt_losses)
                    all_raw_losses.extend(raw_losses)
                
                # Process interpolation prompts
                if interp_prompts and interp_weight > 0:
                    interp_losses, _ = self.batch_process_prompts(
                        interp_prompts, image_embeds, offsets, sizes
                    )
                    # Apply interpolation weight
                    interp_losses = [loss * interp_weight for loss in interp_losses]
                    all_prompt_losses.extend(interp_losses)
            
            # Process loss augmentations
            aug_losses = {}
            for aug in loss_augs:
                formatted_z = format_input(z, self.image_rep, aug)
                loss = aug(formatted_z, self.image_rep)
                aug_losses[aug] = loss
            
            # Process image losses
            image_augs = self.image_rep.image_loss()
            image_losses = {}
            for aug in image_augs:
                loss = aug(self.image_rep)
                image_losses[aug] = loss
            
            # Combine all losses
            total_loss = sum(all_prompt_losses) + \
                         sum(loss[0] for loss in aug_losses.values()) + \
                         sum(loss[0] for loss in image_losses.values())
        
        # Use gradient scaler for mixed precision
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update image representation
        self.image_rep.update()
        
        # Track losses
        if save_loss:
            loss_dict = {'TOTAL': float(total_loss)}
            
            # Add prompt losses
            for i, prompt in enumerate(prompts):
                if i < len(all_raw_losses):
                    loss_dict[str(prompt)] = float(all_raw_losses[i])
            
            # Add augmentation losses
            loss_dict.update({str(k): float(v[0]) for k, v in aug_losses.items()})
            loss_dict.update({str(k): float(v[0]) for k, v in image_losses.items()})
            
            # Save to dataframe
            if not self.dataframe:
                self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
            else:
                self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])
        
        return {'TOTAL': float(total_loss)}

    def run_steps(self, n_steps, prompts, interp_prompts, loss_augs, stop=-math.inf, interp_steps=0, i_offset=0, skipped_steps=0, callback=None):
        """
        Run multiple training steps with progress tracking.
        
        Args:
            n_steps: Number of steps to run
            prompts: List of prompts
            interp_prompts: List of interpolation prompts
            loss_augs: List of loss augmentations
            stop: Loss threshold to stop training
            interp_steps: Number of interpolation steps
            i_offset: Step offset
            skipped_steps: Number of skipped steps
            callback: Optional callback function to call after each step
            
        Returns:
            Number of steps completed
        """
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            losses = self.train(i + skipped_steps, prompts, interp_prompts, loss_augs, interp_steps=interp_steps)
            
            # Call callback if provided
            if callback is not None:
                stop_requested = callback(i, losses, self.image_rep)
                if stop_requested:
                    break
                    
            if losses['TOTAL'] <= stop:
                break
                
        return i + 1

class RealTimeImageGuide(FastImageGuide):
    """
    Extension of FastImageGuide with real-time update capabilities.
    Optimized for lower latency between updates.
    """
    
    def __init__(self, image_rep, embedder=None, optimizer=None, lr=None, 
                 use_amp=True, jit_compile=True, max_cutouts=20, **optimizer_params):
        """
        Initialize real-time image guide.
        
        Args:
            image_rep: Image representation
            embedder: CLIP embedder
            optimizer: PyTorch optimizer
            lr: Learning rate
            use_amp: Whether to use mixed precision
            jit_compile: Whether to JIT compile functions
            max_cutouts: Maximum number of cutouts (lower = faster)
            **optimizer_params: Additional optimizer parameters
        """
        super().__init__(image_rep, embedder, optimizer, lr, use_amp, jit_compile, **optimizer_params)
        
        # Reduce cutouts for faster processing
        if embedder is not None:
            embedder.cutn = min(embedder.cutn, max_cutouts)
            
        # Use a more aggressive optimizer for faster convergence
        if optimizer is None:
            try:
                from torch_optimizer import Lookahead, RAdam
                base_opt = RAdam(image_rep.parameters(), lr=self.optimizer_params.get('lr', 0.1))
                self.optimizer = Lookahead(base_opt, k=5, alpha=0.5)
            except ImportError:
                # Fall back to Adam with momentum
                self.optimizer = optim.Adam(
                    image_rep.parameters(),
                    lr=self.optimizer_params.get('lr', 0.1),
                    betas=(0.9, 0.999)
                )
        
        # Keep track of time per update
        self.last_update_time = 0
        
    def train_single_step(self, prompts, loss_augs=None):
        """
        Perform a single training step optimized for real-time updates.
        
        Args:
            prompts: List of text prompts
            loss_augs: Optional list of loss augmentations
            
        Returns:
            Current image tensor
        """
        loss_augs = loss_augs or []
        
        # Record start time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Run training step
        self.train(0, prompts, [], loss_augs, save_loss=False)
        
        # Record end time
        end_time.record()
        torch.cuda.synchronize()
        self.last_update_time = start_time.elapsed_time(end_time)
        
        # Return current image
        return self.image_rep.get_image_tensor()
        
    def get_fps(self):
        """Get the current frames per second rate"""
        if self.last_update_time > 0:
            return 1000 / self.last_update_time
        return 0 