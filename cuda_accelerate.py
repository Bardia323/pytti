import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from functools import wraps
from pytti import format_input

class MixedPrecisionTraining:
    """
    Wrapper to enable mixed precision training for PyTTI.
    This can provide a significant speedup on newer GPUs.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled)
        
    def patch_image_guide(self, guide_class):
        """Patch a DirectImageGuide or derived class with mixed precision training"""
        original_train = guide_class.train
        
        @wraps(original_train)
        def mixed_precision_train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with autocast if enabled
            with autocast(enabled=self.mp_trainer.enabled):
                z = self.image_rep.decode_training_tensor()
                
                # Embedder section
                if self.embedder is not None:
                    image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)
                    
                    # Cache formatted inputs
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
                
                # Cache loss aug inputs
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
                
                # Compute losses
                prompt_losses = {}
                for prompt in prompts:
                    loss = prompt(formatted_inputs[prompt]['embeds'],
                                formatted_inputs[prompt]['offsets'],
                                formatted_inputs[prompt]['sizes'])
                    # Scale loss by interpolation factor
                    loss[0].mul_(t)
                    prompt_losses[prompt] = loss
                
                aug_losses = {}
                for aug in loss_augs:
                    loss = aug(formatted_z[aug], self.image_rep)
                    aug_losses[aug] = loss
                
                image_augs = self.image_rep.image_loss()
                image_losses = {}
                for aug in image_augs:
                    loss = aug(self.image_rep)
                    image_losses[aug] = loss
                
                # Special handling for EnhancedImageGuide
                if hasattr(self, 'adaptive_weights'):
                    if self.adaptive_weights:
                        self._store_initial_weights(prompt_losses, aug_losses, image_losses)
                        self._update_loss_weights(i, prompt_losses, aug_losses, image_losses)
                
                # Aggregate losses
                total_loss = sum(loss[0] for loss in prompt_losses.values()) + \
                             sum(loss[0] for loss in aug_losses.values()) + \
                             sum(loss[0] for loss in image_losses.values()) + \
                             sum(interp_losses)
            
            # Loss tracking
            if save_loss:
                import pandas as pd
                loss_dict = {'TOTAL': float(total_loss)}
                loss_dict.update({str(k): float(v[0]) for k, v in prompt_losses.items()})
                loss_dict.update({str(k): float(v[0]) for k, v in aug_losses.items()})
                loss_dict.update({str(k): float(v[0]) for k, v in image_losses.items()})
                if not self.dataframe:
                    self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
                else:
                    self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])
            
            # Special handling for backward and optimizer with mixed precision
            self.mp_trainer.scaler.scale(total_loss).backward()
            self.mp_trainer.scaler.step(self.optimizer)
            self.mp_trainer.scaler.update()
            
            # Apply weight updates for EnhancedImageGuide
            if hasattr(self, 'adaptive_weights') and self.adaptive_weights:
                self._apply_weight_updates()
                
            self.image_rep.update()
            
            return {'TOTAL': float(total_loss)}
        
        # Add the mixed precision trainer to the class
        guide_class.mp_trainer = property(lambda self: self._mp_trainer)
        
        # Patch __init__ to create mp_trainer
        original_init = guide_class.__init__
        
        @wraps(original_init)
        def init_with_mp(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._mp_trainer = MixedPrecisionTraining(enabled=True)
        
        guide_class.__init__ = init_with_mp
        guide_class.train = mixed_precision_train
        
        return original_train, original_init
    
    def enable_fp16_clip(self):
        """Convert CLIP models to FP16 for faster inference"""
        from pytti.Perceptor import CLIP_PERCEPTORS
        
        if CLIP_PERCEPTORS is None:
            print("CLIP models not loaded yet")
            return
        
        for i, model in enumerate(CLIP_PERCEPTORS):
            if model.dtype != torch.float16:
                print(f"Converting CLIP model {i} to FP16")
                CLIP_PERCEPTORS[i] = model.half()

def apply_mixed_precision():
    """Apply mixed precision to all relevant classes"""
    from pytti.ImageGuide import DirectImageGuide, EnhancedImageGuide
    
    mp = MixedPrecisionTraining()
    mp.patch_image_guide(DirectImageGuide)
    mp.patch_image_guide(EnhancedImageGuide)
    mp.enable_fp16_clip()
    
    return mp

if __name__ == "__main__":
    print("Run 'apply_mixed_precision()' to enable mixed precision training") 