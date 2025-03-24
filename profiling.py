import torch
import time
from contextlib import contextmanager
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# Global profiling state
profiling_enabled = False
function_timings = defaultdict(list)
memory_stats = defaultdict(list)

@contextmanager
def profile_section(name):
    """Context manager to profile a section of code"""
    if not profiling_enabled:
        yield
        return
        
    # Record starting memory
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    
    # Record timing
    start_time = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize()  # Make sure all CUDA ops are done
        end_time = time.time()
        end_mem = torch.cuda.memory_allocated()
        
        # Store results
        function_timings[name].append(end_time - start_time)
        memory_stats[name].append((start_mem, end_mem, end_mem - start_mem))

def start_profiling():
    """Start collecting profiling data"""
    global profiling_enabled, function_timings, memory_stats
    profiling_enabled = True
    function_timings = defaultdict(list)
    memory_stats = defaultdict(list)
    
def stop_profiling_and_report():
    """Stop profiling and report results"""
    global profiling_enabled
    profiling_enabled = False
    
    # Compute average times
    avg_times = {k: sum(v)/len(v) for k, v in function_timings.items() if v}
    total_time = sum(avg_times.values())
    
    # Sort by time (most expensive first)
    sorted_times = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== PROFILE RESULTS ===")
    print(f"Total execution time: {total_time:.4f}s")
    print("\nBreakdown by section:")
    for name, time in sorted_times:
        percentage = time / total_time * 100
        print(f"{name}: {time:.4f}s ({percentage:.1f}%)")
    
    # Calculate memory stats
    max_allocated = {k: max(m[1] for m in v) for k, v in memory_stats.items() if v}
    max_diff = {k: max(m[2] for m in v) for k, v in memory_stats.items() if v}
    
    # Convert to MB for display
    print("\nMemory usage (MB):")
    for name in sorted(max_allocated.keys()):
        print(f"{name}: Max allocated: {max_allocated[name]/1e6:.1f}MB, Max diff: {max_diff[name]/1e6:.1f}MB")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_times)), [x[1] for x in sorted_times])
    plt.xticks(range(len(sorted_times)), [x[0] for x in sorted_times], rotation=45, ha='right')
    plt.title('Profiling Results: Time per Section')
    plt.tight_layout()
    plt.savefig('profiling_results.png')
    plt.close()
    
    return pd.DataFrame({
        'Section': [x[0] for x in sorted_times],
        'Time (s)': [x[1] for x in sorted_times],
        'Percentage': [x[1]/total_time*100 for x in sorted_times]
    })

# Patch the DirectImageGuide.train method to profile each section
def patch_train_method():
    from pytti.ImageGuide import DirectImageGuide
    
    # Store the original method
    original_train = DirectImageGuide.train
    
    # Define the patched method with profiling
    def patched_train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=True):
        with profile_section("total_train_step"):
            with profile_section("optimizer_zero_grad"):
                self.optimizer.zero_grad()
                
            with profile_section("decode_training_tensor"):
                z = self.image_rep.decode_training_tensor()
            
            # Embedder section
            if self.embedder is not None:
                with profile_section("embedder_forward"):
                    image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)
                
                with profile_section("format_inputs"):
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
            with profile_section("format_loss_augs"):
                formatted_z = {}
                for aug in loss_augs:
                    formatted_z[aug] = format_input(z, self.image_rep, aug)
            
            # Compute losses
            with profile_section("compute_losses"):
                # Interpolation factor
                if i < interp_steps:
                    t = i / interp_steps
                    interp_losses = [prompt(formatted_inputs[prompt]['embeds'],
                                      formatted_inputs[prompt]['offsets'],
                                      formatted_inputs[prompt]['sizes'])[0] * (1 - t) 
                                      for prompt in interp_prompts]
                else:
                    t = 1
                    interp_losses = [0]
                
                # Prompt losses
                prompt_losses = {}
                for prompt in prompts:
                    loss = prompt(formatted_inputs[prompt]['embeds'],
                                  formatted_inputs[prompt]['offsets'],
                                  formatted_inputs[prompt]['sizes'])
                    loss[0].mul_(t)  # Scale by interpolation factor
                    prompt_losses[prompt] = loss
                
                # Augmentation losses
                aug_losses = {}
                for aug in loss_augs:
                    loss = aug(formatted_z[aug], self.image_rep)
                    aug_losses[aug] = loss
                
                # Image losses
                image_augs = self.image_rep.image_loss()
                image_losses = {}
                for aug in image_augs:
                    loss = aug(self.image_rep)
                    image_losses[aug] = loss
            
            # Aggregate losses
            with profile_section("aggregate_loss"):
                total_loss = sum(loss[0] for loss in prompt_losses.values()) + \
                             sum(loss[0] for loss in aug_losses.values()) + \
                             sum(loss[0] for loss in image_losses.values()) + \
                             sum(interp_losses)
            
            # Tracking
            if save_loss:
                with profile_section("save_loss"):
                    loss_dict = {'TOTAL': float(total_loss)}
                    loss_dict.update({str(k): float(v[0]) for k, v in prompt_losses.items()})
                    loss_dict.update({str(k): float(v[0]) for k, v in aug_losses.items()})
                    loss_dict.update({str(k): float(v[0]) for k, v in image_losses.items()})
                    if not self.dataframe:
                        self.dataframe = [pd.DataFrame(loss_dict, index=[i])]
                    else:
                        self.dataframe[0] = pd.concat([self.dataframe[0], pd.DataFrame(loss_dict, index=[i])])
            
            # Backward and optimize
            with profile_section("backward"):
                total_loss.backward()
                
            with profile_section("optimizer_step"):
                self.optimizer.step()
                
            with profile_section("image_update"):
                self.image_rep.update()
            
            return {'TOTAL': float(total_loss)}
    
    # Replace the method
    DirectImageGuide.train = patched_train
    return original_train

def patch_embedder():
    from pytti.Perceptor.Embedder import HDMultiClipEmbedder
    
    # Store original methods
    original_make_cutouts = HDMultiClipEmbedder.make_cutouts
    original_forward = HDMultiClipEmbedder.forward
    
    # Patched method with profiling
    def patched_make_cutouts(self, input, side_x, side_y, cut_size, device):
        with profile_section("make_cutouts"):
            return original_make_cutouts(self, input, side_x, side_y, cut_size, device)
    
    def patched_forward(self, diff_image, input=None, device=None):
        with profile_section("embedder_total"):
            if device is None:
                from pytti import DEVICE
                device = DEVICE
                
            perceptors = self.perceptors
            side_x, side_y = diff_image.image_shape
            
            with profile_section("embedder_format_input"):
                if input is None:
                    from pytti import format_module
                    input = format_module(diff_image, self).to(device=device, memory_format=torch.channels_last)
                else:
                    from pytti import format_input
                    input = format_input(input, diff_image, self).to(device=device, memory_format=torch.channels_last)
            
            max_size = min(side_x, side_y)
            image_embeds = []
            all_offsets = []
            all_sizes = []
            
            paddingx = min(round(side_x * self.padding), side_x)
            paddingy = min(round(side_y * self.padding), side_y)
            
            from pytti.Perceptor.Embedder import PADDING_MODES
            if self.border_mode != 'clamp':
                with profile_section("embedder_padding"):
                    input = torch.nn.functional.pad(
                        input, (paddingx, paddingx, paddingy, paddingy), 
                        mode=PADDING_MODES[self.border_mode]
                    )
                    
            for cut_size, perceptor in zip(self.cut_sizes, perceptors):
                with profile_section("embedder_make_cutouts"):
                    cutouts, offsets, sizes = self.make_cutouts(input, side_x, side_y, cut_size)
                
                with profile_section("embedder_normalize"):
                    from pytti import normalize
                    clip_in = normalize(cutouts)
                
                with profile_section("clip_encode_image"):
                    image_embeds.append(perceptor.encode_image(clip_in).float().unsqueeze(0))
                
                all_offsets.append(offsets)
                all_sizes.append(sizes)
            
            from pytti import cat_with_pad
            return cat_with_pad(image_embeds), torch.stack(all_offsets), torch.stack(all_sizes)
    
    # Replace the methods
    HDMultiClipEmbedder.make_cutouts = patched_make_cutouts
    HDMultiClipEmbedder.forward = patched_forward
    
    return original_make_cutouts, original_forward

def profile_training(guide, steps=100, prompts=None, interp_prompts=None, loss_augs=None):
    """Profile a training run"""
    if prompts is None:
        prompts = []
    if interp_prompts is None:
        interp_prompts = []
    if loss_augs is None:
        loss_augs = []
        
    # Apply patches
    original_train = patch_train_method()
    original_embedder_methods = patch_embedder()
    
    # Run profiling
    start_profiling()
    guide.run_steps(steps, prompts, interp_prompts, loss_augs)
    results = stop_profiling_and_report()
    
    # Restore original methods
    from pytti.ImageGuide import DirectImageGuide
    DirectImageGuide.train = original_train
    
    from pytti.Perceptor.Embedder import HDMultiClipEmbedder
    HDMultiClipEmbedder.make_cutouts = original_embedder_methods[0]
    HDMultiClipEmbedder.forward = original_embedder_methods[1]
    
    return results

if __name__ == "__main__":
    print("Run this from a notebook with a guide instance") 