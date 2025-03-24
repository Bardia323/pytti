import torch
import os
import sys
import time
from contextlib import contextmanager

# Print basic system info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our acceleration modules
try:
    from cuda_accelerate import apply_mixed_precision
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    print("Mixed precision module not found")
    MIXED_PRECISION_AVAILABLE = False

try:
    from cutout_kernel import patch_embedder
    FAST_CUTOUT_AVAILABLE = True
except ImportError:
    print("Fast cutout module not found")
    FAST_CUTOUT_AVAILABLE = False

try:
    from spherical_dist_kernel import patch_spherical_dist
    FAST_SPHERICAL_DIST_AVAILABLE = True
except ImportError:
    print("Fast spherical distance module not found")
    FAST_SPHERICAL_DIST_AVAILABLE = False

try:
    from profiling import profile_training
    PROFILING_AVAILABLE = True
except ImportError:
    print("Profiling module not found")
    PROFILING_AVAILABLE = False

@contextmanager
def timer(name):
    """Simple timing context manager"""
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.4f} seconds")

def apply_all_accelerations():
    """Apply all available accelerations"""
    original_functions = {}
    
    # Apply mixed precision
    if MIXED_PRECISION_AVAILABLE:
        with timer("Setting up mixed precision"):
            mp = apply_mixed_precision()
            original_functions['mixed_precision'] = mp
            print("✓ Mixed precision enabled")
    
    # Apply fast cutout generation
    if FAST_CUTOUT_AVAILABLE:
        with timer("Setting up fast cutout generation"):
            original_make_cutouts = patch_embedder()
            original_functions['cutout'] = original_make_cutouts
            print("✓ Fast cutout generation enabled")
    
    # Apply fast spherical distance calculation
    if FAST_SPHERICAL_DIST_AVAILABLE:
        with timer("Setting up fast spherical distance calculation"):
            original_spherical_dist = patch_spherical_dist()
            original_functions['spherical_dist'] = original_spherical_dist
            print("✓ Fast spherical distance calculation enabled")
    
    print(f"\nAcceleration complete! {len(original_functions)}/3 optimizations applied.")
    return original_functions

def benchmark_guide(guide, steps=100, prompts=None, interp_prompts=None, loss_augs=None):
    """
    Benchmark a guide instance with and without accelerations.
    
    Args:
        guide: DirectImageGuide or EnhancedImageGuide instance
        steps: Number of steps to run
        prompts: List of prompts
        interp_prompts: List of interpolation prompts
        loss_augs: List of loss augmentations
    """
    if prompts is None:
        prompts = []
    if interp_prompts is None:
        interp_prompts = []
    if loss_augs is None:
        loss_augs = []
    
    # First run profiling if available
    if PROFILING_AVAILABLE:
        print("Running profiling on current implementation...")
        profile_training(guide, steps=10, prompts=prompts, 
                        interp_prompts=interp_prompts, loss_augs=loss_augs)
    
    # Apply accelerations
    original_functions = apply_all_accelerations()
    
    # Run with accelerations
    print("\nRunning benchmark with accelerations...")
    with timer(f"Running {steps} steps with accelerations"):
        guide.run_steps(steps, prompts, interp_prompts, loss_augs)
    
    # Run profiling again if available
    if PROFILING_AVAILABLE:
        print("\nRunning profiling with accelerations...")
        profile_training(guide, steps=10, prompts=prompts,
                        interp_prompts=interp_prompts, loss_augs=loss_augs)
    
    return original_functions

def restore_original(original_functions):
    """Restore original functions"""
    if 'mixed_precision' in original_functions:
        print("Restoring original training method")
        from pytti.ImageGuide import DirectImageGuide, EnhancedImageGuide
        DirectImageGuide.train = original_functions['mixed_precision'][0]
        DirectImageGuide.__init__ = original_functions['mixed_precision'][1]
        if hasattr(EnhancedImageGuide, 'train'):
            EnhancedImageGuide.train = original_functions['mixed_precision'][0]
            EnhancedImageGuide.__init__ = original_functions['mixed_precision'][1]
    
    if 'cutout' in original_functions:
        print("Restoring original cutout generation")
        from pytti.Perceptor.Embedder import HDMultiClipEmbedder
        HDMultiClipEmbedder.make_cutouts = original_functions['cutout']
    
    if 'spherical_dist' in original_functions:
        print("Restoring original spherical distance calculation")
        import pytti.Perceptor.Prompt
        pytti.Perceptor.Prompt.spherical_dist_loss = original_functions['spherical_dist']

if __name__ == "__main__":
    print("This script provides functions to accelerate PyTTI training with CUDA.")
    print("Import and use the following functions:")
    print("- apply_all_accelerations(): Apply all available accelerations")
    print("- benchmark_guide(guide, steps=100): Benchmark a guide with and without accelerations")
    print("- restore_original(original_functions): Restore original functions")
    
    print("\nExample usage:")
    print("```")
    print("from accelerate_pytti import apply_all_accelerations")
    print("apply_all_accelerations()")
    print("# Your training code here")
    print("```") 