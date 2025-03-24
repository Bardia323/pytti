import os
import subprocess
import sys
from pathlib import Path

# Define dependencies
DEPENDENCIES = [
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "pillow>=8.0.0",
    "numpy>=1.19.0",
    "matplotlib>=3.3.0",
    "tqdm>=4.50.0",
    "torch-optimizer>=0.3.0",
    "scikit-learn>=0.24.0",
    "kornia>=0.5.0",
    "ipython>=7.0.0"
]

def check_cuda():
    """Check if CUDA is available"""
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Custom CUDA kernels require CUDA.")
        print("You can still run the code, but without the optimized kernels.")
        return False
    return True

def check_nvcc():
    """Check if nvcc (NVIDIA CUDA compiler) is available"""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Found NVIDIA CUDA compiler: {version}")
            return True
        else:
            print("WARNING: nvcc (NVIDIA CUDA compiler) is not found in PATH.")
            print("Custom CUDA kernels might not compile correctly.")
            return False
    except FileNotFoundError:
        print("WARNING: nvcc (NVIDIA CUDA compiler) is not found in PATH.")
        print("Custom CUDA kernels might not compile correctly.")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    for dep in DEPENDENCIES:
        print(f"Installing {dep}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    print("Dependencies installed successfully.")

def compile_cuda_extension():
    """Compile the CUDA extension"""
    print("Compiling CUDA extension...")
    
    if not os.path.exists("fast_clip.py"):
        print("ERROR: fast_clip.py not found! Make sure you're in the right directory.")
        return False
    
    try:
        import torch.utils.cpp_extension
        from torch.utils.cpp_extension import load
        
        # Import fast_clip to trigger compilation
        import fast_clip
        print("CUDA extension compiled successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to compile CUDA extension: {str(e)}")
        return False

def create_test_file():
    """Create a simple test file to verify installation"""
    test_code = """
import torch
import torch.nn.functional as F
from Perceptor.FastEmbedder import fast_spherical_dist_loss_function
from fast_clip import fast_generate_cutouts, fast_spherical_dist_loss

def test_fast_clip():
    # Test if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available, optimized kernels won't work")
        return False
        
    # Create test tensors
    try:
        # Create random image tensor
        image = torch.rand(1, 3, 224, 224, device='cuda')
        
        # Create random sizes and offsets
        sizes = torch.rand(10, device='cuda')
        offsets = torch.rand(10, 2, device='cuda')
        
        # Test cutout generation
        cutouts = fast_generate_cutouts(image, sizes, offsets, 128, 10)
        print(f"✓ Cutout generation successful: {cutouts.shape}")
        
        # Test spherical distance
        x = torch.rand(10, 128, device='cuda')
        y = torch.rand(10, 128, device='cuda')
        
        # Normalize
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        
        # Test loss function
        loss = fast_spherical_dist_loss(x, y)
        print(f"✓ Spherical distance calculation successful: {loss.shape}")
        
        print("All tests passed! The fast_clip extension is working correctly.")
        return True
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_fast_clip()
"""
    
    with open("test_fast_clip.py", "w") as f:
        f.write(test_code)
    
    print("Created test file: test_fast_clip.py")
    print("Run 'python test_fast_clip.py' to verify the installation.")

def main():
    """Main setup function"""
    print("=== Fast CLIP Setup ===")
    
    # Check requirements
    has_cuda = check_cuda()
    has_nvcc = check_nvcc()
    
    if not (has_cuda and has_nvcc):
        print("\nWARNING: Some requirements are missing for optimal performance.")
        response = input("Do you want to continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Setup aborted.")
            return
    
    # Install dependencies
    install_dependencies()
    
    # Compile CUDA extension
    success = compile_cuda_extension()
    
    if success:
        # Create test file
        create_test_file()
        
        print("\nSetup completed successfully!")
        print("To run the real-time example, use: python realtime_example.py")
    else:
        print("\nSetup completed with errors.")
        print("Please check the error messages above.")
    
if __name__ == "__main__":
    main() 