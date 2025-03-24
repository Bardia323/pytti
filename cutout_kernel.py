import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, CUDAExtension
import kornia.augmentation as K

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA not available. Fast cutout generation requires CUDA.")
    CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = True

# Define the CUDA kernel code for fast cutout generation
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA kernel for generating random cutout parameters
__global__ void generate_cutout_params_kernel(
    float* sizes,
    float* offsetsx,
    float* offsetsy,
    int num_cutouts,
    int side_x,
    int side_y,
    float mean,
    float std,
    float cut_pow,
    float min_size_ratio,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cutouts) return;
    
    // Initialize random state
    curandState_t state;
    curand_init(seed + idx, 0, 0, &state);
    
    // Generate random size with normal distribution
    float rand_normal = curand_normal(&state) * std + mean;
    rand_normal = max(min_size_ratio, min(1.0f, rand_normal));
    
    // Apply power for biasing toward smaller/larger cutouts
    float size_ratio = powf(rand_normal, cut_pow);
    
    // Convert to pixel size
    int max_side = min(side_x, side_y);
    int size = int(max_side * size_ratio);
    
    // Size can't be larger than image dimensions
    size = min(size, max_side);
    
    // Store size as float 0-1 ratio
    sizes[idx] = (float)size / (float)max_side;
    
    // Calculate max valid offsets
    int offsetx_max = side_x - size + 1;
    int offsety_max = side_y - size + 1;
    
    // Generate random offsets
    float rand_x = curand_uniform(&state);
    float rand_y = curand_uniform(&state);
    
    // Calculate pixel offsets
    int offsetx = min(int(rand_x * offsetx_max), offsetx_max - 1);
    int offsety = min(int(rand_y * offsety_max), offsety_max - 1);
    
    // Store normalized offsets 0-1
    offsetsx[idx] = (float)offsetx / (float)side_x;
    offsetsy[idx] = (float)offsety / (float)side_y;
}

// CUDA kernel for extracting and resizing cutouts
__global__ void extract_cutouts_kernel(
    float* input,
    float* output,
    float* sizes,
    float* offsetsx,
    float* offsetsy,
    int num_cutouts,
    int batch_size,
    int channels,
    int height,
    int width,
    int target_size
) {
    // NOTE: This is a simplified version. A complete implementation would
    // include bilinear interpolation for resizing, but that's complex for a kernel.
    // This demonstrates the concept but would be replaced with a more sophisticated version.
    
    int cutout_idx = blockIdx.x;
    int pixel_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (cutout_idx >= num_cutouts) return;
    
    // Get the size and offsets for this cutout
    float size_ratio = sizes[cutout_idx];
    float offsetx_ratio = offsetsx[cutout_idx];
    float offsety_ratio = offsetsy[cutout_idx];
    
    int size = int(min(width, height) * size_ratio);
    int offsetx = int(width * offsetx_ratio);
    int offsety = int(height * offsety_ratio);
    
    // Process each output pixel
    if (pixel_idx < target_size * target_size * channels) {
        // Convert linear index to coordinates
        int c = (pixel_idx / (target_size * target_size)) % channels;
        int y = (pixel_idx / target_size) % target_size;
        int x = pixel_idx % target_size;
        
        // Map to input coordinates (nearest neighbor)
        float scale = (float)size / (float)target_size;
        int in_y = int(y * scale) + offsety;
        int in_x = int(x * scale) + offsetx;
        
        // Boundary check
        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
            // Get the value and store it
            int in_offset = ((0 * channels + c) * height + in_y) * width + in_x;
            int out_offset = ((cutout_idx * channels + c) * target_size + y) * target_size + x;
            output[out_offset] = input[in_offset];
        }
    }
}

// PyTorch C++ extension interface
torch::Tensor generate_cutout_params(
    int num_cutouts,
    int side_x,
    int side_y,
    float mean,
    float std,
    float cut_pow,
    float min_size_ratio,
    int64_t seed
) {
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);
    
    // Allocate output tensors
    auto sizes = torch::empty({num_cutouts}, options);
    auto offsetsx = torch::empty({num_cutouts}, options);
    auto offsetsy = torch::empty({num_cutouts}, options);
    
    // Launch kernel
    int threads = 256;
    int blocks = (num_cutouts + threads - 1) / threads;
    
    generate_cutout_params_kernel<<<blocks, threads>>>(
        sizes.data_ptr<float>(),
        offsetsx.data_ptr<float>(),
        offsetsy.data_ptr<float>(),
        num_cutouts,
        side_x,
        side_y,
        mean,
        std,
        cut_pow,
        min_size_ratio,
        static_cast<unsigned long long>(seed)
    );
    
    // Stack the three parameters into a single tensor [num_cutouts, 3]
    return torch::stack({sizes, offsetsx, offsetsy}, 1);
}

// This function would be a more comprehensive cutout extractor
// For now we'll use PyTorch ops as they're highly optimized
torch::Tensor extract_cutouts(
    torch::Tensor input,
    torch::Tensor params,
    int target_size
) {
    // This is just a placeholder showing the interface
    // A real implementation would use the kernel + sophisticated resizing
    
    int num_cutouts = params.size(0);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(torch::kCUDA);
    
    auto output = torch::empty({num_cutouts, channels, target_size, target_size}, options);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_cutout_params", &generate_cutout_params, "Generate cutout parameters");
    m.def("extract_cutouts", &extract_cutouts, "Extract cutouts from input image");
}
"""

try:
    # Build directory for storing the custom extension
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_cutout')
    os.makedirs(build_dir, exist_ok=True)
    
    # Load the custom extension if CUDA is available
    if CUDA_AVAILABLE:
        try:
            # Write the CUDA code to a temporary file
            cuda_file = os.path.join(build_dir, 'cutout_kernel.cu')
            with open(cuda_file, 'w') as f:
                f.write(cuda_source)
            
            # Load the extension with JIT compilation
            cutout_cuda = load(
                name='cutout_cuda',
                sources=[cuda_file],
                build_directory=build_dir,
                verbose=True
            )
            
            # Load successful
            CUDA_EXTENSION_LOADED = True
            
        except Exception as e:
            error_message = str(e)
            if "Ninja is required" in error_message:
                print("Ninja build system not found. You can install it with:")
                print("  pip install ninja")
                print("Using fallback implementation instead.")
            elif "CUDA_HOME environment variable is not set" in error_message:
                print("CUDA_HOME environment variable is not set.")
                print("Please set it to your CUDA install root, for example:")
                if os.name == 'nt':  # Windows
                    print("  SET CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6")
                else:  # Linux/Mac
                    print("  export CUDA_HOME=/usr/local/cuda-11.6")
                print("Using fallback implementation instead.")
            else:
                print(f"Failed to build CUDA extension: {e}")
            CUDA_EXTENSION_LOADED = False
    else:
        CUDA_EXTENSION_LOADED = False
        
except Exception as e:
    print(f"Failed to build CUDA extension: {e}")
    CUDA_EXTENSION_LOADED = False

class FastCutoutGenerator(nn.Module):
    """
    Faster implementation of cutout generation using CUDA kernels.
    This replaces the make_cutouts method in HDMultiClipEmbedder.
    """
    def __init__(self, cutn=40, cut_pow=1.5, padding=0.25, border_mode='clamp', noise_fac=0.1):
        super().__init__()
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.padding = padding
        self.border_mode = border_mode
        self.noise_fac = noise_fac
        self.mean = 0.8
        self.std = 0.3
        self.min_size_ratio = 0.05  # Minimum cutout size ratio
        
        # Add augmentations to match the original implementation
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.3),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4,),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=False, p=0.7),
            nn.Identity(),
        )
        
        # Choose between CUDA or fallback implementation
        self.use_cuda = CUDA_EXTENSION_LOADED
        if self.use_cuda:
            print("Using CUDA accelerated cutout generation")
        else:
            print("Using fallback cutout generation")
            from pytti import DEVICE
            self.device = DEVICE
            
    def make_cutouts(self, input, side_x, side_y, cut_size, device=None):
        """
        Generate cutouts with optimized code path.
        Returns cutouts, offsets, sizes with same API as original.
        """
        if device is None:
            from pytti import DEVICE
            device = DEVICE
            
        if not self.use_cuda:
            # Fallback implementation using PyTorch ops
            return self._make_cutouts_fallback(input, side_x, side_y, cut_size, device)
        
        # Calculate proper min_size_ratio based on original implementation
        max_size = min(side_x, side_y)
        min_size_ratio = cut_size / max_size  # Match original code's minimum size constraint
            
        # Generate random parameters for cutouts
        params = cutout_cuda.generate_cutout_params(
            self.cutn,
            side_x,
            side_y,
            self.mean,
            self.std,
            self.cut_pow,
            min_size_ratio,  # Use calculated value instead of hardcoded 0.05
            torch.randint(0, 2**31 - 1, (1,), device='cuda').item()
        )
        
        # Extract sizes and offsets
        sizes = params[:, 0]
        offsetsx = params[:, 1]
        offsetsy = params[:, 2]
        
        # Actually extract the cutouts using PyTorch ops for now
        # (more complex to implement the full extraction+resize in CUDA)
        cutouts = []
        for i in range(self.cutn):
            size = int(min(side_x, side_y) * sizes[i].item())
            offsetx = int(side_x * offsetsx[i].item())
            offsety = int(side_y * offsetsy[i].item())
            
            # Extract the cutout
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
            
        # Stack cutouts and apply augmentations
        cutouts = torch.cat(cutouts)
        cutouts = self.augs(cutouts)
        
        # Format offsets and sizes like the original implementation
        offsets_list = []
        sizes_list = []
        for i in range(self.cutn):
            offsets_list.append(torch.as_tensor([[offsetsx[i].item()/side_x, offsetsy[i].item()/side_y]]).to(device))
            sizes_list.append(torch.as_tensor([[sizes[i].item(), sizes[i].item()]]).to(device))
        
        offsets = torch.cat(offsets_list)
        sizes_tensor = torch.cat(sizes_list)
        
        # Add noise if requested
        if self.noise_fac:
            facs = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
            
        return cutouts, offsets, sizes_tensor
    
    def _make_cutouts_fallback(self, input, side_x, side_y, cut_size, device):
        """Fallback implementation using pure PyTorch ops"""
        min_size = min(side_x, side_y, cut_size)
        max_size = min(side_x, side_y)
        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)
        cutouts = []
        offsets = []
        sizes = []
        
        for _ in range(self.cutn):
            # Use clip with cut_size/max_size as minimum, exactly like original implementation
            size = int(max_size * 
                   torch.zeros(1,).normal_(mean=self.mean, std=self.std)
                   .clip(cut_size/max_size, 1.) ** self.cut_pow)
                   
            offsetx_max = side_x - size + 1
            offsety_max = side_y - size + 1
            
            if self.border_mode == 'clamp':
                offsetx = torch.clamp((torch.rand([])*(offsetx_max+2*paddingx) - paddingx).floor().int(), 0, offsetx_max)
                offsety = torch.clamp((torch.rand([])*(offsety_max+2*paddingy) - paddingy).floor().int(), 0, offsety_max)
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            else:
                px = min(size, paddingx)
                py = min(size, paddingy)
                offsetx = (torch.rand([])*(offsetx_max+2*px) - px).floor().int()
                offsety = (torch.rand([])*(offsety_max+2*py) - py).floor().int()
                cutout = input[:, :, paddingy + offsety:paddingy + offsety + size, paddingx + offsetx:paddingx + offsetx + size]
                
            cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
            offsets.append(torch.as_tensor([[offsetx/side_x, offsety/side_y]]).to(device))
            sizes.append(torch.as_tensor([[size/side_x, size/side_y]]).to(device))
            
        # Stack cutouts and apply augmentations
        cutouts = torch.cat(cutouts)
        cutouts = self.augs(cutouts)
        
        offsets = torch.cat(offsets)
        sizes = torch.cat(sizes)
        
        if self.noise_fac:
            facs = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts.add_(facs * torch.randn_like(cutouts))
            
        return cutouts, offsets, sizes
        
def patch_embedder():
    """
    Replace the cutout generation in HDMultiClipEmbedder with the faster version.
    """
    from pytti.Perceptor.Embedder import HDMultiClipEmbedder
    
    # Create a fast cutout generator
    fast_generator = FastCutoutGenerator()
    
    # Store original make_cutouts method
    original_make_cutouts = HDMultiClipEmbedder.make_cutouts
    
    # Replace with fast version
    HDMultiClipEmbedder.make_cutouts = fast_generator.make_cutouts
    
    return original_make_cutouts

if __name__ == "__main__":
    print("Run 'patch_embedder()' to enable fast cutout generation") 