import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, CUDAExtension

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA not available. Fast spherical distance calculation requires CUDA.")
    CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = True

# Define the CUDA kernel code for fast spherical distance calculation
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for fast spherical distance calculation
// Fuses normalize -> subtract -> norm -> arcsin -> square -> multiply
__global__ void spherical_dist_loss_kernel(
    const float* x,          // Input tensor x
    const float* y,          // Input tensor y
    float* output,           // Output tensor
    const int batch_size,    // Batch size
    const int embedding_dim  // Embedding dimension
) {
    // Calculate the global thread ID
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process if within the valid range
    if (idx < batch_size) {
        // First compute L2 norms for normalization
        float x_norm = 0.0f;
        float y_norm = 0.0f;
        
        // Compute norms
        for (int i = 0; i < embedding_dim; i++) {
            const int offset = idx * embedding_dim + i;
            x_norm += x[offset] * x[offset];
            y_norm += y[offset] * y[offset];
        }
        
        // Add small epsilon to avoid division by zero
        const float eps = 1e-8f;
        x_norm = sqrtf(x_norm + eps);
        y_norm = sqrtf(y_norm + eps);
        
        // Now compute the dot product of normalized vectors
        float dot_product = 0.0f;
        for (int i = 0; i < embedding_dim; i++) {
            const int offset = idx * embedding_dim + i;
            // Normalize both x and y while computing dot product
            dot_product += (x[offset] / x_norm) * (y[offset] / y_norm);
        }
        
        // Clamp dot product to [-1, 1] to avoid numerical issues
        dot_product = fminf(fmaxf(dot_product, -1.0f), 1.0f);
        
        // Calculate the spherical distance
        // d = arcsin(|x-y|/2)^2 * 2
        // Using trig identity: |x-y|/2 = sqrt((1-cos(theta))/2)
        // where cos(theta) = dot_product
        const float cos_theta = dot_product;
        const float sin_squared_half_theta = (1.0f - cos_theta) / 2.0f;
        
        // arcsin(sqrt(sin_squared_half_theta))^2 * 2
        // = sin_squared_half_theta * (pi/2)^2 * 2  (small angle approximation)
        // But since we're optimizing, actual scale doesn't matter as much as the shape
        
        // Use asin directly with proper bounds
        const float arcsin_val = asinf(sqrtf(fmaxf(sin_squared_half_theta, 0.0f)));
        output[idx] = arcsin_val * arcsin_val * 2.0f;
    }
}

// Interface function exposed to Python
torch::Tensor spherical_dist_loss_cuda(
    torch::Tensor x,
    torch::Tensor y
) {
    // Check dimensions
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(y.dim() == 2, "y must be a 2D tensor");
    TORCH_CHECK(x.size(1) == y.size(1), "x and y must have the same embedding dimension");
    
    // Get dimensions
    const int batch_size = x.size(0);
    const int embedding_dim = x.size(1);
    
    // Create output tensor
    auto output = torch::empty({batch_size}, x.options());
    
    // Calculate number of blocks and threads
    const int threads_per_block = 512;
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    spherical_dist_loss_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        embedding_dim
    );
    
    return output;
}

// Batch many-to-one version for comparing many embeddings to a single reference
torch::Tensor spherical_dist_loss_batch_cuda(
    torch::Tensor x,   // [batch_size, embedding_dim]
    torch::Tensor y    // [1 or batch_size, embedding_dim]
) {
    // Handle the case where y is a single embedding
    if (y.size(0) == 1 && x.size(0) > 1) {
        // Expand y to match x's batch size
        y = y.expand(x.size(0), -1);
    }
    
    // Now use the regular function
    return spherical_dist_loss_cuda(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spherical_dist_loss", &spherical_dist_loss_cuda, "Fast spherical distance loss");
    m.def("spherical_dist_loss_batch", &spherical_dist_loss_batch_cuda, "Fast spherical distance batch loss");
}
"""

try:
    # Build directory for storing the custom extension
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_spherical')
    os.makedirs(build_dir, exist_ok=True)
    
    # Load the custom extension if CUDA is available
    if CUDA_AVAILABLE:
        # Write the CUDA code to a temporary file
        cuda_file = os.path.join(build_dir, 'spherical_dist_kernel.cu')
        with open(cuda_file, 'w') as f:
            f.write(cuda_source)
        
        # Load the extension with JIT compilation
        spherical_dist_cuda = load(
            name='spherical_dist_cuda',
            sources=[cuda_file],
            build_directory=build_dir,
            verbose=True
        )
        
        # Load successful
        CUDA_EXTENSION_LOADED = True
    else:
        CUDA_EXTENSION_LOADED = False
        
except Exception as e:
    print(f"Failed to build CUDA extension: {e}")
    CUDA_EXTENSION_LOADED = False

class FastSphericalDistanceLoss(nn.Module):
    """
    Faster implementation of spherical distance calculation using CUDA kernels.
    This replaces the spherical_dist_loss function in pytti.
    """
    def __init__(self):
        super().__init__()
        self.use_cuda = CUDA_EXTENSION_LOADED
        if self.use_cuda:
            print("Using CUDA accelerated spherical distance calculation")
        else:
            print("Using fallback spherical distance calculation")
    
    def forward(self, x, y):
        """
        Calculate spherical distance between x and y.
        x: Tensor of any shape that can be reshaped to [batch_size, embedding_dim]
        y: Tensor of any shape that can be reshaped to [batch_size, embedding_dim] or [1, embedding_dim]
        """
        if not self.use_cuda:
            # Fallback implementation using F.normalize
            return self._spherical_dist_fallback(x, y)
        
        # Reshape inputs to 2D if needed
        original_shape = x.shape
        original_y_shape = y.shape
        
        try:
            if x.dim() != 2:
                # Reshape to [batch_size, embedding_dim]
                last_dim = x.shape[-1]
                x = x.reshape(-1, last_dim)
                print(f"Reshaped x from {original_shape} to {x.shape}")
            
            if y.dim() != 2:
                # Reshape to [batch_size or 1, embedding_dim]
                last_dim = y.shape[-1]
                y = y.reshape(-1, last_dim)
                print(f"Reshaped y from {original_y_shape} to {y.shape}")
        except Exception as e:
            print(f"Error reshaping tensors: {e}")
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            return self._spherical_dist_fallback(x, y)
        
        # Convert to float32 if needed - CUDA kernel requires float32
        x_float = x.float() if x.dtype != torch.float32 else x
        y_float = y.float() if y.dtype != torch.float32 else y
        
        try:
            # Use the CUDA kernel
            if y.size(0) == 1 and x.size(0) > 1:
                result = spherical_dist_cuda.spherical_dist_loss_batch(x_float, y_float)
            else:
                result = spherical_dist_cuda.spherical_dist_loss(x_float, y_float)
        except RuntimeError as e:
            print(f"CUDA kernel error: {e}")
            print(f"x shape: {x.shape}, dtype: {x.dtype}")
            print(f"y shape: {y.shape}, dtype: {y.dtype}")
            print(f"Falling back to PyTorch implementation")
            # Fall back to PyTorch implementation
            return self._spherical_dist_fallback(x, y)
            
        # Convert result back to original dtype if needed
        if x.dtype != torch.float32:
            result = result.to(x.dtype)
        
        # Reshape result back to original batch dimensions if needed
        if original_shape != x.shape:
            # The result shape should be original_shape without the last dimension
            result_shape = original_shape[:-1]
            result = result.reshape(result_shape)
            
        return result
    
    def _spherical_dist_fallback(self, x, y):
        """Original PyTorch implementation for fallback"""
        # Save original shapes
        original_x_shape = x.shape
        original_y_shape = y.shape
        
        # Reshape if needed
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        if y.dim() != 2:
            y = y.reshape(-1, y.shape[-1])
            
        # Normalize and compute distance
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        result = x.sub(y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        
        # Reshape result back if needed
        if original_x_shape != x.shape:
            result_shape = original_x_shape[:-1]
            result = result.reshape(result_shape)
            
        return result

def patch_spherical_dist():
    """
    Replace the spherical_dist_loss function in pytti with the faster version.
    """
    from pytti.Perceptor.Prompt import spherical_dist_loss
    import pytti.Perceptor.Prompt
    
    # Create fast spherical distance calculator
    fast_dist = FastSphericalDistanceLoss()
    
    # Store original function
    original_spherical_dist = pytti.Perceptor.Prompt.spherical_dist_loss
    
    # Replace with fast version
    pytti.Perceptor.Prompt.spherical_dist_loss = fast_dist.forward
    
    return original_spherical_dist

if __name__ == "__main__":
    print("Run 'patch_spherical_dist()' to enable fast spherical distance calculation") 