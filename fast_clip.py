import os
import torch
import torch.nn as nn
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load
from torch.autograd import Function
from typing import Tuple, List
import math

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for fast_clip")

# Compile the CUDA extension
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>

// CUDA kernel for parallel cutout generation
__global__ void generate_cutouts_kernel(
    const float* input,
    float* output,
    const float* sizes,
    const float* offsets,
    int batch_size,
    int channels,
    int height,
    int width,
    int cutout_size,
    int num_cutouts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cutouts) return;
    
    int offset_x = static_cast<int>(offsets[idx * 2] * width);
    int offset_y = static_cast<int>(offsets[idx * 2 + 1] * height);
    int size = static_cast<int>(sizes[idx] * min(width, height));
    
    // Clamp values to valid ranges
    offset_x = max(0, min(width - size, offset_x));
    offset_y = max(0, min(height - size, offset_y));
    
    // Bilinear resize the cutout to target size
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < cutout_size; h++) {
            for (int w = 0; w < cutout_size; w++) {
                float src_h = offset_y + (float)h * size / cutout_size;
                float src_w = offset_x + (float)w * size / cutout_size;
                
                int src_h1 = static_cast<int>(floor(src_h));
                int src_w1 = static_cast<int>(floor(src_w));
                int src_h2 = min(src_h1 + 1, height - 1);
                int src_w2 = min(src_w1 + 1, width - 1);
                
                float h_weight = src_h - src_h1;
                float w_weight = src_w - src_w1;
                
                float val1 = input[c * height * width + src_h1 * width + src_w1];
                float val2 = input[c * height * width + src_h1 * width + src_w2];
                float val3 = input[c * height * width + src_h2 * width + src_w1];
                float val4 = input[c * height * width + src_h2 * width + src_w2];
                
                float val = (1 - h_weight) * (1 - w_weight) * val1 +
                           (1 - h_weight) * w_weight * val2 +
                           h_weight * (1 - w_weight) * val3 +
                           h_weight * w_weight * val4;
                           
                output[idx * channels * cutout_size * cutout_size + 
                       c * cutout_size * cutout_size + 
                       h * cutout_size + w] = val;
            }
        }
    }
}

// CUDA kernel for spherical distance calculation
__global__ void spherical_dist_kernel(
    const float* x,
    const float* y,
    float* output,
    int batch_size,
    int embedding_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Compute dot product of normalized vectors
    float dot_product = 0.0f;
    float x_norm = 0.0f;
    float y_norm = 0.0f;
    
    for (int i = 0; i < embedding_dim; i++) {
        float x_val = x[idx * embedding_dim + i];
        float y_val = y[idx * embedding_dim + i];
        x_norm += x_val * x_val;
        y_norm += y_val * y_val;
        dot_product += x_val * y_val;
    }
    
    x_norm = sqrt(x_norm);
    y_norm = sqrt(y_norm);
    
    // Avoid division by zero
    if (x_norm < 1e-6 || y_norm < 1e-6) {
        output[idx] = 0.0f;
        return;
    }
    
    // Compute cosine similarity
    float cos_sim = dot_product / (x_norm * y_norm);
    cos_sim = min(max(cos_sim, -1.0f), 1.0f);  // Clamp to [-1, 1]
    
    // Compute spherical distance
    float dist = acos(cos_sim);
    output[idx] = 2.0f * pow(sin(dist / 2.0f), 2);
}

// Python bindings
torch::Tensor generate_cutouts_cuda(
    torch::Tensor input,
    torch::Tensor sizes,
    torch::Tensor offsets,
    int cutout_size,
    int num_cutouts
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    auto output = torch::zeros({num_cutouts, channels, cutout_size, cutout_size}, 
                               input.options());
                               
    const int threads = 256;
    const int blocks = (num_cutouts + threads - 1) / threads;
    
    generate_cutouts_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sizes.data_ptr<float>(),
        offsets.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        cutout_size,
        num_cutouts
    );
    
    return output;
}

torch::Tensor spherical_dist_cuda(
    torch::Tensor x,
    torch::Tensor y
) {
    const int batch_size = x.size(0);
    const int embedding_dim = x.size(1);
    
    auto output = torch::zeros({batch_size}, x.options());
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    spherical_dist_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        embedding_dim
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_cutouts", &generate_cutouts_cuda, "Generate Cutouts (CUDA)");
    m.def("spherical_dist", &spherical_dist_cuda, "Spherical Distance (CUDA)");
}
"""

# Write CUDA code to a temporary file
cuda_path = os.path.join(os.path.dirname(__file__), "fast_clip_cuda.cu")
with open(cuda_path, "w") as f:
    f.write(cuda_source)

# Compile the extension
fast_clip_cuda = load(
    name="fast_clip_cuda",
    sources=[cuda_path],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

class FastCutoutGenerator(Function):
    @staticmethod
    def forward(ctx, image, sizes, offsets, cutout_size, num_cutouts):
        return fast_clip_cuda.generate_cutouts(image, sizes, offsets, cutout_size, num_cutouts)
    
    @staticmethod
    def backward(ctx, grad_output):
        # We don't need backward for cutout generation
        return None, None, None, None, None

class FastSphericalDist(Function):
    @staticmethod
    def forward(ctx, x, y):
        result = fast_clip_cuda.spherical_dist(x, y)
        ctx.save_for_backward(x, y, result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, result = ctx.saved_tensors
        
        # Gradient for spherical distance: grad * d(arcsin²)/dx
        # Compute normalized vectors
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        y_norm = torch.nn.functional.normalize(y, dim=-1)
        
        # Compute gradients
        x_grad = grad_output.unsqueeze(-1) * (x_norm - y_norm)
        y_grad = grad_output.unsqueeze(-1) * (y_norm - x_norm)
        
        return x_grad, y_grad

def fast_generate_cutouts(image, sizes, offsets, cutout_size, num_cutouts):
    """
    Generate cutouts using CUDA acceleration
    
    Args:
        image: Input image tensor [B, C, H, W]
        sizes: Tensor of relative sizes [num_cutouts]
        offsets: Tensor of relative offsets [num_cutouts, 2]
        cutout_size: Size of the output cutouts
        num_cutouts: Number of cutouts to generate
        
    Returns:
        Tensor of cutouts [num_cutouts, C, cutout_size, cutout_size]
    """
    return FastCutoutGenerator.apply(image, sizes, offsets, cutout_size, num_cutouts)

def fast_spherical_dist_loss(x, y):
    """
    Compute spherical distance loss with CUDA acceleration
    
    Args:
        x: First embedding tensor [B, D]
        y: Second embedding tensor [B, D]
        
    Returns:
        Tensor of spherical distances [B]
    """
    return FastSphericalDist.apply(x, y)

class FastClipTrainer:
    """Optimized CLIP training with CUDA kernels"""
    
    def __init__(self, clip_models, cutn=40, cut_pow=1.5):
        """
        Initialize the fast CLIP trainer
        
        Args:
            clip_models: List of CLIP models
            cutn: Number of cutouts
            cut_pow: Power for size distribution
        """
        self.clip_models = clip_models
        self.cut_sizes = [p.visual.input_resolution for p in clip_models]
        self.cutn = cutn
        self.cut_pow = cut_pow
        
    def prepare_cutout_params(self, width, height, device=torch.device("cuda")):
        """Generate parameters for cutouts"""
        max_size = min(width, height)
        
        # Generate random sizes
        sizes = torch.zeros(self.cutn, device=device).normal_(mean=0.8, std=0.3)
        sizes = sizes.clamp(min(self.cut_sizes) / max_size, 1.0) ** self.cut_pow
        
        # Generate random offsets
        offsets_x = torch.rand(self.cutn, device=device)
        offsets_y = torch.rand(self.cutn, device=device)
        offsets = torch.stack([offsets_x, offsets_y], dim=-1)
        
        return sizes, offsets
        
    def process_image(self, image_tensor):
        """
        Process an image through fast CLIP embedding
        
        Args:
            image_tensor: Image tensor [1, C, H, W]
            
        Returns:
            List of embedding tensors for each CLIP model
        """
        height, width = image_tensor.shape[2], image_tensor.shape[3]
        image_embeds = []
        
        # Generate cutout parameters once
        sizes, offsets = self.prepare_cutout_params(width, height, image_tensor.device)
        
        # Process each CLIP model
        for i, (cut_size, model) in enumerate(zip(self.cut_sizes, self.clip_models)):
            # Generate cutouts
            cutouts = fast_generate_cutouts(
                image_tensor, 
                sizes, 
                offsets, 
                cut_size, 
                self.cutn
            )
            
            # Normalize cutouts
            from pytti import normalize
            cutouts = normalize(cutouts)
            
            # Get embeddings
            with torch.no_grad():
                embedding = model.encode_image(cutouts).float()
            
            image_embeds.append(embedding.unsqueeze(0))
            
        return image_embeds
        
    def compute_loss(self, image_embeds, text_embeds):
        """
        Compute loss between image and text embeddings
        
        Args:
            image_embeds: List of image embedding tensors
            text_embeds: List of text embedding tensors
            
        Returns:
            Loss value
        """
        total_loss = 0
        
        for img_embed, txt_embed in zip(image_embeds, text_embeds):
            # Compute spherical distance
            dist = fast_spherical_dist_loss(img_embed, txt_embed)
            total_loss += dist.mean()
            
        return total_loss / len(image_embeds) 