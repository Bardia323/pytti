import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pytti
from pytti import DEVICE, normalize, format_input
from pytti.Image import MultiResImage
from Perceptor import Prompt, CLIP_PERCEPTORS
from Perceptor.FastEmbedder import FastHDMultiClipEmbedder
from CLIP import clip
from FastImageGuide import RealTimeImageGuide

def to_pil(tensor):
    """Convert tensor to PIL image"""
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def create_text_prompt(text, perceptors, weight=1.0, stop=0.0):
    """Create a CLIP text prompt"""
    # Tokenize text
    tokens = clip.tokenize([text]).to(DEVICE)
    
    # Get embeddings for each model
    with torch.no_grad():
        embeds = torch.cat([p.encode_text(tokens).float() for p in perceptors])
    
    # Create prompt
    return Prompt(embeds, weight, stop, text, text)

def realtime_training_example(width=512, height=512, num_steps=300, prompt_text="a beautiful sunset over mountains"):
    """
    Run a real-time training example
    
    Args:
        width: Image width
        height: Image height
        num_steps: Number of steps to run
        prompt_text: Text prompt to use
    """
    print("Initializing CLIP models...")
    if not CLIP_PERCEPTORS:
        from Perceptor import init_clip
        init_clip(["ViT-B/32"])
    
    print("Creating image representation...")
    # Create a multi-resolution image
    image = MultiResImage(width, height, gamma=1.0)
    image.encode_random()  # Start with random noise
    
    print("Setting up embedder...")
    # Create fast embedder with fewer cutouts for speed
    embedder = FastHDMultiClipEmbedder(perceptors=CLIP_PERCEPTORS, cutn=20, cut_pow=1.0)
    
    print("Creating text prompt...")
    # Create text prompt
    prompt = create_text_prompt(prompt_text, CLIP_PERCEPTORS, weight=1.0)
    
    print("Setting up real-time trainer...")
    # Create real-time guide
    guide = RealTimeImageGuide(
        image_rep=image,
        embedder=embedder,
        lr=0.1,
        use_amp=True,
        jit_compile=True,
        max_cutouts=20
    )
    
    # Create figure for visualization
    plt.figure(figsize=(12, 6))
    
    print(f"Starting real-time training for prompt: '{prompt_text}'")
    fps_values = []
    current_fps = 0
    
    for step in range(num_steps):
        # Perform one step of training
        current_image = guide.train_single_step([prompt])
        
        # Get current FPS
        new_fps = guide.get_fps()
        if new_fps > 0:
            current_fps = new_fps
            fps_values.append(current_fps)
        
        # Display progress every 10 steps
        if step % 10 == 0 or step == num_steps - 1:
            # Clear output
            clear_output(wait=True)
            
            # Display image and stats
            plt.clf()
            
            # Image
            plt.subplot(1, 2, 1)
            plt.imshow(to_pil(current_image))
            plt.title(f"Step {step+1}/{num_steps}")
            plt.axis('off')
            
            # FPS graph
            plt.subplot(1, 2, 2)
            plt.plot(fps_values, 'b-')
            plt.title(f"FPS: {current_fps:.2f}")
            plt.xlabel("Step")
            plt.ylabel("FPS")
            plt.grid(True)
            
            plt.tight_layout()
            plt.pause(0.01)
            
            print(f"Step {step+1}/{num_steps} - FPS: {current_fps:.2f}")
    
    # Final image
    final_image = to_pil(image.get_image_tensor())
    
    # Save final image
    final_image.save("realtime_result.png")
    print(f"Final image saved to 'realtime_result.png'")
    
    # Calculate average FPS
    avg_fps = sum(fps_values) / len(fps_values)
    print(f"Average FPS: {avg_fps:.2f}")
    
    return final_image

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Real-time CLIP training example")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps")
    parser.add_argument("--prompt", type=str, default="a beautiful sunset over mountains", help="Text prompt")
    args = parser.parse_args()
    
    # Run example
    realtime_training_example(
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        prompt_text=args.prompt
    ) 