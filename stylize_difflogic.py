import torch
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from pytti.Image.DiffLogicCAImage import DiffLogicCAImage
from torchvision.transforms import functional as TF, transforms

def load_images(content_path, style_path, size=(256, 256)):
    """Load content and style images"""
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Content image {content_path} not found")
    if not os.path.exists(style_path):
        raise FileNotFoundError(f"Style image {style_path} not found")
    
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    # Resize images
    content_img = content_img.resize(size, Image.LANCZOS)
    style_img = style_img.resize(size, Image.LANCZOS)
    
    return content_img, style_img

def display_images(content_img, style_img, result_img, title="Image Stylization"):
    """Display all three images side by side"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(content_img)
    plt.title("Content")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(style_img)
    plt.title("Style")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title("Result")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def gram_matrix(input_tensor):
    """Calculate Gram Matrix for style loss"""
    # reshape tensor
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    
    # compute gram matrix
    G = torch.mm(features, features.t())
    
    # normalize
    return G.div(b * c * h * w)

def stylize_with_difflogic(content_path, style_path, ca_channels=32, 
                          perception_kernels=16, steps=15, iterations=300, 
                          size=(256, 256), content_weight=1.0, style_weight=1e6):
    """Stylize a content image with a style image using DiffLogicCA"""
    content_img, style_img = load_images(content_path, style_path, size)
    
    # Convert images to tensors
    content_tensor = TF.to_tensor(content_img).unsqueeze(0)
    style_tensor = TF.to_tensor(style_img).unsqueeze(0)
    
    # Create DiffLogicCA model
    width, height = size
    model = DiffLogicCAImage(width, height, ca_channels=ca_channels, 
                           rgb_channels=3, perception_kernels=perception_kernels, 
                           steps=steps)
    
    # Initialize with content image
    model.encode_image(content_img)
    
    # Define feature extractor (for content and style representation)
    # Here we'll use a simplified approach with basic convolutions
    feature_extractor = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU()
    ).to(model.device)
    
    # Extract content and style features
    with torch.no_grad():
        content_features = feature_extractor(content_tensor.to(model.device))
        style_features = feature_extractor(style_tensor.to(model.device))
        style_gram = gram_matrix(style_features)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    frames = []
    for iteration in range(iterations):
        # Run the CA for multiple steps
        model.run_ca(steps, hard=False)  # Use soft logic during training
        
        # Get output image
        output = model.get_image_tensor().unsqueeze(0)
        
        # Extract features from output
        output_features = feature_extractor(output)
        output_gram = gram_matrix(output_features)
        
        # Calculate losses
        content_loss = torch.nn.functional.mse_loss(output_features, content_features)
        style_loss = torch.nn.functional.mse_loss(output_gram, style_gram)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Update model parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{iterations}, "
                 f"Loss: {total_loss.item():.4f}, "
                 f"Content: {content_loss.item():.4f}, "
                 f"Style: {style_loss.item():.4f}")
        
        # Display progress and save frames
        if (iteration + 1) % 50 == 0 or iteration == iterations - 1:
            with torch.no_grad():
                # Run with hard logic for visualization
                model.run_ca(steps, hard=True)
                result_img = model.decode_image()
                display_images(content_img, style_img, result_img, 
                             f"Stylization (Iteration {iteration+1})")
                frames.append(result_img)
    
    # Save final result
    final_result = model.decode_image()
    final_result.save("difflogic_stylized.png")
    print("Final stylized image saved as difflogic_stylized.png")
    
    # Save the stylization process as a GIF
    if frames:
        frames[0].save('difflogic_stylization.gif',
                      save_all=True, append_images=frames[1:], 
                      optimize=False, duration=300, loop=0)
        print("Stylization process saved as difflogic_stylization.gif")
    
    return final_result

def game_of_life_style(content_path, size=(256, 256), ca_channels=16, 
                     perception_kernels=8, steps=25, iterations=50):
    """Apply a Game of Life-inspired style to an image using DiffLogicCA"""
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Content image {content_path} not found")
    
    content_img = Image.open(content_path).convert('RGB')
    content_img = content_img.resize(size, Image.LANCZOS)
    
    # Create DiffLogicCA model
    width, height = size
    model = DiffLogicCAImage(width, height, ca_channels=ca_channels, 
                           rgb_channels=3, perception_kernels=perception_kernels, 
                           steps=steps)
    
    # Initialize with content image
    model.encode_image(content_img)
    
    # Create a Game of Life-like style by training on a simple pattern
    # Use only a few iterations to keep some of the original content
    frames = []
    frames.append(model.decode_image())
    
    for iteration in range(iterations):
        # Run the CA for multiple steps
        model.run_ca(steps, hard=True)
        
        if iteration % 5 == 0 or iteration == iterations - 1:
            result_img = model.decode_image()
            frames.append(result_img)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(content_img)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(result_img)
            plt.title(f"Game of Life Style (Step {iteration+1})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # Save final result
    final_result = model.decode_image()
    final_result.save("difflogic_game_of_life.png")
    print("Final Game of Life styled image saved as difflogic_game_of_life.png")
    
    # Save the process as a GIF
    if frames:
        frames[0].save('difflogic_game_of_life.gif',
                      save_all=True, append_images=frames[1:], 
                      optimize=False, duration=300, loop=0)
        print("Process saved as difflogic_game_of_life.gif")
    
    return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffLogicCA Image Stylization")
    subparsers = parser.add_subparsers(dest="command", help="Stylization mode")
    
    # Parser for normal stylization
    style_parser = subparsers.add_parser("stylize", help="Traditional content+style transfer")
    style_parser.add_argument("--content", type=str, required=True, help="Path to content image")
    style_parser.add_argument("--style", type=str, required=True, help="Path to style image")
    style_parser.add_argument("--size", type=int, default=256, help="Output size (square)")
    style_parser.add_argument("--channels", type=int, default=32, help="Number of CA channels")
    style_parser.add_argument("--steps", type=int, default=15, help="Number of CA steps")
    style_parser.add_argument("--iterations", type=int, default=300, help="Training iterations")
    
    # Parser for Game of Life mode
    gol_parser = subparsers.add_parser("gameoflife", help="Game of Life-inspired effect")
    gol_parser.add_argument("--content", type=str, required=True, help="Path to content image")
    gol_parser.add_argument("--size", type=int, default=256, help="Output size (square)")
    gol_parser.add_argument("--channels", type=int, default=16, help="Number of CA channels")
    gol_parser.add_argument("--steps", type=int, default=25, help="Number of CA steps")
    gol_parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    
    args = parser.parse_args()
    
    if args.command == "stylize":
        stylize_with_difflogic(args.content, args.style, ca_channels=args.channels,
                              steps=args.steps, iterations=args.iterations, 
                              size=(args.size, args.size))
    elif args.command == "gameoflife":
        game_of_life_style(args.content, size=(args.size, args.size), 
                          ca_channels=args.channels, steps=args.steps,
                          iterations=args.iterations)
    else:
        parser.print_help() 