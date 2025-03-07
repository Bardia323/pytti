import torch
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from pytti.Image.DiffLogicCAImage import DiffLogicCAImage
from torchvision.transforms import functional as TF

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    if not os.path.exists(filename):
        print(f"Checkpoint {filename} not found")
        return 0, float('inf')
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    return epoch, loss

def display_progress(ca_image, target_image, step, loss):
    """Display current state and target image side by side"""
    current_img = ca_image.decode_image()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(current_img)
    plt.title(f"Current (Step {step}, Loss: {loss:.6f})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(target_image)
    plt.title("Target")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_difflogic_ca(target_image_path, width=64, height=64, ca_channels=16, 
                      perception_kernels=16, steps=20, epochs=1000, 
                      save_every=100, resume=False):
    """Train a DiffLogicCA to generate a target image"""
    # Load and resize target image
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image {target_image_path} not found")
    
    target_pil = Image.open(target_image_path).convert('RGB')
    target_pil = target_pil.resize((width, height), Image.LANCZOS)
    
    # Convert to tensor (0-1 range)
    target_tensor = TF.to_tensor(target_pil)
    
    # Create DiffLogicCA model
    model = DiffLogicCAImage(width, height, ca_channels=ca_channels, 
                           rgb_channels=3, perception_kernels=perception_kernels, 
                           steps=steps)
    
    # Initialize with center seed
    model.encode_random()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_file = "difflogic_ca_checkpoint.pth"
    
    if resume and os.path.exists(checkpoint_file):
        start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_file)
    
    # Training loop
    frames = []  # For creating GIF
    for epoch in range(start_epoch, epochs):
        # Reset to initial state
        model.encode_random()
        
        # Forward pass - run CA for multiple steps
        for _ in range(steps):
            model.step(hard=False)  # Use soft logic during training
        
        # Get final state RGB channels
        output = model.get_image_tensor()
        
        # Calculate loss (MSE between output and target)
        loss = torch.nn.functional.mse_loss(output, target_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Display progress
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                # Run with hard logic for visualization
                model.encode_random()
                for _ in range(steps):
                    model.step(hard=True)
                display_progress(model, target_pil, epoch+1, loss.item())
                
                # Save frame for GIF
                frames.append(model.decode_image())
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch+1, loss.item(), checkpoint_file)
            
            # Also save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, optimizer, epoch+1, loss.item(), "difflogic_ca_best.pth")
    
    # Save final model
    save_checkpoint(model, optimizer, epochs, loss.item(), "difflogic_ca_final.pth")
    
    # Save the training progression as a GIF
    if frames:
        frames[0].save('difflogic_ca_training.gif',
                      save_all=True, append_images=frames[1:], 
                      optimize=False, duration=200, loop=0)
        print("Training progression saved as difflogic_ca_training.gif")
    
    return model

def demonstrate_trained_model(model_path, target_image_path, width=64, height=64, 
                             ca_channels=16, perception_kernels=16, steps=20):
    """Load a trained model and demonstrate it"""
    # Create model with same parameters
    model = DiffLogicCAImage(width, height, ca_channels=ca_channels, 
                           rgb_channels=3, perception_kernels=perception_kernels, 
                           steps=steps)
    
    # Load trained parameters
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")
    
    # Load target image for comparison
    target_pil = Image.open(target_image_path).convert('RGB')
    target_pil = target_pil.resize((width, height), Image.LANCZOS)
    
    # Generate sequence
    frames = []
    model.encode_random()
    frames.append(model.decode_image())
    
    for i in range(steps):
        model.step(hard=True)
        if (i + 1) % 2 == 0 or i == steps - 1:
            frames.append(model.decode_image())
            display_progress(model, target_pil, i+1, 0)
    
    # Save as GIF
    frames[0].save('difflogic_ca_trained_demo.gif',
                  save_all=True, append_images=frames[1:], 
                  optimize=False, duration=200, loop=0)
    print("Demonstration saved as difflogic_ca_trained_demo.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiffLogicCA on a target image")
    parser.add_argument("--target", type=str, required=True, help="Path to target image")
    parser.add_argument("--width", type=int, default=64, help="Image width")
    parser.add_argument("--height", type=int, default=64, help="Image height")
    parser.add_argument("--channels", type=int, default=16, help="Number of CA channels")
    parser.add_argument("--kernels", type=int, default=16, help="Number of perception kernels")
    parser.add_argument("--steps", type=int, default=20, help="Number of CA steps")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--demo", type=str, help="Path to model checkpoint for demo (skip training)")
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_trained_model(args.demo, args.target, args.width, args.height, 
                                 args.channels, args.kernels, args.steps)
    else:
        train_difflogic_ca(args.target, args.width, args.height, args.channels, 
                          args.kernels, args.steps, args.epochs, args.save_every, 
                          args.resume) 