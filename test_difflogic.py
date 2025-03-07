import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytti.Image.DiffLogicCAImage import DiffLogicCAImage
from torchvision.transforms import functional as TF
import os

def display_image(img, title=None):
    """Display a PIL image with optional title"""
    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def display_state_channels(ca_image, num_channels=8, figsize=(16, 16)):
    """Display the state channels of a DiffLogicCAImage"""
    state = ca_image.state.detach().cpu().numpy()
    height, width, channels = state.shape
    
    # Determine grid dimensions
    grid_size = min(num_channels, channels)
    grid_cols = int(np.ceil(np.sqrt(grid_size)))
    grid_rows = int(np.ceil(grid_size / grid_cols))
    
    plt.figure(figsize=figsize)
    for i in range(grid_size):
        plt.subplot(grid_rows, grid_cols, i+1)
        plt.imshow(state[:, :, i], cmap='gray')
        plt.title(f"Channel {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_growing_pattern(width=64, height=64, ca_channels=16, steps=20, save_gif=True):
    """Test the DiffLogicCA image with a growing pattern from a center seed"""
    print("Creating DiffLogicCA image...")
    ca_image = DiffLogicCAImage(width, height, ca_channels=ca_channels, steps=steps)
    
    # Initialize with just a center seed
    print("Initializing with center seed...")
    ca_image.encode_random()  # This will use the center seed logic
    
    frames = []
    if save_gif:
        # Save initial state
        img = ca_image.decode_image()
        frames.append(img)
    
    # Run simulation
    print(f"Running CA for {steps} steps...")
    for i in range(steps):
        ca_image.step(hard=True)
        
        if i % 5 == 0 or i == steps - 1:
            print(f"Step {i+1}/{steps}")
            # Display current state
            img = ca_image.decode_image()
            display_image(img, title=f"Step {i+1}")
            display_state_channels(ca_image)
            
            if save_gif:
                frames.append(img)
    
    # Save the frames as a GIF
    if save_gif and frames:
        print("Saving GIF...")
        frames[0].save('difflogic_ca_growth.gif',
                      save_all=True, append_images=frames[1:], 
                      optimize=False, duration=200, loop=0)
        print("Saved as difflogic_ca_growth.gif")

def test_image_generation(width=64, height=64, ca_channels=16, steps=20, image_path=None, save_gif=True):
    """Test the DiffLogicCA image with an input image"""
    print("Creating DiffLogicCA image...")
    ca_image = DiffLogicCAImage(width, height, ca_channels=ca_channels, steps=steps)
    
    # Load or create an input image
    if image_path and os.path.exists(image_path):
        print(f"Loading image from {image_path}")
        pil_image = Image.open(image_path).convert('RGB')
        pil_image = pil_image.resize((width, height), Image.LANCZOS)
    else:
        print("No image provided, creating a simple pattern")
        # Create a simple pattern (checkerboard)
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        pattern[::8, ::8] = 255  # White dots on a black background
        pil_image = Image.fromarray(pattern)
    
    display_image(pil_image, title="Input Image")
    
    # Encode the image
    print("Encoding image...")
    ca_image.encode_image(pil_image)
    
    frames = []
    if save_gif:
        # Save initial state
        img = ca_image.decode_image()
        frames.append(img)
    
    # Run simulation
    print(f"Running CA for {steps} steps...")
    for i in range(steps):
        ca_image.step(hard=True)
        
        if i % 5 == 0 or i == steps - 1:
            print(f"Step {i+1}/{steps}")
            # Display current state
            img = ca_image.decode_image()
            display_image(img, title=f"Step {i+1}")
            display_state_channels(ca_image)
            
            if save_gif:
                frames.append(img)
    
    # Save the frames as a GIF
    if save_gif and frames:
        print("Saving GIF...")
        frames[0].save('difflogic_ca_evolution.gif',
                      save_all=True, append_images=frames[1:], 
                      optimize=False, duration=200, loop=0)
        print("Saved as difflogic_ca_evolution.gif")

if __name__ == "__main__":
    print("DiffLogicCA Image Test")
    print("1. Test Growing Pattern")
    print("2. Test Image Generation")
    choice = input("Enter your choice (1-2): ")
    
    if choice == "1":
        test_growing_pattern()
    elif choice == "2":
        image_path = input("Enter path to image file (leave empty for default pattern): ")
        test_image_generation(image_path=image_path if image_path else None)
    else:
        print("Invalid choice!") 