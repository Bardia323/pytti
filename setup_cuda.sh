#!/bin/bash
echo "Setting up CUDA environment for PyTTI acceleration"

# Try to find CUDA installation
FOUND=0

# Common CUDA installation paths on Linux
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-11.8"
    "/usr/local/cuda-11.7"
    "/usr/local/cuda-11.6"
    "/usr/local/cuda-11.5"
    "/usr/local/cuda-11.4"
    "/usr/local/cuda-11.3"
    "/opt/cuda"
)

for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CUDA_HOME=$path
        FOUND=1
        break
    fi
done

if [ $FOUND -eq 1 ]; then
    echo "Found CUDA at $CUDA_HOME"
    echo "Setting CUDA_HOME environment variable..."
    
    # For immediate use
    export CUDA_HOME="$CUDA_HOME"
    
    # Add to .bashrc for persistence
    if ! grep -q "export CUDA_HOME=" ~/.bashrc; then
        echo "export CUDA_HOME=\"$CUDA_HOME\"" >> ~/.bashrc
        echo "export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        echo "Added CUDA_HOME to ~/.bashrc"
    else
        echo "CUDA_HOME already in ~/.bashrc, updating it"
        sed -i "s|export CUDA_HOME=.*|export CUDA_HOME=\"$CUDA_HOME\"|g" ~/.bashrc
    fi
    
    echo "Installing ninja build system..."
    pip install ninja
    
    echo ""
    echo "Setup complete! For the changes to take effect, either:"
    echo "1. Restart your terminal"
    echo "2. Run: source ~/.bashrc"
    echo "3. Run: export CUDA_HOME=\"$CUDA_HOME\""
else
    echo "Could not find CUDA installation."
    echo "Please install CUDA from https://developer.nvidia.com/cuda-downloads"
    echo "Common installation paths are:"
    echo "  - /usr/local/cuda"
    echo "  - /usr/local/cuda-X.Y (where X.Y is the version)"
    echo "After installing, run this script again."
fi 