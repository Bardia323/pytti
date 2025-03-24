@echo off
echo Setting up CUDA environment for PyTTI acceleration

REM Try to find CUDA installation
set FOUND=0

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
    set FOUND=1
    goto :found
)

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
    set FOUND=1
    goto :found
)

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
    set FOUND=1
    goto :found
)

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
    set FOUND=1
    goto :found
)

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5
    set FOUND=1
    goto :found
)

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4
    set FOUND=1
    goto :found
)

IF EXIST "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
    set FOUND=1
    goto :found
)

:found
if %FOUND%==1 (
    echo Found CUDA at %CUDA_HOME%
    echo Setting CUDA_HOME environment variable...
    setx CUDA_HOME "%CUDA_HOME%"
    echo Installing ninja build system...
    pip install ninja
    echo.
    echo Setup complete! Please restart your Python environment.
) else (
    echo Could not find CUDA installation.
    echo Please install CUDA from https://developer.nvidia.com/cuda-downloads
    echo Then run this script again.
)

pause 