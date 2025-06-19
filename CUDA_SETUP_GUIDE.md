# CUDA and RTX 4090 Setup Guide for Phi-3 Mini Fine-tuning

This guide walks you through the complete setup process for using your NVIDIA RTX 4090 GPU to fine-tune the Phi-3 Mini model.

## Prerequisites

- Windows 10 or 11 operating system
- NVIDIA GeForce RTX 4090 GPU
- At least 32GB system RAM recommended
- At least 100GB free disk space

## Installation Steps

### 1. Install CUDA Toolkit

The CUDA Toolkit provides the development environment for creating high-performance GPU-accelerated applications.

1. Run `install_cuda.bat`
2. Follow the on-screen instructions (recommended to use Express Installation)
3. Restart your computer after installation

### 2. Install PyTorch and Dependencies

After restarting your computer:

1. Run `install_pytorch.bat`
2. This will install PyTorch with CUDA support and all other required dependencies

### 3. Verify Your Installation

1. Run `check_bnb.bat` to verify that bitsandbytes is correctly installed and compatible with your CUDA setup
2. Ensure the output shows that CUDA is available and the RTX 4090 is detected

### 4. Run the Fine-tuning Pipeline

1. Run `run_example_pipeline.bat` to fine-tune the Phi-3 Mini model on the example dataset

## Troubleshooting

### CUDA Not Found

If you see errors related to CUDA not being found:

1. Ensure you've installed the CUDA Toolkit using `install_cuda.bat`
2. Make sure you've restarted your computer after CUDA installation
3. Verify that NVIDIA drivers are up to date

### bitsandbytes Issues

If you encounter errors with bitsandbytes:

1. Try reinstalling with: `pip uninstall bitsandbytes -y && pip install bitsandbytes`
2. Verify that the installed version is compatible with your CUDA version

### Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce batch size using the `--batch_size` parameter
2. Enable gradient accumulation with `--gradient_accumulation_steps 4` (or higher)

## Additional Resources

- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/
- PyTorch CUDA Setup Guide: https://pytorch.org/docs/stable/notes/cuda.html
- bitsandbytes GitHub: https://github.com/TimDettmers/bitsandbytes

## Next Steps

After successful installation and verification, refer to the `rtx4090_optimization_tips.md` file for specific optimizations to get the best performance from your RTX 4090 GPU.
