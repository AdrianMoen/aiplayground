@echo off
echo Fixing bitsandbytes and CUDA integration
echo =======================================
echo.

cd %~dp0\..\..

REM Check if CUDA is installed
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: CUDA nvcc compiler not found in PATH.
    echo This might indicate CUDA is not properly installed or not in your PATH.
    echo.
    echo Continuing anyway as CUDA might be installed but not in PATH...
    echo.
)

REM Check CUDA version
echo Checking for CUDA installation...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: nvidia-smi command failed. This might indicate driver issues.
    echo.
    echo Continuing anyway as CUDA installation might still be valid...
    echo.
)

REM Uninstall current bitsandbytes
echo Uninstalling current bitsandbytes...
pip uninstall -y bitsandbytes

REM Install CUDA-enabled bitsandbytes
echo.
echo Installing CUDA-enabled bitsandbytes...
pip install bitsandbytes

REM Verify if torch is installed with CUDA
echo.
echo Checking PyTorch CUDA support...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 

REM If torch doesn't have CUDA, reinstall it
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo PyTorch does not have CUDA support. Reinstalling PyTorch with CUDA...
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

REM Reinstall other key dependencies
echo.
echo Reinstalling key dependencies...
pip uninstall -y accelerate peft transformers
pip install accelerate peft transformers

echo.
echo Verifying bitsandbytes installation...
python -c "import bitsandbytes as bnb; print(f'bitsandbytes version: {bnb.__version__}'); print('CUDA modules available:'); print(bnb.cuda_setup.get_available_modules() if hasattr(bnb, 'cuda_setup') else 'No CUDA support in bitsandbytes'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo Installation complete! Please check the output above to verify CUDA support.
echo If you see errors, you may need to restart your computer or check your CUDA installation.
echo.
