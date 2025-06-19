@echo off
echo Installing PyTorch and dependencies for Phi-3 Mini fine-tuning
echo =========================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.8 or higher and try again.
    echo.
    echo You can download Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if CUDA is installed
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo CUDA is not installed or not in your PATH.
    echo Please run install_cuda.bat first, restart your computer, and try again.
    pause
    exit /b 1
)

REM Get CUDA version
for /f "tokens=* usebackq" %%a in (`nvcc --version ^| findstr "release"`) do (
    set CUDA_VERSION_LINE=%%a
)
echo Detected CUDA: %CUDA_VERSION_LINE%

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install -r requirements.txt

REM Verify installations
echo.
echo Verifying PyTorch installation with CUDA...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device count: {torch.cuda.device_count()}'); print(f'Current GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo Installation complete!
echo You can now run the fine-tuning pipeline with:
echo run_example_pipeline.bat
echo.
pause
