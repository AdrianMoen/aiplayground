@echo off
echo Installing CUDA Toolkit and other dependencies for Phi-3 Mini fine-tuning
echo =======================================================================
echo.

REM Check if PowerShell is available
powershell -Command "& {exit 0}" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo PowerShell is not available on this system. Cannot continue.
    pause
    exit /b 1
)

echo Downloading CUDA Toolkit 12.1...
powershell -Command "& {Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe' -OutFile 'cuda_installer.exe'}"

echo.
echo Installing CUDA Toolkit 12.1...
echo This may take several minutes. Please be patient.
echo.
echo IMPORTANT: During installation, choose 'Express Installation' for simplicity.
echo.
pause
start "" "cuda_installer.exe"

echo.
echo After CUDA installation completes:
echo 1. Restart your computer
echo 2. Run the 'install_pytorch.bat' script to install PyTorch and other dependencies
echo.
pause
