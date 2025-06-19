@echo off
echo Installing CUDA Toolkit and other dependencies for Phi-3 Mini fine-tuning
echo =======================================================================
echo.

cd %~dp0\..\..

REM Check if PowerShell is available
powershell -Command "& {exit 0}" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo PowerShell is not available on this system. Cannot continue.
    exit /b 1
)

REM Check if CUDA is already installed
nvcc --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA already appears to be installed:
    nvcc --version
    echo.
    echo Would you like to continue with the installation anyway? (Y/N)
    set /p continue_choice=
    if /i NOT "%continue_choice%"=="Y" (
        echo Installation aborted by user.
        exit /b 0
    )
)

REM Check if installer exists in current directory
if exist "cuda_12.1.0_531.14_windows.exe" (
    echo CUDA installer found in current directory.
    set installer_path=%CD%\cuda_12.1.0_531.14_windows.exe
) else (
    echo Downloading CUDA Toolkit 12.1...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe' -OutFile 'cuda_installer.exe'}"
    set installer_path=%CD%\cuda_installer.exe
)

echo.
echo Installing CUDA Toolkit 12.1...
echo This may take several minutes. Please be patient.
echo.
echo IMPORTANT: During installation, choose 'Express Installation' for simplicity.
echo.
pause
start "" "%installer_path%"

echo.
echo After CUDA installation completes:
echo 1. Restart your computer
echo 2. Run the 'setup\cuda\install_pytorch.bat' script to install PyTorch and other dependencies
echo.
