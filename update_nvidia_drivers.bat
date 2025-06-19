@echo off
echo Checking and updating NVIDIA drivers
echo ==================================
echo.

REM Check for administrative privileges
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo This script requires administrative privileges.
    echo Please run as administrator.
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
powershell -Command "& {$gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*'}; if ($gpu) { Write-Host ('NVIDIA GPU found: ' + $gpu.Name) } else { Write-Host 'No NVIDIA GPU detected'; exit 1 }}"
if %ERRORLEVEL% neq 0 (
    echo No NVIDIA GPU detected. Cannot continue.
    pause
    exit /b 1
)

echo.
echo Current NVIDIA Driver information:
powershell -Command "& {$gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*'}; Write-Host ('Name: ' + $gpu.Name); Write-Host ('Driver Version: ' + $gpu.DriverVersion)}"

echo.
echo Would you like to download the latest NVIDIA driver? (Y/N)
set /p download_choice=

if /i "%download_choice%"=="Y" (
    echo.
    echo Opening NVIDIA driver download page...
    start "" "https://www.nvidia.com/Download/index.aspx"
    echo.
    echo Please:
    echo 1. Select your GPU model (GeForce RTX 4090)
    echo 2. Download and install the latest driver
    echo 3. Restart your computer after installation
    echo.
) else (
    echo.
    echo Skipping driver download.
)

echo.
echo If you've already installed the latest drivers, you can now proceed with CUDA installation.
echo Run install_cuda.bat to install the CUDA Toolkit.
echo.
pause
