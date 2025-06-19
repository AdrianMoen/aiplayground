@echo off
echo System Requirements Check for Phi-3 Mini Fine-tuning
echo =================================================
echo.

cd %~dp0\..\..

echo Checking CPU...
powershell -Command "& {$cpu = Get-CimInstance -ClassName Win32_Processor; Write-Host ('CPU: ' + $cpu.Name); Write-Host ('Cores: ' + $cpu.NumberOfCores); Write-Host ('Logical Processors: ' + $cpu.NumberOfLogicalProcessors)}"

echo.
echo Checking RAM...
powershell -Command "& {$ram = Get-CimInstance -ClassName Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum; $ramGB = [math]::Round($ram.Sum / 1GB, 2); Write-Host ('Total RAM: ' + $ramGB + ' GB')}"

echo.
echo Checking GPU...
powershell -Command "& {$gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*'}; if ($gpu) { Write-Host ('NVIDIA GPU found: ' + $gpu.Name); Write-Host ('Driver Version: ' + $gpu.DriverVersion) } else { Write-Host 'No NVIDIA GPU detected'; }}"

echo.
echo Checking Disk Space...
powershell -Command "& {$disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter 'DeviceID=\"C:\"'; $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2); $totalSpaceGB = [math]::Round($disk.Size / 1GB, 2); Write-Host ('C: Drive - Free: ' + $freeSpaceGB + ' GB / Total: ' + $totalSpaceGB + ' GB')}"

echo.
echo Checking Python...
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install Python 3.8 or higher.
) else (
    for /f "tokens=*" %%a in ('python --version 2^>^&1') do set python_version=%%a
    echo %python_version% detected
)

echo.
echo Checking CUDA...
nvcc --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA is installed:
    nvcc --version | findstr release
) else (
    echo CUDA not found in PATH.
    
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo NVIDIA driver is installed but CUDA is not in PATH.
        echo GPU status:
        nvidia-smi | findstr "Driver Version"
    ) else (
        echo Neither CUDA nor NVIDIA drivers appear to be properly installed.
    )
)

echo.
echo System Requirements Analysis:
echo ----------------------------
powershell -Command "& {$ram = Get-CimInstance -ClassName Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum; $ramGB = [math]::Round($ram.Sum / 1GB, 2); if ($ramGB -ge 32) { Write-Host 'RAM: Good (' $ramGB 'GB)' } else if ($ramGB -ge 16) { Write-Host 'RAM: Sufficient (' $ramGB 'GB), but 32GB+ recommended' } else { Write-Host 'RAM: Insufficient (' $ramGB 'GB) - 16GB minimum, 32GB+ recommended' }}"

powershell -Command "& {$gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*'}; if ($gpu -and $gpu.Name -like '*RTX*') { Write-Host 'GPU: Good (' $gpu.Name ')' } else if ($gpu) { Write-Host 'GPU: Detected (' $gpu.Name '), but RTX series recommended' } else { Write-Host 'GPU: Not detected - NVIDIA GPU required' }}"

powershell -Command "& {$disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter 'DeviceID=\"C:\"'; $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2); if ($freeSpaceGB -ge 100) { Write-Host 'Disk Space: Good (' $freeSpaceGB 'GB free)' } else if ($freeSpaceGB -ge 50) { Write-Host 'Disk Space: Sufficient (' $freeSpaceGB 'GB free), but 100GB+ recommended' } else { Write-Host 'Disk Space: Low (' $freeSpaceGB 'GB free) - 50GB minimum, 100GB+ recommended' }}"

python -c "import sys; ver = sys.version_info; print(f'Python: {'Good (Python ' + str(ver.major) + '.' + str(ver.minor) + ')' if (ver.major == 3 and ver.minor >= 8) else 'Insufficient - Python 3.8+ required'}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python: Not installed - Python 3.8+ required
)

nvcc --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA: Good
) else (
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo CUDA: Not found in PATH but NVIDIA drivers installed
    ) else (
        echo CUDA: Not installed - CUDA 12.1+ required
    )
)

echo.
