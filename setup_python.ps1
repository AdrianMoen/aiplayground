# setup_python.ps1
# Script to install Python and required dependencies for Phi-3 Mini fine-tuning

# Set execution policy for this script
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Define Python version to install
$pythonVersion = "3.10.11"
$pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
$pythonInstaller = "$env:TEMP\python-$pythonVersion-amd64.exe"

Write-Host "Downloading Python $pythonVersion..." -ForegroundColor Green
Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller

Write-Host "Installing Python $pythonVersion..." -ForegroundColor Green
# Install Python with pip, add to PATH, and disable path length limit
& $pythonInstaller /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_pip=1 Include_doc=0

# Wait for installation to complete
Start-Sleep -Seconds 30

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "Verifying Python installation..." -ForegroundColor Green
python --version
pip --version

# Install required dependencies
Write-Host "Installing required packages..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host "You can now run the Phi-3 Mini fine-tuning pipeline using run_phi3_pipeline.bat" -ForegroundColor Green
