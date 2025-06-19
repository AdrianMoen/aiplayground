@echo off
echo Testing PyTorch CUDA Compatibility
echo ===============================
echo.

cd %~dp0\..\..

echo Checking PyTorch installation and CUDA support...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 

if %ERRORLEVEL% neq 0 (
    echo Failed to import PyTorch. Please make sure PyTorch is installed.
    exit /b 1
)

echo.
echo Running PyTorch CUDA test...
python -c "import torch; print('Testing CUDA tensor creation...'); x = torch.rand(5, 3).cuda() if torch.cuda.is_available() else torch.rand(5, 3); print(x); print('Testing tensor operations...'); y = x + x; print(y); print('Device:', x.device); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

if %ERRORLEVEL% neq 0 (
    echo Failed to perform PyTorch CUDA operations.
    echo This may indicate an issue with your PyTorch CUDA configuration.
    exit /b 1
)

echo.
echo Testing PyTorch model allocation on GPU...
python -c "import torch, torch.nn as nn; print('Creating a simple neural network...'); model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1)); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Moving model to {device}...'); model = model.to(device); print('Model device:', next(model.parameters()).device); print('Running a forward pass...'); output = model(torch.rand(5, 10).to(device)); print('Output shape:', output.shape); print('Output device:', output.device)"

if %ERRORLEVEL% neq 0 (
    echo Failed to allocate PyTorch model on GPU.
    echo This may indicate an issue with your PyTorch CUDA configuration.
    exit /b 1
)

echo.
echo PyTorch CUDA Verification Complete!
echo.
