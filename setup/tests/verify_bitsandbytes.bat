@echo off
echo Testing bitsandbytes CUDA Compatibility
echo ===================================
echo.

cd %~dp0\..\..

echo Checking bitsandbytes installation and CUDA support...
python -c "import bitsandbytes as bnb; print('bitsandbytes version:', bnb.__version__)"

if %ERRORLEVEL% neq 0 (
    echo Failed to import bitsandbytes. Please make sure bitsandbytes is installed.
    exit /b 1
)

echo.
echo Checking if bitsandbytes has CUDA support...
python -c "import bitsandbytes as bnb; from bitsandbytes.cuda_setup.main import get_compute_capability; cc = get_compute_capability(); print(f'Compute capability: {cc}'); print('CUDA available in bitsandbytes:', cc is not None and cc > 0)"

if %ERRORLEVEL% neq 0 (
    echo Failed to check bitsandbytes CUDA support.
    echo This may indicate an issue with your bitsandbytes installation.
    exit /b 1
)

echo.
echo Testing 8-bit linear layer creation...
python -c "import torch; import bitsandbytes as bnb; print('Creating 8-bit linear layer...'); linear_8bit = bnb.nn.Linear8bitLt(10, 10, bias=True); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Moving layer to {device}...'); linear_8bit = linear_8bit.to(device); print('Running a forward pass...'); output = linear_8bit(torch.rand(5, 10).to(device)); print('Output shape:', output.shape); print('Output device:', output.device)"

if %ERRORLEVEL% neq 0 (
    echo Failed to create and test bitsandbytes 8-bit linear layer.
    echo This may indicate an issue with your bitsandbytes CUDA configuration.
    exit /b 1
)

echo.
echo Testing 4-bit quantization functionality...
python -c "import torch; import bitsandbytes as bnb; from bitsandbytes.nn import LinearFP4; print('Checking if 4-bit quantization is available...'); try: linear_4bit = LinearFP4(10, 10); print('4-bit quantization is available!'); except Exception as e: print(f'4-bit quantization error: {str(e)}'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); try: linear_4bit = linear_4bit.to(device); output = linear_4bit(torch.rand(5, 10).to(device)); print('Successfully created and used 4-bit layer'); except Exception as e: print(f'Could not test 4-bit layer on device: {str(e)}')"

echo.
echo bitsandbytes Verification Complete!
echo.
