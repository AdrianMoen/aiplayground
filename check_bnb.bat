@echo off
echo Checking bitsandbytes and CUDA compatibility
echo ==========================================
echo.

python -c "import bitsandbytes as bnb; print(f'bitsandbytes version: {bnb.__version__}'); bnb.cuda_setup.print_cuda_setup(); print('\\nCUDA compatibility check:'); print(f'CUDA available modules: {bnb.cuda_setup.get_available_modules()}')"

echo.
pause
