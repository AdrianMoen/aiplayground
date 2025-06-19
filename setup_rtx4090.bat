@echo off
echo RTX 4090 Setup for Phi-3 Mini Fine-tuning
echo =========================================
echo.

:menu
echo Choose an option:
echo 1. Check NVIDIA GPU and update drivers
echo 2. Install CUDA Toolkit
echo 3. Install PyTorch and dependencies
echo 4. Verify CUDA and bitsandbytes setup
echo 5. Run fine-tuning pipeline on example dataset
echo 6. Exit
echo.

set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" goto check_gpu
if "%choice%"=="2" goto install_cuda
if "%choice%"=="3" goto install_pytorch
if "%choice%"=="4" goto verify_setup
if "%choice%"=="5" goto run_pipeline
if "%choice%"=="6" goto end

echo Invalid choice. Please try again.
goto menu

:check_gpu
echo.
echo Checking NVIDIA GPU and drivers...
call update_nvidia_drivers.bat
echo.
goto menu

:install_cuda
echo.
echo Installing CUDA Toolkit...
call install_cuda.bat
echo.
goto menu

:install_pytorch
echo.
echo Installing PyTorch and dependencies...
call install_pytorch.bat
echo.
goto menu

:verify_setup
echo.
echo Verifying CUDA and bitsandbytes setup...
call check_bnb.bat
echo.
goto menu

:run_pipeline
echo.
echo Running fine-tuning pipeline on example dataset...
call run_example_pipeline.bat
echo.
goto menu

:end
echo.
echo Exiting...
exit /b 0
