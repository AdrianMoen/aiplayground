@echo off
echo Phi-3 Mini Fine-tuning Pipeline with Example Dataset
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please run setup_environment.bat first.
    goto install_python
)

REM Check if CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo CUDA is not available. Please run install_cuda.bat and install_pytorch.bat first.
    echo.
    echo Would you like to install CUDA now? (Y/N)
    set /p install_choice=
    
    if /i "%install_choice%"=="Y" (
        call install_cuda.bat
        echo After CUDA installation and reboot, please run install_pytorch.bat and then this script again.
        pause
        exit /b 1
    ) else (
        echo.
        echo Please install CUDA and PyTorch manually and run this script again.
        pause
        exit /b 1
    )
)

REM If Python and CUDA are installed, run the pipeline
echo Python and CUDA are available. Running the pipeline...
goto run_pipeline

:install_python
echo.
echo Would you like to install Python now? (Y/N)
set /p install_choice=

if /i "%install_choice%"=="Y" (
    echo.
    echo Running Python installer...
    call setup_environment.bat
    goto check_python
) else (
    echo.
    echo Please install Python and run this script again.
    pause
    exit /b 1
)

:check_python
echo.
echo Checking if Python was installed successfully...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python installation failed. Please install Python manually.
    pause
    exit /b 1
)

:run_pipeline
echo.
echo Creating necessary directories...
if not exist "model_output" mkdir model_output

echo.
echo Fine-tuning Phi-3 Mini on example dataset...
python finetune_phi3_qlora.py --dataset example_dataset.json --output_dir model_output/phi3-mini-finetuned --save_merged_model --merged_model_dir model_output/phi3-mini-merged

echo.
echo Starting REPL interface with fine-tuned model...
python phi3_repl.py --model_path model_output/phi3-mini-finetuned --web_ui

echo.
echo Pipeline completed successfully!
pause
