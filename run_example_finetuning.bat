@echo off
echo Running Phi-3 Mini fine-tuning pipeline with example dataset...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please run setup_environment.bat first.
    echo.
    pause
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "model_output" mkdir model_output

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

REM Run fine-tuning on example dataset
echo.
echo Fine-tuning Phi-3 Mini on example dataset...
python finetune_phi3_qlora.py --dataset example_dataset.json --output_dir model_output/phi3-mini-finetuned --epochs 3 --batch_size 8

REM Run REPL with fine-tuned model
echo.
echo Starting REPL interface with fine-tuned model...
python phi3_repl.py --base_model microsoft/Phi-3-mini-4k-instruct --adapter_path model_output/phi3-mini-finetuned --web_ui

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
pause
