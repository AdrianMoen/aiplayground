@echo off
echo Phi-3 Mini Fine-tuning Pipeline
echo ==============================
echo.

REM Check for Python installation
py -0 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.8 or higher.
    echo.
    echo Installation steps:
    echo 1. Download Python from https://www.python.org/downloads/
    echo 2. Run the installer and make sure to check "Add Python to PATH"
    echo 3. After installation, run this script again
    exit /b 1
)

REM If Python is installed, check for Python 3
py -3 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python 3 not found! Please install Python 3.8 or higher.
    echo.
    echo Available Python versions:
    py -0
    echo.
    echo Installation steps:
    echo 1. Download Python 3.8+ from https://www.python.org/downloads/
    echo 2. Run the installer and make sure to check "Add Python to PATH"
    echo 3. After installation, run this script again
    exit /b 1
)

echo Python 3 found. Continuing with pipeline...
echo.

echo Running full Phi-3 Mini fine-tuning pipeline with example dataset...
echo This will:
echo  1. Set up the environment
echo  2. Fine-tune Phi-3 Mini on the example dataset
echo  3. Save the fine-tuned model
echo  4. Launch the REPL interface
echo.

set input_file=example_dataset.json
set epochs=3
set batch_size=4
set output_dir=./phi3-mini-finetuned
set merged_dir=./phi3-mini-finetuned-merged

REM Use Python launcher
set python_cmd=py -3

echo Step 1: Setting up environment...
%python_cmd% run_pipeline.py --setup_env

echo Step 2: Fine-tuning on example dataset...
%python_cmd% run_pipeline.py --run_finetuning --dataset %input_file% --epochs %epochs% --batch_size %batch_size% --output_dir %output_dir%

echo Step 3: Saving merged model...
%python_cmd% run_pipeline.py --save_merged_model --output_dir %output_dir% --merged_model_dir %merged_dir%

echo Step 4: Launching REPL interface...
%python_cmd% run_pipeline.py --run_repl --use_merged_model --merged_model_dir %merged_dir% --web_ui

echo Done!
