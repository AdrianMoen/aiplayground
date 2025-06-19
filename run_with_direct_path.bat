@echo off
echo Phi-3 Mini Fine-tuning Pipeline with Example Dataset
echo =====================================================
echo.

SET PYTHON_PATH="C:\Program Files\Python310\python.exe"
SET PIP_PATH="C:\Program Files\Python310\Scripts\pip.exe"

echo Using Python at %PYTHON_PATH%
echo.

REM Install required packages
echo Installing required packages...
%PYTHON_PATH% -m pip install -r requirements.txt

REM Create output directory
echo Creating output directory...
if not exist "model_output" mkdir model_output

REM Run fine-tuning
echo.
echo Starting fine-tuning on example dataset...
%PYTHON_PATH% finetune_phi3_qlora.py --dataset example_dataset.json --output_dir model_output/phi3-mini-finetuned --epochs 1 --batch_size 4

REM Run REPL with fine-tuned model
echo.
echo Starting REPL interface with fine-tuned model...
%PYTHON_PATH% phi3_repl.py --base_model microsoft/Phi-3-mini-4k-instruct --adapter_path model_output/phi3-mini-finetuned --web_ui

echo.
echo Pipeline completed!
pause
