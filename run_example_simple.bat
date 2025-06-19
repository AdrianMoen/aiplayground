@echo off
echo Phi-3 Mini Fine-tuning Pipeline with Example Dataset
echo =====================================================
echo.

REM Try different Python command variations
SET PYTHON_CMD=none

echo Checking for Python installation...

where python >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
  echo Python found using 'python' command
  SET PYTHON_CMD=python
  goto found_python
)

where python3 >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
  echo Python found using 'python3' command
  SET PYTHON_CMD=python3
  goto found_python
)

where py >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
  echo Python found using 'py' command
  SET PYTHON_CMD=py
  goto found_python
)

echo Python is not found in PATH.
echo.
echo Please check the following:
echo 1. Python may be installed but not in your PATH environment variable
echo 2. The system may need to be restarted after Python installation
echo 3. You may need to install Python manually from https://www.python.org/downloads/
echo.
echo Installing requirements directly...
echo.

REM Try to use Python from common installation locations
IF EXIST "C:\Python310\python.exe" (
  echo Found Python at C:\Python310\python.exe
  "C:\Python310\python.exe" -m pip install -r requirements.txt
  SET PYTHON_CMD="C:\Python310\python.exe"
  goto found_python
)

IF EXIST "C:\Program Files\Python310\python.exe" (
  echo Found Python at C:\Program Files\Python310\python.exe
  "C:\Program Files\Python310\python.exe" -m pip install -r requirements.txt
  SET PYTHON_CMD="C:\Program Files\Python310\python.exe"
  goto found_python
)

IF EXIST "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
  echo Found Python at %LOCALAPPDATA%\Programs\Python\Python310\python.exe
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
  SET PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  goto found_python
)

echo No Python installation found. Please install Python and try again.
pause
exit /b 1

:found_python
echo.
echo Using Python: %PYTHON_CMD%
echo.

REM Install required packages
echo Installing required packages...
%PYTHON_CMD% -m pip install -r requirements.txt

REM Create output directory
echo Creating output directory...
if not exist "model_output" mkdir model_output

REM Run fine-tuning
echo.
echo Starting fine-tuning on example dataset...
%PYTHON_CMD% finetune_phi3_qlora.py --dataset example_dataset.json --output_dir model_output/phi3-mini-finetuned --epochs 1 --batch_size 4

REM Run REPL with fine-tuned model
echo.
echo Starting REPL interface with fine-tuned model...
%PYTHON_CMD% phi3_repl.py --base_model microsoft/Phi-3-mini-4k-instruct --adapter_path model_output/phi3-mini-finetuned --web_ui

echo.
echo Pipeline completed!
pause
