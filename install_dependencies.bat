@echo off
echo Setting up dependencies for Phi-3 Mini fine-tuning...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Dependencies installed successfully!
echo You can now run the example pipeline using run_example_pipeline.bat
echo.
pause
