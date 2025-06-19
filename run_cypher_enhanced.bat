@echo off
echo Starting Phi-3 Cypher Converter (Command Line)
echo ==========================================
echo.

cd %~dp0

REM Check if model exists
if not exist model_output\phi3-mini-cypher (
    echo The fine-tuned model doesn't exist yet.
    echo Please run prepare_movie_cypher.bat first to prepare the dataset and fine-tune the model.
    pause
    exit /b 1
)

echo Using adapter model...
set MODEL_PARAM=--adapter_path model_output\phi3-mini-cypher

echo Starting Cypher REPL interface in command line mode...
python cypher_repl.py %MODEL_PARAM%

echo.
pause
