@echo off
echo Starting Basic Phi-3 Cypher Converter (Command Line)
echo =============================================
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
echo Starting Basic Cypher REPL interface...
python basic_cypher_repl.py --adapter_path model_output\phi3-mini-cypher

echo.
pause
