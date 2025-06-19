@echo off
echo Starting Phi-3 Cypher Converter REPL
echo ==================================
echo.

cd %~dp0

REM Check if model exists
if not exist model_output\phi3-mini-cypher (
    echo The fine-tuned model doesn't exist yet.
    echo Please run prepare_movie_cypher.bat first to prepare the dataset and fine-tune the model.
    pause
    exit /b 1
)

echo Starting REPL interface with fine-tuned Cypher model...
python phi3_repl.py --model_path model_output\phi3-mini-cypher --web_ui

echo.
pause
