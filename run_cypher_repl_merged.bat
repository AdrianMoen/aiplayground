@echo off
echo Starting Phi-3 Cypher Converter REPL (Merged Model)
echo ================================================
echo.

cd %~dp0

REM Check if merged model exists
if not exist model_output\phi3-mini-cypher-merged (
    echo The merged fine-tuned model doesn't exist yet.
    echo Please run the fine-tuning with --save_merged_model option.
    pause
    exit /b 1
)

echo Starting REPL interface with fine-tuned Cypher merged model...
python phi3_repl.py --saved_model_dir model_output\phi3-mini-cypher-merged --web_ui

echo.
pause
