@echo off
echo Converting Movie-Cypher Dataset for Fine-tuning
echo =============================================
echo.

cd %~dp0

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please run setup.bat first.
    pause
    exit /b 1
)

echo Converting CSV dataset to fine-tuning format...
python utils\convert_movie_cypher.py --input_file data\movie_nl_cypher_1000.csv --output_file data\movie_nl_cypher_dataset.json --format alpaca --split 0.1

if %ERRORLEVEL% neq 0 (
    echo Failed to convert dataset. Please check the error message above.
    pause
    exit /b 1
)

echo.
echo Would you like to run fine-tuning with this dataset now? (Y/N)
set /p run_choice=

if /i "%run_choice%"=="Y" (
    echo.
    echo Starting fine-tuning process...
    python finetune_phi3_qlora.py --dataset data\movie_nl_cypher_dataset.json --output_dir model_output\phi3-mini-cypher --epochs 3 --batch_size 8
    
    if %ERRORLEVEL% neq 0 (
        echo Fine-tuning process encountered an error. Please check the output above.
        pause
        exit /b 1
    )
    
    echo.
    echo Fine-tuning completed successfully!
    echo.
    echo Would you like to start the REPL interface with the fine-tuned model? (Y/N)
    set /p repl_choice=
    
    if /i "%repl_choice%"=="Y" (
        echo.
        echo Starting REPL interface...
        python phi3_repl.py --model_path model_output\phi3-mini-cypher --web_ui
    ) else (
        echo.
        echo You can start the REPL interface later by running:
        echo python phi3_repl.py --model_path model_output\phi3-mini-cypher --web_ui
    )
) else (
    echo.
    echo You can run fine-tuning later with:
    echo python finetune_phi3_qlora.py --dataset data\movie_nl_cypher_dataset.json --output_dir model_output\phi3-mini-cypher
)

echo.
pause
