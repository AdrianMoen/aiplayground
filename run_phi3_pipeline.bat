@echo off
echo Phi-3 Mini Fine-tuning Pipeline
echo ============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.8 or higher and try again.
    goto end
)

:menu
echo Choose an option:
echo 1. Setup environment
echo 2. Prepare custom dataset
echo 3. Fine-tune model
echo 4. Save merged model
echo 5. Run REPL interface
echo 6. Run full pipeline
echo 7. Exit
echo.

set /p choice=Enter your choice (1-7): 

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto prepare
if "%choice%"=="3" goto finetune
if "%choice%"=="4" goto save_model
if "%choice%"=="5" goto repl
if "%choice%"=="6" goto full
if "%choice%"=="7" goto end

echo Invalid choice. Please try again.
goto menu

:setup
echo Setting up environment...
python run_pipeline.py --setup_env
echo.
goto menu

:prepare
echo Preparing custom dataset...
set /p input_file=Enter path to input data file: 
set /p output_file=Enter path to save processed dataset (or press Enter for default): 
set /p format=Enter format (alpaca or chatml) (or press Enter for default): 

set cmd=python run_pipeline.py --prepare_data --input_file "%input_file%"
if not "%output_file%"=="" set cmd=%cmd% --output_file "%output_file%"
if not "%format%"=="" set cmd=%cmd% --data_format %format%

%cmd%
echo.
goto menu

:finetune
echo Fine-tuning the model...
set /p dataset=Enter dataset path or name (or press Enter to use the prepared dataset): 
set /p output_dir=Enter output directory for fine-tuned model (or press Enter for default): 
set /p epochs=Enter number of epochs (or press Enter for default): 
set /p batch_size=Enter batch size (or press Enter for default): 
set /p use_wandb=Use Weights & Biases for logging? (y/n) (default: n): 

set cmd=python run_pipeline.py --run_finetuning
if not "%dataset%"=="" set cmd=%cmd% --dataset "%dataset%"
if not "%output_dir%"=="" set cmd=%cmd% --output_dir "%output_dir%"
if not "%epochs%"=="" set cmd=%cmd% --epochs %epochs%
if not "%batch_size%"=="" set cmd=%cmd% --batch_size %batch_size%
if /i "%use_wandb%"=="y" set cmd=%cmd% --use_wandb

%cmd%
echo.
goto menu

:save_model
echo Saving merged model...
set /p base_model=Enter base model name (or press Enter for default): 
set /p adapter_path=Enter adapter path (or press Enter for default): 
set /p merged_model_dir=Enter merged model directory (or press Enter for default): 

set cmd=python run_pipeline.py --save_merged_model
if not "%base_model%"=="" set cmd=%cmd% --base_model "%base_model%"
if not "%adapter_path%"=="" set cmd=%cmd% --output_dir "%adapter_path%"
if not "%merged_model_dir%"=="" set cmd=%cmd% --merged_model_dir "%merged_model_dir%"

%cmd%
echo.
goto menu

:repl
echo Running REPL interface...
set /p use_4bit=Use 4-bit quantization? (y/n) (default: n): 
set /p web_ui=Use web UI? (y/n) (default: n): 
set /p use_merged=Use merged model instead of adapter? (y/n) (default: n): 

set cmd=python run_pipeline.py --run_repl
if /i "%use_4bit%"=="y" set cmd=%cmd% --use_4bit
if /i "%web_ui%"=="y" set cmd=%cmd% --web_ui
if /i "%use_merged%"=="y" set cmd=%cmd% --use_merged_model

%cmd%
echo.
goto menu

:full
echo Running full pipeline...
set /p input_file=Enter path to input data file: 
set /p epochs=Enter number of epochs (or press Enter for default): 
set /p web_ui=Use web UI for REPL? (y/n) (default: n): 
set /p save_merged=Save merged model? (y/n) (default: n): 

set cmd=python run_pipeline.py --setup_env --prepare_data --input_file "%input_file%" --run_finetuning --run_repl
if not "%epochs%"=="" set cmd=%cmd% --epochs %epochs%
if /i "%web_ui%"=="y" set cmd=%cmd% --web_ui
if /i "%save_merged%"=="y" set cmd=%cmd% --save_merged_model

%cmd%
echo.
goto menu

:end
echo Exiting...
exit /b 0
