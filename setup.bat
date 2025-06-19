@echo off
echo Phi-3 Mini Setup Toolkit
echo =======================
echo.

REM Check if running as administrator
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo This script should be run as administrator for proper installation.
    echo Right-click and select "Run as administrator" for best results.
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

:menu
cls
echo Phi-3 Mini Setup Toolkit
echo =======================
echo.
echo Environment Setup:
echo  1. Check system requirements
echo  2. Install/Update NVIDIA drivers
echo  3. Install CUDA Toolkit
echo  4. Setup PyTorch with CUDA
echo  5. Fix bitsandbytes CUDA integration
echo.
echo Verification:
echo  6. Verify CUDA installation
echo  7. Test PyTorch + CUDA
echo  8. Test Phi-3 Mini model loading
echo  9. Run full environment test
echo.
echo Project:
echo 10. Run fine-tuning pipeline
echo 11. Start REPL interface
echo.
echo 0. Exit
echo.
set /p choice=Enter your choice (0-11): 

if "%choice%"=="0" goto end
if "%choice%"=="1" goto check_system
if "%choice%"=="2" goto nvidia_drivers
if "%choice%"=="3" goto cuda_install
if "%choice%"=="4" goto pytorch_setup
if "%choice%"=="5" goto fix_bnb
if "%choice%"=="6" goto verify_cuda
if "%choice%"=="7" goto test_pytorch
if "%choice%"=="8" goto test_phi3
if "%choice%"=="9" goto full_test
if "%choice%"=="10" goto run_pipeline
if "%choice%"=="11" goto start_repl

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:check_system
echo.
echo Checking system requirements...
call setup\utils\check_system.bat
pause
goto menu

:nvidia_drivers
echo.
echo Checking NVIDIA drivers...
call setup\cuda\update_drivers.bat
pause
goto menu

:cuda_install
echo.
echo Installing CUDA Toolkit...
call setup\cuda\install_cuda.bat
pause
goto menu

:pytorch_setup
echo.
echo Setting up PyTorch with CUDA...
call setup\cuda\install_pytorch.bat
pause
goto menu

:fix_bnb
echo.
echo Fixing bitsandbytes CUDA integration...
call setup\cuda\fix_cuda_bnb.bat
pause
goto menu

:verify_cuda
echo.
echo Verifying CUDA installation...
call setup\tests\verify_cuda.bat
pause
goto menu

:test_pytorch
echo.
echo Testing PyTorch with CUDA...
call setup\tests\verify_pytorch.bat
pause
goto menu

:test_phi3
echo.
echo Testing Phi-3 Mini model loading...
call setup\tests\test_phi3_load.bat
pause
goto menu

:full_test
echo.
echo Running full environment test...
call setup\tests\full_environment_test.bat
pause
goto menu

:run_pipeline
echo.
echo Running fine-tuning pipeline...
call run_example_pipeline.bat
pause
goto menu

:start_repl
echo.
echo Starting REPL interface...
python phi3_repl.py --web_ui
pause
goto menu

:end
echo.
echo Exiting...
exit /b 0
