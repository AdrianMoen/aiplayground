@echo off
echo Full Environment Test for Phi-3 Fine-tuning
echo =======================================
echo.

cd %~dp0\..\..

echo ===== STEP 1: System Requirements Check =====
call setup\utils\check_system.bat
if %ERRORLEVEL% neq 0 (
    echo System requirements check failed. Please review the output above.
    pause
    exit /b 1
)

echo.
echo ===== STEP 2: CUDA Verification =====
call setup\tests\verify_cuda.bat
if %ERRORLEVEL% neq 0 (
    echo CUDA verification failed. Please review the output above.
    pause
    exit /b 1
)

echo.
echo ===== STEP 3: PyTorch CUDA Verification =====
call setup\tests\verify_pytorch.bat
if %ERRORLEVEL% neq 0 (
    echo PyTorch CUDA verification failed. Please review the output above.
    pause
    exit /b 1
)

echo.
echo ===== STEP 4: bitsandbytes Verification =====
call setup\tests\verify_bitsandbytes.bat
if %ERRORLEVEL% neq 0 (
    echo bitsandbytes verification failed. Please review the output above.
    pause
    exit /b 1
)

echo.
echo ===== STEP 5: Phi-3 Model Load Test =====
call setup\tests\test_phi3_load.bat
if %ERRORLEVEL% neq 0 (
    echo Phi-3 model load test failed. Please review the output above.
    pause
    exit /b 1
)

echo.
echo =============================================
echo All environment tests PASSED successfully!
echo Your system is ready for Phi-3 fine-tuning with QLoRA.
echo.
echo Next steps:
echo 1. Run the example fine-tuning pipeline: run_example_pipeline.bat
echo 2. Or try the Phi-3 REPL interface: python phi3_repl.py
echo =============================================
echo.
pause
