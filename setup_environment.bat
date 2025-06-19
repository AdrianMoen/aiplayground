@echo off
echo Setting up Python and dependencies for Phi-3 Mini fine-tuning...
echo.

REM Run the PowerShell setup script with elevated privileges
powershell -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File setup_python.ps1' -Verb RunAs"

echo.
echo When setup is complete, you can run the fine-tuning pipeline using run_phi3_pipeline.bat
echo.
