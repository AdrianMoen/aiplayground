@echo off
echo Verifying CUDA Installation
echo =========================
echo.

cd %~dp0\..\..

echo Checking NVIDIA driver...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo NVIDIA driver is installed and functioning:
    nvidia-smi | findstr "Driver Version"
    echo.
    echo GPU information:
    nvidia-smi | findstr "GeForce"
) else (
    echo NVIDIA driver is NOT properly installed or not functioning.
    echo Please install the appropriate NVIDIA drivers for your GPU.
    exit /b 1
)

echo.
echo Checking CUDA installation...
nvcc --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA is installed:
    nvcc --version | findstr release
) else (
    echo CUDA is NOT found in your PATH.
    echo.
    echo Checking if CUDA is installed but not in PATH...
    
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
        echo CUDA appears to be installed but is not in your PATH.
        echo Please add CUDA bin directory to your PATH environment variable.
        echo.
        echo Typically this would be something like:
        echo C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
    ) else (
        echo CUDA does not appear to be installed.
        echo Please run the CUDA installer.
    )
    exit /b 1
)

echo.
echo Testing CUDA functionality...
echo #include ^<stdio.h^> > cuda_test.cu
echo #include ^<cuda_runtime.h^> >> cuda_test.cu
echo int main^(^) { >> cuda_test.cu
echo   int deviceCount = 0; >> cuda_test.cu
echo   cudaError_t error = cudaGetDeviceCount^(^&deviceCount^); >> cuda_test.cu
echo   if ^(error != cudaSuccess^) { >> cuda_test.cu
echo     printf^("CUDA Error: %%s\n", cudaGetErrorString^(error^)^); >> cuda_test.cu
echo     return -1; >> cuda_test.cu
echo   } >> cuda_test.cu
echo   printf^("CUDA device count: %%d\n", deviceCount^); >> cuda_test.cu
echo   for ^(int i = 0; i ^< deviceCount; ++i^) { >> cuda_test.cu
echo     cudaDeviceProp prop; >> cuda_test.cu
echo     cudaGetDeviceProperties^(^&prop, i^); >> cuda_test.cu
echo     printf^("Device %%d: %%s\n", i, prop.name^); >> cuda_test.cu
echo   } >> cuda_test.cu
echo   return 0; >> cuda_test.cu
echo } >> cuda_test.cu

nvcc cuda_test.cu -o cuda_test.exe >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA compiler functioning correctly.
    echo Running CUDA test program...
    echo.
    cuda_test.exe
    del cuda_test.cu cuda_test.exe >nul 2>&1
) else (
    echo Failed to compile CUDA test program.
    echo This may indicate an issue with your CUDA installation.
    del cuda_test.cu >nul 2>&1
    exit /b 1
)

echo.
echo CUDA Verification Complete!
echo.
