@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )
cd /d "%~dp0.."
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v -o tmp\mma_gemm.exe tools\mma_gemm.cu > tmp\mma_gemm_build.log 2>&1
if errorlevel 1 ( type tmp\mma_gemm_build.log & exit /b 1 )
tmp\mma_gemm.exe %*
