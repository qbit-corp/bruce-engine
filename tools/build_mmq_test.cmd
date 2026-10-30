@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )
cd /d "%~dp0.."
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v -Xcompiler "/O2 /Zc:preprocessor" -o tmp\mmq_test.exe tools\mmq_test.cu > tmp\mmq_test_build.log 2>&1
if errorlevel 1 ( type tmp\mmq_test_build.log & exit /b 1 )
tmp\mmq_test.exe %*
