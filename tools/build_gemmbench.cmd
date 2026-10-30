@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo vcvars64 failed
  exit /b 1
)
cd /d "%~dp0"
nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler "/O2 /W3 /Zc:preprocessor" -o gemmbench.exe gemmbench.cu
if errorlevel 1 (
  echo build failed
  exit /b 1
)
echo build ok: %~dp0gemmbench.exe
