@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -v -o "%~dp0gemvbench.exe" "%~dp0gemvbench.cu"
