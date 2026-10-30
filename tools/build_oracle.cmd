@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo vcvars64 failed
  exit /b 1
)
cd /d "%~dp0"
set REF=%~dp0doryref
if "%LLAMA_BIN%"=="" (
  set BIN=%~dp0doryref\build-cuda
) else (
  set BIN=%LLAMA_BIN%
)
cl /nologo /O2 /std:c++17 /EHsc /utf-8 /I "%REF%\include" /I "%REF%\ggml\include" oracle_dump.cpp ^
  /Fe:oracle_dump.exe ^
  /link /LIBPATH:"%BIN%\src" /LIBPATH:"%BIN%\ggml\src" /LIBPATH:"%BIN%\ggml\src\ggml-cuda" ^
  llama.lib ggml.lib ggml-base.lib ggml-cpu.lib ggml-cuda.lib
if errorlevel 1 (
  echo build failed
  exit /b 1
)
echo build ok: %~dp0oracle_dump.exe
