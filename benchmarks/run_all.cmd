@echo off
setlocal
pushd "%~dp0"

python run_context_bench.py --engine both --fresh
if errorlevel 1 goto :fail

python analyze_context.py
if errorlevel 1 goto :fail

python make_context_chart.py
if errorlevel 1 goto :fail

popd
exit /b 0

:fail
echo.
echo Benchmark pipeline failed.
popd
exit /b 1
