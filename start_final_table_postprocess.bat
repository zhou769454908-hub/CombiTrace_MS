@echo off
cd /d "%~dp0"
if defined THERMO_BUILD_PYTHON (
  "%THERMO_BUILD_PYTHON%" run_final_table_postprocess.py
) else (
  python run_final_table_postprocess.py
)
if errorlevel 1 pause
