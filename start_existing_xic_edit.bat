@echo off
cd /d "%~dp0"
if defined THERMO_BUILD_PYTHON (
  "%THERMO_BUILD_PYTHON%" run_existing_xic_edit.py
) else (
  python run_existing_xic_edit.py
)
if errorlevel 1 pause
