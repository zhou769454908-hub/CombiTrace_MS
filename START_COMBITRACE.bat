@echo off
cd /d "%~dp0"
if defined THERMO_BUILD_PYTHON (
  "%THERMO_BUILD_PYTHON%" app.py
) else (
  python app.py
)
if errorlevel 1 pause
