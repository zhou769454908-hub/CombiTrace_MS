@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if defined THERMO_BUILD_PYTHON (
  set "BUILD_PY=%THERMO_BUILD_PYTHON%"
) else (
  set "BUILD_PY=python"
)
"%BUILD_PY%" -c "import sys; print('Selected Python:',sys.executable)"
if errorlevel 1 (
  echo Python could not start. Open your working Anaconda Prompt, or set THERMO_BUILD_PYTHON.
  pause
  exit /b 1
)
echo This installs BUILD TOOLS ONLY into the interpreter shown above.
echo It does NOT reinstall requirements.txt or change model parameters.
echo Close this window to cancel.
pause
"%BUILD_PY%" -m pip install "pyinstaller>=6.0,<7" pyinstaller-hooks-contrib
set "BUILD_RC=%ERRORLEVEL%"
if not "%BUILD_RC%"=="0" echo Installation failed. Keep the complete error text.
pause
exit /b %BUILD_RC%
