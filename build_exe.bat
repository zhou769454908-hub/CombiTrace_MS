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
"%BUILD_PY%" build_exe.py  %*
set "BUILD_RC=%ERRORLEVEL%"
echo.
echo Exit code: %BUILD_RC%
echo See LAST_BUILD_ATTEMPT.txt for this run's log folder.
pause
exit /b %BUILD_RC%
