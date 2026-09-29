@echo off
setlocal
cd /d "%~dp0"
where Rscript.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Rscript.exe was not found in PATH.
  echo Install 64-bit R, then run this command from the R bin folder or select Rscript.exe in the software.
  pause
  exit /b 1
)
echo Installing/updating official KruveLab MS2Quant package...
Rscript.exe --vanilla "%~dp0r_scripts\install_ms2quant.R"
if errorlevel 1 (
  echo.
  echo [FAILED] Check the messages above. Windows commonly needs a compatible 64-bit Java/JDK for rJava.
  pause
  exit /b 1
)
echo.
echo [OK] Installation command completed. Use the software button 'Detect R/MS2Quant' to verify.
pause
