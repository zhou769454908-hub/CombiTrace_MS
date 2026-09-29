@echo off
setlocal
where Rscript.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Rscript.exe was not found in PATH.
  pause
  exit /b 1
)
Rscript.exe --vanilla -e "cat('R=',R.version.string,'\n'); cat('MS2Quant available=',requireNamespace('MS2Quant',quietly=TRUE),'\n'); if(requireNamespace('MS2Quant',quietly=TRUE)){cat('MS2Quant version=',as.character(packageVersion('MS2Quant')),'\n'); cat('MS2Quant_predict_IE=',exists('MS2Quant_predict_IE',envir=asNamespace('MS2Quant'),inherits=FALSE),'\n')}"
pause
