@echo off
cd /d "%~dp0"
python run_continuous_lab.py
if errorlevel 1 pause
