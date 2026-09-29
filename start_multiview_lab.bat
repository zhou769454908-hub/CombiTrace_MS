@echo off
cd /d "%~dp0"
python run_multiview_lab.py
if errorlevel 1 pause
