@echo off
title AquaForge — DEV MODE (auto-reload)
cd /d "%~dp0"
echo Starting AquaForge in DEVELOPMENT mode...
echo Changes to .py files will auto-reload.
echo Press Ctrl+C to stop.
echo.
py -3 -m streamlit run app.py --server.runOnSave true --server.headless true
if errorlevel 1 pause
