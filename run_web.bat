@echo off
title AquaForge — web UI
cd /d "%~dp0"
echo Starting web UI... your browser should open to http://localhost:8501
echo Close this window or press Ctrl+C to stop the server.
echo.
py -3 -m streamlit run app.py
if errorlevel 1 pause
