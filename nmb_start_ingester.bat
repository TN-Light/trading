@echo off
echo Starting NMB Live Options Ingester...
cd /d "C:\Users\amanu\Desktop\Trading"
call .venv\Scripts\activate.bat
python nmb_live_options_ingester.py
pause
