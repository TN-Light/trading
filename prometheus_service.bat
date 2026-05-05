@echo off
:: ============================================================================
:: PROMETHEUS Trading Service — Auto-Start Launcher
:: Runs SWING paper mode: 15-minute scans from the first valid 09:30 close
:: ============================================================================

cd /d "C:\Users\amanu\Desktop\Trading"
set "ROOT=%CD%"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=C:\Program Files\Python312\python.exe"

set "LOGDIR=%ROOT%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TS=%%A"
set "LOGFILE=%LOGDIR%\prometheus_service_%TS%.log"

echo [%date% %time%] PROMETHEUS service starting... >> "%LOGFILE%"
echo [%date% %time%] Mode: SWING PAPER (15-minute scans, first valid scan ~09:30 IST) >> "%LOGFILE%"

:: Run swing-only paper mode
:: --data-source auto = Kite if available, else Angel One/yfinance fallbacks
"%PYTHON%" prometheus/main.py paper --data-source auto --fetch-retries 2 >> "%LOGFILE%" 2>&1

echo [%date% %time%] PROMETHEUS service stopped. >> "%LOGFILE%"
