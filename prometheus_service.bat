@echo off
:: ============================================================================
:: PROMETHEUS Paper Trading Service — Launcher
:: Runs paper mode: scans at 3:35 PM, auto-executes paper trades
:: ============================================================================

cd /d "C:\Users\amanu\Desktop\Trading"

:: Log file for debugging
set LOGFILE=C:\Users\amanu\Desktop\Trading\prometheus_service.log

echo [%date% %time%] PROMETHEUS service starting... >> "%LOGFILE%"

:: Run paper mode (scans at 3:35 PM after daily candle, auto-executes paper trades)
"C:\Program Files\Python312\python.exe" prometheus/main.py paper --interval 300 >> "%LOGFILE%" 2>&1

echo [%date% %time%] PROMETHEUS service stopped. >> "%LOGFILE%"
