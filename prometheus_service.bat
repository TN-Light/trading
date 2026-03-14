@echo off
:: ============================================================================
:: PROMETHEUS Signal Service — Launcher
:: Runs signal mode with continuous scanning during market hours
:: ============================================================================

cd /d "C:\Users\amanu\Desktop\Trading"

:: Log file for debugging
set LOGFILE=C:\Users\amanu\Desktop\Trading\prometheus_service.log

echo [%date% %time%] PROMETHEUS service starting... >> "%LOGFILE%"

:: Run signal mode (scans every 5 minutes during market hours)
"C:\Program Files\Python312\python.exe" prometheus/main.py signal --interval 300 >> "%LOGFILE%" 2>&1

echo [%date% %time%] PROMETHEUS service stopped. >> "%LOGFILE%"
