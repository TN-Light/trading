@echo off
:: ============================================================================
:: PROMETHEUS Trading Service — Auto-Start Launcher
:: Runs COMBINED mode: Swing (1PM + 3:35PM) + Intraday (9:45-14:30) + Multi-Account
:: ============================================================================

cd /d "C:\Users\amanu\Desktop\Trading"

:: Log file for debugging
set LOGFILE=C:\Users\amanu\Desktop\Trading\prometheus_service.log

:: Rotate log if > 10 MB (10485760 bytes)
if exist "%LOGFILE%" (
    for %%A in ("%LOGFILE%") do (
        if %%~zA GTR 10485760 (
            if exist "%LOGFILE%.2" del "%LOGFILE%.2"
            if exist "%LOGFILE%.1" ren "%LOGFILE%.1" prometheus_service.log.2
            ren "%LOGFILE%" prometheus_service.log.1
        )
    )
)

echo [%date% %time%] PROMETHEUS service starting... >> "%LOGFILE%"
echo [%date% %time%] Mode: COMBINED (swing + intraday) + MULTI-ACCOUNT >> "%LOGFILE%"

:: Run combined mode: swing + intraday with 4 paper accounts
:: --combined  = both swing scans (1PM, 3:35PM) + intraday scans (9:45-14:30)
:: --multi-account = 4 parallel accounts (15K, 50K, 1L, 2L)
:: Intraday scan auto-aligns to bar interval (5min bars→300s, 15min bars→900s)
"C:\Program Files\Python312\python.exe" prometheus/main.py paper --combined --multi-account >> "%LOGFILE%" 2>&1

echo [%date% %time%] PROMETHEUS service stopped. >> "%LOGFILE%"
