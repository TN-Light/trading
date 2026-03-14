@echo off
:: ============================================================================
:: PROMETHEUS Signal Service — Stop
:: Kills any running Prometheus signal mode process
:: ============================================================================

echo Stopping PROMETHEUS service...

:: Kill python processes running prometheus main.py
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | find "prometheus" >nul && (
        echo Killing PID %%a
        taskkill /pid %%a /f >nul 2>&1
    )
)

echo PROMETHEUS service stopped.
echo [%date% %time%] Service stopped manually. >> "C:\Users\amanu\Desktop\Trading\prometheus_service.log"
pause
