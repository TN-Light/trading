@echo off
:: ============================================================================
:: PROMETHEUS Service — Stop
:: Kills any running Prometheus python process
:: ============================================================================

echo Stopping PROMETHEUS service...

:: Method 1: Kill python processes running prometheus
for /f "tokens=2 delims=," %%a in ('wmic process where "name='python.exe'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | find "prometheus" >nul && (
        echo Killing PID %%a
        taskkill /pid %%a /f >nul 2>&1
    )
)

:: Method 2: Fallback — kill any python process with prometheus in command line
wmic process where "name='python.exe' and CommandLine like '%%prometheus%%'" call terminate >nul 2>&1

echo PROMETHEUS service stopped.
echo [%date% %time%] Service stopped manually. >> "C:\Users\amanu\Desktop\Trading\prometheus_service.log"
pause
