@echo off
:: Register Prometheus as a scheduled task (run once, then delete this file)
schtasks /create /tn "PrometheusSignalService" /tr "wscript.exe C:\Users\amanu\Desktop\Trading\prometheus_service.vbs" /sc onlogon /rl highest /f
if %errorlevel% equ 0 (
    echo SUCCESS: Prometheus will auto-start on logon.
    echo You can verify in Task Scheduler: PrometheusSignalService
) else (
    echo FAILED: Try running this script as Administrator.
)
pause
