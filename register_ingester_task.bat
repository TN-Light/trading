@echo off
:: Check for Admin Privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :admin
) else (
    echo Requesting Administrative Privileges to Register Task...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~dpnx0\"' -Verb RunAs"
    exit /b
)

:admin
echo Registering Windows Task Scheduler Job for 09:14 AM Daily...
schtasks /create /tn "NMB_Live_Options_Ingester" /tr "%~dp0nmb_start_ingester.bat" /sc daily /st 09:14 /f
echo.
echo Task successfully registered!
pause
