@echo off
setlocal enableextensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VBS_PATH=%ROOT%\prometheus_service.vbs"
set "RUN_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
set "RUN_VALUE=PrometheusSignalService"
set "RUN_CMD=C:\Windows\System32\wscript.exe %VBS_PATH%"

set "RUNKEY_OK=0"

echo Registering auto-start for PROMETHEUS...
echo Workspace: %ROOT%

reg add "%RUN_KEY%" /v "%RUN_VALUE%" /t REG_SZ /d "%RUN_CMD%" /f >nul 2>&1
if errorlevel 1 goto runkey_fail
set "RUNKEY_OK=1"
echo [OK] HKCU Run key created for current user.
goto runkey_done
:runkey_fail
echo [WARN] Could not create HKCU Run key fallback.
:runkey_done

echo.
echo Summary:
if "%RUNKEY_OK%"=="1" echo - HKCU Run key: READY
if "%RUNKEY_OK%"=="0" echo - HKCU Run key: NOT READY

if "%RUNKEY_OK%"=="0" goto all_failed
echo.
echo SUCCESS: The auto-start method is active.
goto end_msg
:all_failed
echo.
echo FAILED: No auto-start method could be registered.
echo Try running this script as Administrator.
:end_msg
exit /b 0
