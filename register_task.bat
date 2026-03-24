@echo off
setlocal enableextensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VBS_PATH=%ROOT%\prometheus_service.vbs"
set "TASK_NAME=PrometheusSignalService"
set "RUN_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
set "RUN_VALUE=PrometheusSignalService"
set "RUN_CMD=C:\Windows\System32\wscript.exe %VBS_PATH%"

set "TASK_OK=0"
set "RUNKEY_OK=0"
set "SHORTCUT_OK=0"

echo Registering auto-start for PROMETHEUS...
echo Workspace: %ROOT%

schtasks /create /tn "%TASK_NAME%" /tr "%RUN_CMD%" /sc onlogon /rl highest /f >nul 2>&1
if errorlevel 1 goto task_fail
set "TASK_OK=1"
echo [OK] Scheduled Task created: %TASK_NAME%
goto task_done
:task_fail
echo [WARN] Scheduled Task creation failed (likely needs Administrator rights).
:task_done

reg add "%RUN_KEY%" /v "%RUN_VALUE%" /t REG_SZ /d "%RUN_CMD%" /f >nul 2>&1
if errorlevel 1 goto runkey_fail
set "RUNKEY_OK=1"
echo [OK] HKCU Run key created for current user.
goto runkey_done
:runkey_fail
echo [WARN] Could not create HKCU Run key fallback.
:runkey_done

cscript //nologo "%ROOT%\create_shortcut.vbs" >nul 2>&1
if errorlevel 1 goto shortcut_fail
set "SHORTCUT_OK=1"
echo [OK] Startup shortcut refreshed.
goto shortcut_done
:shortcut_fail
echo [WARN] Could not refresh Startup shortcut.
:shortcut_done

echo.
echo Summary:
if "%TASK_OK%"=="1" echo - Scheduled Task: READY
if "%TASK_OK%"=="0" echo - Scheduled Task: NOT READY
if "%RUNKEY_OK%"=="1" echo - HKCU Run key: READY
if "%RUNKEY_OK%"=="0" echo - HKCU Run key: NOT READY
if "%SHORTCUT_OK%"=="1" echo - Startup shortcut: READY
if "%SHORTCUT_OK%"=="0" echo - Startup shortcut: NOT READY

if "%TASK_OK%%RUNKEY_OK%%SHORTCUT_OK%"=="000" goto all_failed
echo.
echo SUCCESS: At least one auto-start method is active.
goto end_msg
:all_failed
echo.
echo FAILED: No auto-start method could be registered.
echo Try running this script as Administrator.
:end_msg
exit /b 0
