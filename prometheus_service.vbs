' ============================================================================
' PROMETHEUS Signal Service — Hidden Launcher
' Runs the batch file without showing a console window
' ============================================================================

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "C:\Users\amanu\Desktop\Trading\prometheus_service.bat" & chr(34), 0, False
Set WshShell = Nothing
