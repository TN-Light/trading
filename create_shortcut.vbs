' ============================================================================
' Create Prometheus startup shortcut
' ============================================================================

Set WshShell = CreateObject("WScript.Shell")
Set oLink = WshShell.CreateShortcut(WshShell.SpecialFolders("Startup") & "\Prometheus.lnk")

oLink.TargetPath = "wscript.exe"
oLink.Arguments = "C:\Users\amanu\Desktop\Trading\prometheus_service.vbs"
oLink.WorkingDirectory = "C:\Users\amanu\Desktop\Trading"
oLink.Description = "PROMETHEUS Signal Service"
oLink.Save

WScript.Echo "Shortcut created in Startup folder."
