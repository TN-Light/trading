' ============================================================================
' Create Prometheus startup shortcut
' ============================================================================

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)

Set oLink = WshShell.CreateShortcut(WshShell.SpecialFolders("Startup") & "\Prometheus.lnk")
oLink.TargetPath = "C:\Windows\System32\wscript.exe"
oLink.Arguments = Chr(34) & scriptDir & "\prometheus_service.vbs" & Chr(34)
oLink.WorkingDirectory = scriptDir
oLink.Description = "PROMETHEUS Signal Service"
oLink.Save

WScript.Echo "Shortcut created in Startup folder."
