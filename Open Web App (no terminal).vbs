' Starts the AquaForge web UI with auto-reload for development
' Changes to .py files will reload automatically when saved.
'
' First time only: run "py -3 -m pip install -r requirements.txt"

Option Explicit

Dim fso, shell, scriptDir, cmdLine
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Kill any existing instances first
shell.Run "taskkill /F /IM python.exe /FI ""WINDOWTITLE eq AquaForge*""", 0, True
WScript.Sleep 1000

cmdLine = "cmd /c " & Chr(34) & "cd /d " & Chr(34) & scriptDir & Chr(34) & " && set STREAMLIT_SERVER_HEADLESS=true&& py -3 -m streamlit run app.py --server.runOnSave=true" & Chr(34)

shell.Run cmdLine, 0, False
WScript.Sleep 5000
shell.Run "http://127.0.0.1:8501", 1, False
