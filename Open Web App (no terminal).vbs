' Starts the Vessel detection web UI in the background (no console window)
' and opens your default browser. Bookmark: http://127.0.0.1:8501
'
' First time only: install deps in a terminal once:
'   py -3 -m pip install -r requirements.txt

Option Explicit

Dim fso, shell, scriptDir, cmdLine
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Headless so Streamlit does not try to open a browser from a hidden process;
' we open the URL explicitly after the server has time to bind.
cmdLine = "cmd /c " & Chr(34) & "cd /d " & Chr(34) & scriptDir & Chr(34) & " && set STREAMLIT_SERVER_HEADLESS=true&& py -3 -m streamlit run app.py" & Chr(34)

shell.Run cmdLine, 0, False
WScript.Sleep 5000
shell.Run "http://127.0.0.1:8501", 1, False
