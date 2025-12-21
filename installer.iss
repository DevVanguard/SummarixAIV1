[Setup]
AppName=SummarixAI
AppVersion=1.0.0
DefaultDirName={pf}\SummarixAI
DefaultGroupName=SummarixAI
OutputDir=installer
OutputBaseFilename=SummarixAI-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin

[Files]
Source: "dist\SummarixAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\SummarixAI"; Filename: "{app}\SummarixAI.exe"
Name: "{commondesktop}\SummarixAI"; Filename: "{app}\SummarixAI.exe"

[Run]
Filename: "{app}\SummarixAI.exe"; Description: "Launch SummarixAI"; Flags: nowait postinstall skipifsilent
