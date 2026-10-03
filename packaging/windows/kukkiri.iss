#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Spanning
  #define Spanning "no"
#endif
#ifndef Compression
  #define Compression "lzma2/max"
#endif

[Setup]
AppId={{6F1C8E2A-3B7D-4C1E-9A55-0D4E2B7C9F10}
AppName=kukkiri
AppVersion={#AppVersion}
AppPublisher=kukkiri authors
AppPublisherURL=https://kukkiri.com
DefaultDirName={localappdata}\Programs\kukkiri
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename=kukkiri-setup-{#AppVersion}
Compression={#Compression}
SolidCompression=yes
LZMANumBlockThreads=4
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DiskSpanning={#Spanning}
DiskSliceSize=2000000000
SetupIconFile=kukkiri.ico
UninstallDisplayIcon={app}\kukkiri.ico
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\..\dist\stage\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "kukkiri.ico"; DestDir: "{app}"

[Icons]
Name: "{autoprograms}\kukkiri"; Filename: "{app}\pythonw.exe"; Parameters: """{app}\kukkiri-gui.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\kukkiri.ico"
Name: "{autodesktop}\kukkiri"; Filename: "{app}\pythonw.exe"; Parameters: """{app}\kukkiri-gui.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\kukkiri.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\pythonw.exe"; Parameters: """{app}\kukkiri-gui.pyw"""; Description: "Start kukkiri"; Flags: postinstall nowait skipifsilent

[InstallDelete]
Type: filesandordirs; Name: "{app}\weights"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
