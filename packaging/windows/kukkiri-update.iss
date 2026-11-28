#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F1C8E2A-3B7D-4C1E-9A55-0D4E2B7C9F10}
AppName=kukkiri
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\kukkiri
UsePreviousAppDir=yes
DisableDirPage=auto
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
Uninstallable=no
CreateUninstallRegKey=no
UpdateUninstallLogAppName=no
OutputDir=..\..\dist
OutputBaseFilename=kukkiri-update-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=kukkiri.ico
WizardStyle=modern

[InstallDelete]
Type: filesandordirs; Name: "{app}\Lib\site-packages\kukkiri.new"
Type: filesandordirs; Name: "{app}\Lib\site-packages\kukkiri.old"

[Files]
Source: "..\..\dist\update\kukkiri\*"; DestDir: "{app}\Lib\site-packages\kukkiri.new"; Flags: recursesubdirs ignoreversion

[Code]
function KukkiriRunning(): Boolean;
var
  Locator, Service, Procs: Variant;
  Path: String;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Path := ExpandConstant('{app}');
    StringChangeEx(Path, '\', '\\', True);
    Procs := Service.ExecQuery('SELECT ProcessId FROM Win32_Process WHERE ExecutablePath LIKE ''' + Path + '\\python%''');
    Result := Procs.Count > 0;
  except
    Result := False;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not FileExists(ExpandConstant('{app}\python.exe')) then
    Result := 'kukkiri is not installed in ' + ExpandConstant('{app}') + '. Install kukkiri with the full installer first.'
  else if KukkiriRunning() then
    Result := 'kukkiri is running. Close it (stopping any restore) and run the update again.';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Pkg, NewPkg, OldPkg: String;
begin
  if CurStep = ssPostInstall then begin
    Pkg := ExpandConstant('{app}\Lib\site-packages\kukkiri');
    NewPkg := Pkg + '.new';
    OldPkg := Pkg + '.old';
    if not RenameFile(Pkg, OldPkg) then
      RaiseException('Could not move the installed kukkiri aside; nothing was changed.');
    if not RenameFile(NewPkg, Pkg) then begin
      RenameFile(OldPkg, Pkg);
      RaiseException('Could not put the new kukkiri in place; the installed one was kept.');
    end;
    DelTree(OldPkg, True, True, True);
  end;
end;
