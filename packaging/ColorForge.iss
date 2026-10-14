; Installer Windows (Inno Setup 6)
#define AppVersion GetEnv("CF_VERSION")
[Setup]
AppId={{6C1F4E2A-3B7D-4C8E-9A51-0F2D7E9B4C11}
AppName=ColorForge AI
AppVersion={#AppVersion}
AppPublisher=ColorForge
DefaultDirName={autopf}\ColorForge AI
DefaultGroupName=ColorForge AI
OutputDir=..\dist
OutputBaseFilename=ColorForge-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\ColorForge.exe

[Languages]
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un'icona sul desktop"; Flags: unchecked

[Files]
Source: "..\dist\ColorForge\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\ColorForge AI"; Filename: "{app}\ColorForge.exe"
Name: "{group}\Disinstalla ColorForge AI"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ColorForge AI"; Filename: "{app}\ColorForge.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ColorForge.exe"; Description: "Avvia ColorForge AI"; Flags: nowait postinstall skipifsilent
