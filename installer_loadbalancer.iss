


[Setup]
AppName=LoadBalancerSystem
AppVersion=1.0
AppVerName=LoadBalancerSystem Version 1.0
DefaultDirName={pf}\LoadBalancerSystem
DefaultGroupName=LoadBalancerSystem
OutputDir=.\Output
OutputBaseFilename=LoadBalancerSystemInstaller
LicenseFile=LICENSE.txt

[Files]
; Main executable from PyInstaller build
Source: "dist\LoadBalancerSystem.exe"; DestDir: "{app}"; Flags: ignoreversion

; License file (for easy access from install directory)
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion

; App icon used for shortcuts
Source: "loadbalancer.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\LoadBalancerSystem"; Filename: "{app}\LoadBalancerSystem.exe"; IconFilename: "{app}\loadbalancer.ico"

; Desktop shortcut
Name: "{commondesktop}\LoadBalancerSystem"; Filename: "{app}\LoadBalancerSystem.exe"; IconFilename: "{app}\loadbalancer.ico"

; Optional: Uninstall shortcut in Start Menu
Name: "{group}\Uninstall LoadBalancerSystem"; Filename: "{uninstallexe}"

[Registry]
; Optional registry entry for install path (e.g., for future updates or integrations)
Root: HKCU; Subkey: "Software\LoadBalancerSystem"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"

[Run]
; Optionally run app after install
; Filename: "{app}\TrackBallCoorApp.exe"; Description: "Launch TrackBallCoorApp"; Flags: nowait postinstall skipifsilent





