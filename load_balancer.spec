
# load_balancer_ui.spec

# PyInstaller spec file for load_balancer_ui

# Import necessary modules
from PyInstaller.building.build_main import EXE, PyInstallerIOError

# Define the entry point script
entry_point = 'src/ui/load_balancer_ui.py'

# Define the analysis object
a = Analysis(
    ['src/ui/load_balancer_ui.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=['PySide6'],  # Exclude PySide6 if you are using PyQt5
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

# Define the PyInstaller object
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Define the EXE object
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='load_balancer_ui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False if you want a windowed application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    set_exe_build_timestamp=False,  # Disable setting the build timestamp
)

# Define the COLLECT object
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='load_balancer_ui',
)



