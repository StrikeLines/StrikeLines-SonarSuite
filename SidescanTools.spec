# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)
source_root = project_root / "src"

datas = [
    (str(project_root / "README.md"), "."),
    (str(project_root / "LICENSE"), "."),
    (str(source_root / "sidescantools" / "res" / "icon.ico"), "sidescantools/res"),
    (
        str(source_root / "sidescantools" / "res" / "sonarsuite-logo.png"),
        "sidescantools/res",
    ),
]
binaries = []
hiddenimports = []

# These packages carry native DLLs and/or runtime databases which cannot be
# reconstructed from Python import analysis alone. PyInstaller de-duplicates
# entries also found by its standard hooks.
for package_name in ("rasterio", "pyproj", "pyxtf"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

analysis = Analysis(
    [str(source_root / "sidescantools" / "windows_app.py")],
    pathex=[str(source_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "packaging" / "windows_runtime_hook.py")],
    excludes=[
        "napari",
        "pygmt",
        "PySide2",
        "PySide6",
        "PyQt6",
        "tkinter",
        "pytest",
    ],
    noarchive=False,
    optimize=1,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SidescanTools",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(source_root / "sidescantools" / "res" / "icon.ico"),
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SidescanTools",
)
