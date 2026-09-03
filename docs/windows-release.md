# Building the Windows release

SidescanTools is distributed as a PyInstaller one-folder application. End
users do not need Python, Conda, or the source repository installed.

## Build environment

Build on 64-bit Windows from the full project environment. The environment
must contain PyInstaller plus the application dependencies, including PyQt5,
Rasterio/GDAL, PyProj/PROJ, SciPy, scikit-image, and pyxtf.

From an activated environment, install the project and build dependency:

```powershell
python -m pip install -e .
python -m pip install pyinstaller
```

Then run:

```powershell
.\scripts\build_windows.ps1 -Python (Get-Command python).Source
```

The script runs the test suite, builds the application, and runs a frozen-app
smoke test covering Qt startup, XTF support, PROJ transformation, and in-memory
GeoTIFF creation. It writes the release to:

```text
dist\SidescanTools\SidescanTools.exe
```

Distribute the complete `dist\SidescanTools` directory. Do not copy the EXE by
itself; the adjacent `_internal` directory contains Qt, GDAL, PROJ, and other
required runtime files.

## Release verification

Before publishing, test the copied bundle on a clean Windows computer:

1. Launch without command-line arguments and confirm the idle workspace opens.
2. Open both a JSF and an XTF file.
3. Create and edit a contact database.
4. Export EPSG:4326 and EPSG:3857 GeoTIFFs.
5. Restart the application and confirm per-file CFG settings are restored.

After the one-folder build is proven, it can be wrapped in an installer such
as Inno Setup. The installer may be a single download even though the installed
application remains a folder of native libraries and data files.
