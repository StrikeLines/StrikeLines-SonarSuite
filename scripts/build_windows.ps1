param(
    [string]$Python = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "The Windows release must be built on Windows."
}

if (-not $Python) {
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
}

Push-Location $ProjectRoot
try {
    & $Python -c "import PyInstaller, PyQt5, pyproj, rasterio, scipy, skimage, pyxtf"
    if ($LASTEXITCODE -ne 0) {
        throw "The build environment is missing one or more release dependencies."
    }

    if (-not $SkipTests) {
        $env:QT_QPA_PLATFORM = "offscreen"
        $env:PYTHONPATH = (Join-Path $ProjectRoot "src")
        & $Python -m pytest -q
        if ($LASTEXITCODE -ne 0) {
            throw "Tests failed; release build cancelled."
        }
    }

    & $Python -m PyInstaller --noconfirm --clean SidescanTools.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    $Executable = Join-Path $ProjectRoot "dist\SidescanTools\SidescanTools.exe"
    if (-not (Test-Path -LiteralPath $Executable)) {
        throw "Build completed without producing $Executable"
    }

    $SmokeError = Join-Path $ProjectRoot "sidescantools-smoke-error.txt"
    Remove-Item -LiteralPath $SmokeError -ErrorAction SilentlyContinue
    $SmokeProcess = Start-Process -FilePath $Executable `
        -ArgumentList "--smoke-test" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    if ($SmokeProcess.ExitCode -ne 0) {
        $SmokeDetails = if (Test-Path -LiteralPath $SmokeError) {
            Get-Content -LiteralPath $SmokeError -Raw
        } else {
            "No diagnostic file was produced."
        }
        throw "Frozen application smoke test failed:`n$SmokeDetails"
    }
    Remove-Item -LiteralPath $SmokeError -ErrorAction SilentlyContinue
    Write-Host "Windows release created: $Executable"
}
finally {
    Pop-Location
}
