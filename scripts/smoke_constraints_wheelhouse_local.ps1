# E2E smoke: constraints-desktop.txt + offline wheelhouse. Run from repo root.
# Requires Python 3.11+ and pip (no uv). Cleans dist/ and .smoke_wheelhouse, then:
# pip dev install, ruff/mypy/pytest, build, twine, generate constraints,
# pip download wheelhouse (wheels only), offline venv install, scindra-engine --version.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

$SmokeDir = Join-Path $Root ".smoke_wheelhouse"
$Wheelhouse = Join-Path $SmokeDir "wheelhouse"
$Venv2 = Join-Path $SmokeDir "venv2"
$Constraints = Join-Path $SmokeDir "constraints-desktop.txt"

Write-Host "==> Clean dist/ and smoke temp dir"
if (Test-Path $SmokeDir) { Remove-Item -Recurse -Force $SmokeDir }
if (Test-Path (Join-Path $Root "dist")) {
    Remove-Item -Path (Join-Path $Root "dist\*") -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "==> Dev install"
python -m pip install -q -e ".[dev]"

Write-Host "==> Lint"
ruff check .

Write-Host "==> Type check"
mypy src

Write-Host "==> Test"
$env:PYTHONPATH = "src"
pytest

Write-Host "==> Build"
python -m build

Write-Host "==> Twine check"
python -m twine check dist/*

Write-Host "==> Resolve wheel and generate constraints"
$Wheel = Get-ChildItem -Path (Join-Path $Root "dist") -Filter "scindra_engine-*.whl" | Select-Object -First 1
if (-not $Wheel) { throw "No wheel found in dist/" }
New-Item -ItemType Directory -Path $SmokeDir -Force | Out-Null
python scripts/generate_constraints_desktop.py --wheel $Wheel.FullName --out $Constraints

Write-Host "==> Build wheelhouse (wheels only)"
New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null
python -m pip download --only-binary=:all: -d $Wheelhouse -c $Constraints $Wheel.FullName
if ($LASTEXITCODE -ne 0) { Write-Error "pip download failed"; exit 1 }
$Sdists = Get-ChildItem -Path "$Wheelhouse\*" -Include "*.tar.gz","*.zip" -File -ErrorAction SilentlyContinue
if ($Sdists) {
    Write-Error "sdist found in wheelhouse (expected wheels only): $($Sdists.FullName -join ', ')"
    exit 1
}

Write-Host "==> Create fresh venv and install offline"
python -m venv $Venv2
$Venv2Pip = Join-Path (Join-Path $Venv2 "Scripts") "pip.exe"
& $Venv2Pip install --no-index --find-links $Wheelhouse $Wheel.FullName
if ($LASTEXITCODE -ne 0) { Write-Error "pip install --no-index failed (wheelhouse may be incomplete)"; exit 1 }

Write-Host "==> Run scindra-engine --version"
$Cli = Join-Path (Join-Path $Venv2 "Scripts") "scindra-engine.exe"
& $Cli --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "scindra-engine --version failed"
    exit 1
}

Write-Host "Smoke constraints wheelhouse local: OK"
