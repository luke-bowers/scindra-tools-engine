# Smoke test for release: lint, typecheck, test, build, twine check, install wheel, run --version.
# Run from repo root. Requires uv and Python 3.11.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

Write-Host "==> Lint"
uv run ruff check .

Write-Host "==> Type check"
uv run mypy src

Write-Host "==> Test"
uv run pytest

Write-Host "==> Build"
uv run python -m build

Write-Host "==> Twine check"
uv run twine check dist/*

Write-Host "==> Create temp venv and install wheel"
$SmokeVenv = Join-Path $Root ".smoke_venv"
if (Test-Path $SmokeVenv) { Remove-Item -Recurse -Force $SmokeVenv }
uv venv $SmokeVenv --python 3.11
& (Join-Path $SmokeVenv "Scripts" "Activate.ps1")
pip install -q --upgrade pip
$Wheel = Get-ChildItem -Path (Join-Path $Root "dist") -Filter "scindra_engine-*.whl" | Select-Object -First 1
if (-not $Wheel) { throw "No wheel found in dist/" }
pip install -q $Wheel.FullName

Write-Host "==> Run scindra-engine --version"
$versionOutput = scindra-engine --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "scindra-engine --version failed"
    exit 1
}

Write-Host "Smoke release local: OK"
