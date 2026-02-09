# E2E smoke test for E5: video I/O utilities.
# Run from repo root. Requires Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& (Join-Path $ScriptDir "smoke_e3.ps1")

Write-Host "==> Video I/O tests"
python -m pytest -k video_io

Write-Host "==> Out dir smoke"
python -c "from scindra_engine.video_io import VideoReader; import pathlib; p = pathlib.Path('out'); p.mkdir(exist_ok=True);"

Write-Host "Smoke E5: OK"

