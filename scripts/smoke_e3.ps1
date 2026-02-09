# E2E smoke test for E3: schemas, validation, and schema generation.
# Run from repo root. Requires Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

Write-Host "==> Install dev dependencies"
python -m pip install -e ".[dev]"

Write-Host "==> Lint"
python -m ruff check .

Write-Host "==> Type check"
python -m mypy src

Write-Host "==> Test"
python -m pytest

Write-Host "==> Generate schemas"
python scripts/generate_schemas.py

Write-Host "==> Check schema drift"
python scripts/check_schemas_up_to_date.py

Write-Host "==> Quick import test"
python -c "from scindra_engine.schemas import AnalysisConfig; AnalysisConfig(assay={'selection_mode':'AUTO'}, video={'path':'x.mp4'}, outputs={'out_dir':'out'})"

Write-Host "Smoke E3: OK"
