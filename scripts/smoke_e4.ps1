# E2E smoke test for E4: assay registry.
# Run from repo root. Requires uv and Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& (Join-Path $ScriptDir "smoke_e3.ps1")

Write-Host "==> Assay registry checks"
uv run python -c "from scindra_engine.assays.registry import list_assays; a=list_assays(); assert 'OPEN_FIELD' in a and len(a)==12"
uv run python -c "from scindra_engine.assays.registry import get_assay; d=get_assay('EPM'); assert d.zone_template_id=='EPM_ARMS'"

Write-Host "Smoke E4: OK"

