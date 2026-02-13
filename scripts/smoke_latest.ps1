# Adaptive smoke test for scindra-engine CLI.
# Run from repo root. Requires uv and Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

Write-Host "==> Install dev dependencies (uv sync)"
uv sync --extra dev

Write-Host "==> Lint"
uv run ruff check .

Write-Host "==> Type check"
uv run mypy src

Write-Host "==> Test"
uv run pytest

Write-Host "==> Run smoke_latest"
uv run python scripts/smoke_latest.py
