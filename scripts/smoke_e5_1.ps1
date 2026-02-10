# E2E smoke test for E5.1: CLI commands (engine-info, probe, extract-frames, validate-config).
# Run from repo root. Requires uv and Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& (Join-Path $ScriptDir "smoke_e5.ps1")

Write-Host "==> CLI engine-info test"
uv run python -m scindra_engine.cli engine-info --json

Write-Host "==> Generate synthetic video for CLI tests"
$SynthVideo = uv run python -c "
from pathlib import Path
from tests.fixtures.synth_video import make_synth_video
out_dir = Path('out/cli_smoke')
out_dir.mkdir(parents=True, exist_ok=True)
video = make_synth_video(out_dir, num_frames=16, size=(64, 48), fps=10.0)
print(video)
"

Write-Host "==> CLI probe test"
uv run scindra-engine probe --video $SynthVideo --json

Write-Host "==> CLI extract-frames test"
$FramesDir = "out/cli_smoke/frames"
if (Test-Path $FramesDir) {
    Remove-Item -Recurse -Force $FramesDir
}
uv run scindra-engine extract-frames --video $SynthVideo --out $FramesDir --count 5

Write-Host "==> Verify 5 PNGs were written"
uv run python -c "
from pathlib import Path
frames_dir = Path('out/cli_smoke/frames')
frames = list(frames_dir.glob('*.png'))
assert len(frames) == 5, f'Expected 5 PNGs, found {len(frames)}'
print(f'Found {len(frames)} PNG files as expected')
"

Write-Host "Smoke E5.1: OK"
