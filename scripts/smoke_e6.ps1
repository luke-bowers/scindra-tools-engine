# E2E smoke test for E6: track-centroid pipeline.
# Run from repo root. Requires uv and Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& (Join-Path $ScriptDir "smoke_e5_1.ps1")

Write-Host "==> Generate synthetic mouse+shadow video"
$SynthVideo = uv run python -c "
from pathlib import Path
from tests.fixtures.synth_mouse_shadow import make_synth_mouse_shadow_video
out_dir = Path('out/e6')
out_dir.mkdir(parents=True, exist_ok=True)
video = make_synth_mouse_shadow_video(out_dir)
print(video)
"

Write-Host "==> Run track-centroid on synthetic video"
uv run scindra-engine track-centroid --video $SynthVideo --out out/e6

Write-Host "==> Validate per_frame.csv and coverage"
uv run python -c "
from pathlib import Path
import csv

run_dirs = sorted(Path('out/e6').glob('run_*'), key=lambda p: p.stat().st_mtime)
assert run_dirs, 'No run directories created'
run_dir = run_dirs[-1]
csv_path = run_dir / 'per_frame.csv'
assert csv_path.exists(), f'Missing {csv_path}'
with csv_path.open() as f:
    reader = csv.reader(f)
    header = next(reader)
    expected = ['frame', 'x', 'y', 'area', 'confidence', 'flags']
    assert header == expected, f'Unexpected header: {header}'
    rows = list(reader)
    assert rows, 'No rows in per_frame.csv'
    tracked = sum(1 for row in rows if row[1] and row[2])
    coverage = tracked / len(rows)
    assert coverage >= 0.9, f'Coverage too low: {coverage:.3f}'
print('Synthetic tracking coverage OK:', coverage)
"

if ($env:VIDEO_PATH) {
    Write-Host "==> Run track-centroid on real video"
    uv run scindra-engine track-centroid --video $env:VIDEO_PATH --out out/e6_real
    uv run python -c "
from pathlib import Path
run_dirs = sorted(Path('out/e6_real').glob('run_*'), key=lambda p: p.stat().st_mtime)
assert run_dirs, 'No run directories created for real video'
csv_path = run_dirs[-1] / 'per_frame.csv'
assert csv_path.exists(), f'Missing {csv_path}'
print('Real video output OK:', csv_path)
"
}

Write-Host "Smoke E6: OK"
