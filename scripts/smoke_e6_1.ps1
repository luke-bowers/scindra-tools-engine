# E2E smoke test for E6.1: centroid visualization outputs (overlay + heatmap).
# Run from repo root. Requires uv and Python 3.11+.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

& (Join-Path $ScriptDir "smoke_e6.ps1")

Write-Host "==> Generate synthetic mouse+shadow video for E6.1"
$SynthVideo = uv run python -c "
from pathlib import Path
from tests.fixtures.synth_mouse_shadow import make_synth_mouse_shadow_video
out_dir = Path('out/e6_1')
out_dir.mkdir(parents=True, exist_ok=True)
video = make_synth_mouse_shadow_video(out_dir)
print(video)
"

Write-Host "==> Run track-centroid with overlay and heatmap on synthetic video"
uv run scindra-engine track-centroid --video $SynthVideo --out out/e6_1 --overlay --heatmap

Write-Host "==> Validate per_frame.csv, overlay.mp4, and heatmap.png"
uv run python -c "
from pathlib import Path

run_dirs = sorted(Path('out/e6_1').glob('run_*'), key=lambda p: p.stat().st_mtime)
assert run_dirs, 'No run directories created'
run_dir = run_dirs[-1]
csv_path = run_dir / 'per_frame.csv'
overlay_path = run_dir / 'overlay.mp4'
heatmap_path = run_dir / 'heatmap.png'
for p in (csv_path, overlay_path, heatmap_path):
    assert p.exists(), f'Missing {p}'
print('Synthetic visualization outputs OK:', csv_path, overlay_path, heatmap_path)
"

if ($env:VIDEO_PATH) {
    Write-Host "==> Run track-centroid with overlay and heatmap on real video"
    uv run scindra-engine track-centroid --video $env:VIDEO_PATH --out out/e6_1_real --overlay --heatmap
    uv run python -c "
from pathlib import Path

run_dirs = sorted(Path('out/e6_1_real').glob('run_*'), key=lambda p: p.stat().st_mtime)
assert run_dirs, 'No run directories created for real video'
run_dir = run_dirs[-1]
csv_path = run_dir / 'per_frame.csv'
overlay_path = run_dir / 'overlay.mp4'
heatmap_path = run_dir / 'heatmap.png'
for p in (csv_path, overlay_path, heatmap_path):
    assert p.exists(), f'Missing {p}'
print('Real video visualization outputs OK:', csv_path, overlay_path, heatmap_path)
"
}

Write-Host "Smoke E6.1: OK"

