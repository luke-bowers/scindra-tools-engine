#!/usr/bin/env bash
# E2E smoke test for E5.1: CLI commands (engine-info, probe, extract-frames, validate-config).
# Run from repo root. Requires Python 3.11+.
set -eu

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

"$SCRIPT_DIR/smoke_e5.sh"

echo "==> CLI engine-info test"
python -m scindra_engine.cli engine-info --json

echo "==> Generate synthetic video for CLI tests"
SYNTH_VIDEO=$(python -c "
from pathlib import Path
from tests.fixtures.synth_video import make_synth_video
out_dir = Path('out/cli_smoke')
out_dir.mkdir(parents=True, exist_ok=True)
video = make_synth_video(out_dir, num_frames=16, size=(64, 48), fps=10.0)
print(video)
")

echo "==> CLI probe test"
scindra-engine probe --video "$SYNTH_VIDEO" --json

echo "==> CLI extract-frames test"
FRAMES_DIR="out/cli_smoke/frames"
rm -rf "$FRAMES_DIR"
scindra-engine extract-frames --video "$SYNTH_VIDEO" --out "$FRAMES_DIR" --count 5

echo "==> Verify 5 PNGs were written"
python -c "
from pathlib import Path
frames_dir = Path('out/cli_smoke/frames')
frames = list(frames_dir.glob('*.png'))
assert len(frames) == 5, f'Expected 5 PNGs, found {len(frames)}'
print(f'Found {len(frames)} PNG files as expected')
"

echo "Smoke E5.1: OK"
