#!/usr/bin/env bash
# E2E smoke test for E5: video I/O utilities.
# Run from repo root. Requires Python 3.11+.
set -eu

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

"$SCRIPT_DIR/smoke_e3.sh"

echo "==> Video I/O tests"
python -m pytest -k video_io

echo "==> Out dir smoke"
python -c "from scindra_engine.video_io import VideoReader; import pathlib; p = pathlib.Path('out'); p.mkdir(exist_ok=True);"

echo "Smoke E5: OK"

