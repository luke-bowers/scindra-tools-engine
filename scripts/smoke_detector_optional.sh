#!/usr/bin/env bash
# Smoke test for scindra-engine with detector extras installed.
# Usage:
#   ./scripts/smoke_detector_optional.sh
#   YOLOX_ONNX_MODEL_PATH=/path/to/model.onnx ./scripts/smoke_detector_optional.sh
set -euo pipefail

echo "==> Installing scindra-engine[dev,detector]"
pip install -e ".[dev,detector]"

echo "==> Running ruff"
ruff check src/ tests/

echo "==> Running mypy"
mypy src/

echo "==> Running pytest (including detector_optional tests)"
pytest -v

if [ -n "${YOLOX_ONNX_MODEL_PATH:-}" ]; then
    echo "==> Running detect-mouse with real model"
    OUT_DIR="out/smoke_detector_optional"
    mkdir -p "$OUT_DIR"
    # Generate a synthetic video for testing
    python -c "
from tests.fixtures.synth_mouse_shadow import make_synth_mouse_shadow_video
from pathlib import Path
make_synth_mouse_shadow_video(Path('$OUT_DIR'), num_frames=60)
"
    scindra-engine detect-mouse \
        --video "$OUT_DIR/synth_mouse_shadow.mp4" \
        --out "$OUT_DIR/detect_out" \
        --model "$YOLOX_ONNX_MODEL_PATH" \
        --every-n 5

    # Check outputs exist
    test -f "$OUT_DIR/detect_out/detections.csv" || { echo "FAIL: detections.csv missing"; exit 1; }
    echo "==> detect-mouse outputs verified"
else
    echo "SKIP: no YOLOX model provided (set YOLOX_ONNX_MODEL_PATH to enable)"
fi

echo "==> SMOKE_DETECTOR_OPTIONAL OK"
