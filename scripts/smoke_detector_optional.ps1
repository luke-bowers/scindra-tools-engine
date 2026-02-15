# Smoke test for scindra-engine with detector extras installed.
# Usage:
#   .\scripts\smoke_detector_optional.ps1
#   $env:YOLOX_ONNX_MODEL_PATH="C:\path\to\model.onnx"; .\scripts\smoke_detector_optional.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> Installing scindra-engine[dev,detector]"
pip install -e ".[dev,detector]"

Write-Host "==> Running ruff"
ruff check src/ tests/

Write-Host "==> Running mypy"
mypy src/

Write-Host "==> Running pytest (including detector_optional tests)"
pytest -v

$modelPath = $env:YOLOX_ONNX_MODEL_PATH
if ($modelPath) {
    Write-Host "==> Running detect-mouse with real model"
    $outDir = "out\smoke_detector_optional"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    python -c @"
from tests.fixtures.synth_mouse_shadow import make_synth_mouse_shadow_video
from pathlib import Path
make_synth_mouse_shadow_video(Path('$outDir'), num_frames=60)
"@

    scindra-engine detect-mouse `
        --video "$outDir\synth_mouse_shadow.mp4" `
        --out "$outDir\detect_out" `
        --model "$modelPath" `
        --every-n 5

    if (-not (Test-Path "$outDir\detect_out\detections.csv")) {
        Write-Error "FAIL: detections.csv missing"
        exit 1
    }
    Write-Host "==> detect-mouse outputs verified"
} else {
    Write-Host "SKIP: no YOLOX model provided (set YOLOX_ONNX_MODEL_PATH to enable)"
}

Write-Host "==> SMOKE_DETECTOR_OPTIONAL OK"
