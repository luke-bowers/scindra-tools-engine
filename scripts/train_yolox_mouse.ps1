# train_yolox_mouse.ps1
# End-to-end: convert dataset, train YOLOX-Nano, export to ONNX.
#
# Prerequisites:
#   1. Python 3.10+ with CUDA (recommended) or CPU
#   2. Kumar Lab OFA_Dataset extracted somewhere
#   3. YOLOX installed: pip install yolox==0.3.0 (or from source)
#
# Usage:
#   .\scripts\train_yolox_mouse.ps1 -DatasetDir "C:\datasets\OFA_Dataset"

param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetDir,

    [string]$OutputDir = "YOLOX_outputs",
    [string]$ModelDir  = "models",
    [int]$Epochs       = 80,
    [int]$BatchSize    = 16,
    [int]$GPUs         = 1
)

$ErrorActionPreference = "Stop"

Write-Host "=== YOLOX Mouse Detector Training Pipeline ===" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Convert dataset ---
Write-Host "[1/4] Converting Kumar Lab dataset to COCO format..." -ForegroundColor Yellow
$CocoDir = "datasets\kumar_mouse_coco"

python scripts\convert_kumar_to_coco.py `
    --dataset-dir $DatasetDir `
    --out-dir $CocoDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dataset conversion failed." -ForegroundColor Red
    exit 1
}

# --- Step 2: Download YOLOX-Nano pretrained weights (COCO) ---
Write-Host ""
Write-Host "[2/4] Checking for YOLOX-Nano pretrained weights..." -ForegroundColor Yellow
$WeightsDir = "weights"
$NanoWeights = "$WeightsDir\yolox_nano.pth"

if (-not (Test-Path $NanoWeights)) {
    New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
    Write-Host "  Downloading yolox_nano.pth from GitHub releases..."
    $url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
    Invoke-WebRequest -Uri $url -OutFile $NanoWeights
    Write-Host "  Downloaded: $NanoWeights"
} else {
    Write-Host "  Found existing weights: $NanoWeights"
}

# --- Step 3: Train ---
Write-Host ""
Write-Host "[3/4] Training YOLOX-Nano on mouse dataset ($Epochs epochs)..." -ForegroundColor Yellow
Write-Host "  Batch size: $BatchSize, GPUs: $GPUs"
Write-Host "  Dataset: $CocoDir"
Write-Host ""

$env:YOLOX_DATA_DIR = $CocoDir
$env:YOLOX_OUTPUT_DIR = $OutputDir

# Note: -e is --start_epoch in YOLOX, NOT --max_epoch.
# max_epoch is set in the experiment file (yolox_mouse_exp.py).
python -m yolox.tools.train `
    -f scripts\yolox_mouse_exp.py `
    -d $GPUs `
    -b $BatchSize `
    --fp16 `
    -c $NanoWeights

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed." -ForegroundColor Red
    exit 1
}

# --- Step 4: Export to ONNX ---
Write-Host ""
Write-Host "[4/4] Exporting best checkpoint to ONNX..." -ForegroundColor Yellow

$BestCkpt = "$OutputDir\yolox_mouse_nano\best_ckpt.pth"
$OnnxOut = "$ModelDir\yolox_mouse_640.onnx"

if (-not (Test-Path $BestCkpt)) {
    # Try latest checkpoint
    $BestCkpt = "$OutputDir\yolox_mouse_nano\latest_ckpt.pth"
}

python scripts\export_yolox_onnx.py `
    -f scripts\yolox_mouse_exp.py `
    -c $BestCkpt `
    --out $OnnxOut

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: ONNX export failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Model exported to: $OnnxOut" -ForegroundColor Green
Write-Host "Sidecar metadata:  $ModelDir\yolox_mouse_640.json" -ForegroundColor Green
Write-Host ""
Write-Host "Use with scindra-engine:" -ForegroundColor Cyan
Write-Host "  scindra-engine track-centroid --detector --detector-model $OnnxOut --video your_video.mp4 --out out/" -ForegroundColor White
Write-Host ""
Write-Host "Or set the environment variable:" -ForegroundColor Cyan
Write-Host "  `$env:SCINDRA_YOLOX_ONNX_PATH = '$(Resolve-Path $OnnxOut)'" -ForegroundColor White
