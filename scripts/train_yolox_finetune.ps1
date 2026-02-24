# train_yolox_finetune.ps1
# Fine-tune YOLOX from an existing checkpoint (e.g. Kumar-trained best_ckpt.pth).
# Use after preparing a COCO dataset (e.g. prepare_cvat_coco.py for CVAT exports).
#
# Prerequisites:
#   - .venv-train activated (or python has yolox)
#   - Prepared COCO dataset at $CocoDir (annotations/train.json, val.json, images/train/, images/val/)
#   - Checkpoint to resume from (e.g. YOLOX_outputs/yolox_mouse_nano/best_ckpt.pth)
#
# Usage:
#   .\.venv-train\Scripts\Activate.ps1
#   .\scripts\train_yolox_finetune.ps1 -CocoDir "datasets\ezm_coco" -Checkpoint "YOLOX_outputs\yolox_mouse_nano\best_ckpt.pth"

param(
    [Parameter(Mandatory=$true)]
    [string]$CocoDir,

    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,

    [string]$OutputDir = "YOLOX_outputs",
    [string]$ModelDir  = "models",
    [string]$ExpName  = "yolox_mouse_nano",
    [int]$BatchSize   = 16,
    [int]$GPUs        = 1
)

$ErrorActionPreference = "Stop"

Write-Host "=== YOLOX Fine-tune (from checkpoint) ===" -ForegroundColor Cyan
Write-Host "  Dataset:  $CocoDir" -ForegroundColor White
Write-Host "  Checkpoint: $Checkpoint" -ForegroundColor White
Write-Host ""

if (-not (Test-Path $Checkpoint)) {
    Write-Host "ERROR: Checkpoint not found: $Checkpoint" -ForegroundColor Red
    exit 1
}

$trainJson = Join-Path $CocoDir "annotations\train.json"
if (-not (Test-Path $trainJson)) {
    Write-Host "ERROR: Dataset missing annotations. Expected: $trainJson" -ForegroundColor Red
    Write-Host "  Run: python scripts/prepare_cvat_coco.py --cvat-export inputs/EZM_Dataset --out-dir $CocoDir" -ForegroundColor Yellow
    exit 1
}

$env:YOLOX_DATA_DIR = $CocoDir
$env:YOLOX_OUTPUT_DIR = $OutputDir

Write-Host "[1/2] Training (fine-tune)..." -ForegroundColor Yellow
python -m yolox.tools.train `
    -f scripts\yolox_mouse_exp.py `
    -d $GPUs `
    -b $BatchSize `
    --fp16 `
    -c $Checkpoint

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/2] Exporting best checkpoint to ONNX..." -ForegroundColor Yellow
$BestCkpt = "$OutputDir\$ExpName\best_ckpt.pth"
if (-not (Test-Path $BestCkpt)) {
    $BestCkpt = "$OutputDir\$ExpName\last_epoch_ckpt.pth"
}
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
$OnnxOut = "$ModelDir\yolox_mouse_640.onnx"

python scripts\export_yolox_onnx.py `
    -f scripts\yolox_mouse_exp.py `
    -c $BestCkpt `
    --out $OnnxOut

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: ONNX export failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. Model: $OnnxOut" -ForegroundColor Green
