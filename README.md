# scindra-engine

Scindra tools engine.

## Requirements

- Python 3.11+

## Installation

```bash
pip install scindra-engine
```

For development, the project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible installs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Usage

```bash
scindra-engine --version
```

## Detector-assisted tracking (optional)

The engine supports an optional YOLOX-based object detector that constrains
centroid tracking to a detection-based ROI. This reduces false-tracking on
shadows and other artefacts.

### Installing detector dependencies

```bash
uv pip install scindra-engine[detector]
```

This adds `onnxruntime` (CPU-only, MIT license). The core engine continues to
work without it.

### Running with the detector

```bash
# Track with detector-assisted ROI
scindra-engine track-centroid \
  --video path/to/video.mp4 \
  --out out/results \
  --detector \
  --detector-model /path/to/yolox_mouse_640.onnx

# Standalone detection (debug / verification)
scindra-engine detect-mouse \
  --video path/to/video.mp4 \
  --out out/detections \
  --model /path/to/yolox_mouse_640.onnx
```

### Model provisioning

The engine **does not** ship model weights by default. Model resolution order:

1. `--detector-model` / `--model` CLI flag (highest precedence)
2. `SCINDRA_YOLOX_ONNX_PATH` environment variable
3. Packaged asset at `scindra_engine/assets/models/yolox_mouse_640.onnx` (for
   desktop bundling)

If no model is found, the detector is unavailable and the pipeline continues
with classical-only tracking (a `DETECTOR_UNAVAILABLE` warning is emitted).

An optional `.json` sidecar (same name as the `.onnx` file) can specify model
metadata (input size, score threshold, NMS IoU, class names). If absent, safe
defaults are used.

### Training a mouse detector

You can train a YOLOX-Nano mouse detector from the [Kumar Lab Single Mouse Tracking Annotated Dataset](https://www.kumarlab.org/2019/02/12/single-mouse-tracking-annotated-dataset/) (OFA_Dataset). Training uses a **separate** Python environment (YOLOX and its dependencies are not in the main project).

#### 1. Get the dataset

- Download **OFA_Dataset** (e.g. from [Zenodo](https://zenodo.org/record/5806397) or the Kumar Lab FTP; use an FTP client like FileZilla if the link is `ftp://`).
- Extract it so you have a folder containing `Ref/` (images) and `Ell/` (ellipse-fit `.txt` annotations). Example: `inputs/OFA_Dataset/` or `C:\datasets\OFA_Dataset`.

#### 2. Create the training environment

YOLOX is not in `pyproject.toml`. Create a dedicated venv and install YOLOX plus dependencies:

```powershell
# Windows (PowerShell) – from repo root
python -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pycocotools loguru tqdm pyyaml tabulate
pip install --no-deps "git+https://github.com/Megvii-BaseDetection/YOLOX.git"
```

On Linux/macOS, use `source .venv-train/bin/activate` and the same `pip install` lines. If you don’t have CUDA, omit the PyTorch index URL and install default `torch`/`torchvision`.

**Optional:** On Windows, YOLOX’s COCO evaluation may try to load a C++ extension that needs ninja. The experiment file `scripts/yolox_mouse_exp.py` patches the evaluator to use standard `pycocotools` COCOeval so training works without ninja.

#### 3. Run the training pipeline

The script **converts** the Kumar dataset to COCO, **trains** YOLOX-Nano, and **exports** the best checkpoint to ONNX. It must be run with the **training** venv active (so `python` is the one that has `yolox`).

**Windows (PowerShell):**

```powershell
.\.venv-train\Scripts\Activate.ps1
.\scripts\train_yolox_mouse.ps1 -DatasetDir "inputs\OFA_Dataset"
```

**Linux/macOS (bash):** There is no `train_yolox_mouse.sh`. Run the same steps manually:

```bash
source .venv-train/bin/activate
export YOLOX_DATA_DIR=datasets/kumar_mouse_coco
export YOLOX_OUTPUT_DIR=YOLOX_outputs

# Convert
python scripts/convert_kumar_to_coco.py --dataset-dir inputs/OFA_Dataset --out-dir datasets/kumar_mouse_coco

# Download pretrained weights (once)
mkdir -p weights
# Download yolox_nano.pth from https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth into weights/

# Train (80 epochs, batch 16, 1 GPU; max_epoch is in scripts/yolox_mouse_exp.py)
python -m yolox.tools.train -f scripts/yolox_mouse_exp.py -d 1 -b 16 --fp16 -c weights/yolox_nano.pth

# Export best checkpoint to ONNX
python scripts/export_yolox_onnx.py -f scripts/yolox_mouse_exp.py -c YOLOX_outputs/yolox_mouse_nano/best_ckpt.pth --out models/yolox_mouse_640.onnx
```

**Optional parameters (PowerShell):** `-OutputDir`, `-ModelDir`, `-Epochs` (display only; actual `max_epoch` is in `yolox_mouse_exp.py`), `-BatchSize`, `-GPUs`.

#### 4. Verify labels (optional)

After conversion, you can check that bounding boxes match the mice before training:

```bash
python scripts/verify_training_labels.py
```

Images are written to `out/verify_labels/`. If boxes are misaligned, fix the conversion (see `scripts/convert_kumar_to_coco.py` and the ellipse file format).

#### 5. Use the trained model

The pipeline writes:

- **ONNX:** `models/yolox_mouse_640.onnx` (and `models/yolox_mouse_640.json` if generated).
- **Checkpoints:** `YOLOX_outputs/yolox_mouse_nano/` (e.g. `best_ckpt.pth`, `last_epoch_ckpt.pth`).

Use the ONNX model with the engine:

```bash
scindra-engine track-centroid --video your_video.mp4 --out out/ --detector --detector-model models/yolox_mouse_640.onnx
# or
export SCINDRA_YOLOX_ONNX_PATH=models/yolox_mouse_640.onnx
scindra-engine track-centroid --video your_video.mp4 --out out/ --detector
```

For a quick visual check of detections only:

```bash
scindra-engine detect-mouse --video your_video.mp4 --out out/detections --model models/yolox_mouse_640.onnx
```

## Smoke tests

The `smoke_latest` script provides an adaptive smoke test that detects available CLI capabilities and exercises the most end-to-end path available. It uses CLI commands exclusively and produces artifacts in `out/smoke_latest/<timestamp>/`.

**Basic usage:**
```bash
bash scripts/smoke_latest.sh
```

**With optional real video testing:**
```bash
VIDEO_PATH=/path/to/video.mp4 bash scripts/smoke_latest.sh
```

**Strict mode (fail on real video errors):**
```bash
STRICT_REAL_VIDEO=1 VIDEO_PATH=/path/to/video.mp4 bash scripts/smoke_latest.sh
```

The script automatically adapts to available capabilities:
- **Baseline UX** (always): `engine-info`, `probe`, `extract-frames`
- **E3 Config UX** (if available): `init-config`, `validate-config`
- **Track-Centroid** (if available): runs on synthetic fixtures, validates outputs including `overlay.mp4` and `heatmap.png` when supported
- **E6.2+ features** (if available): validates `manifest.json` and `support_bundle.zip`
- **Auto-Setup** (if available): tests arena and assay detection
- **Batch** (if available): tests batch processing

**Windows PowerShell:**
```powershell
.\scripts\smoke_latest.ps1
```

## How to cut a release

1. **Bump version**  
   Edit `version` in [pyproject.toml](pyproject.toml) (e.g. `0.1.0` → `0.2.0`). The package reads this as the single source of truth.

2. **Validate locally**  
   Run the smoke script so lint, typecheck, tests, build, twine check, and a local wheel install are exercised before tagging:
   - **mac/Linux:** `./scripts/smoke_release_local.sh`
   - **Windows:** `./scripts/smoke_release_local.ps1`  
   Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. For full wheelhouse/offline validation (constraints + offline install), run `./scripts/smoke_constraints_wheelhouse_local.sh` or `./scripts/smoke_constraints_wheelhouse_local.ps1` (requires Python 3.11+ and pip only).

3. **Tag**  
   Create a tag that matches the version in `pyproject.toml` (with a `v` prefix), e.g.:
   ```bash
   git tag v0.2.0
   ```

4. **Push the tag**  
   Pushing the tag triggers the release workflow (lint, typecheck, tests, version check, build, twine check, publish to PyPI, create GitHub Release):
   ```bash
   git push origin v0.2.0
   ```

5. **Where artifacts appear**  
   - **GitHub:** The tag’s [Releases](https://docs.github.com/en/repositories/releasing-projects-on-github) page will have the sdist, wheel, and `constraints-desktop.txt` attached.
   - **PyPI:** The package will be published to [PyPI](https://pypi.org/project/scindra-engine/) once the workflow completes.

### constraints-desktop.txt and Desktop Pro wheelhouse

`constraints-desktop.txt` is a pip constraints file generated from the built wheel’s resolved environment: after installing the wheel in a temporary venv, we run `pip freeze` and strip `pip`, `setuptools`, `wheel`, and the project itself. The result pins exact versions of the runtime dependencies only, so `pip download -c constraints-desktop.txt <wheel>` uses the same dependency versions every time without conflicting with the local wheel.

**Desktop Pro** bundles an offline wheelhouse per platform. It downloads `constraints-desktop.txt` from the GitHub Release and uses it with `pip download -c constraints-desktop.txt ...` when building the wheelhouse, so the same dependency versions are used every time. The file is not published to PyPI; it is only attached to the GitHub Release.

### PyPI publishing (Trusted Publishing / OIDC)

The workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) when no token is configured, so you don’t need to store a PyPI API token in GitHub.

**One-time setup on PyPI:**

1. Open your project on PyPI → **Settings** → **Publishing** → **Add a new trusted publisher**.
2. Choose **GitHub Actions**.
3. Set **Owner** and **Repository** to this repo.
4. Set **Workflow name** to `release.yml`.
5. Optionally set **Environment** (e.g. `pypi`) if you use a GitHub environment for approvals.

The workflow job already has `id-token: write` and uses `pypa/gh-action-pypi-publish` without username/password when OIDC is configured.

**Fallback (API token):**  
If you prefer not to use Trusted Publishing, add a repository secret `PYPI_API_TOKEN` with a PyPI API token, and in the workflow pass it to the publish step (e.g. `password: ${{ secrets.PYPI_API_TOKEN }}`, `username: __token__`). See the [action’s documentation](https://github.com/pypa/gh-action-pypi-publish) for details.

## License

[Apache-2.0](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
