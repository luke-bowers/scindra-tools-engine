"""Export a trained YOLOX model to ONNX format for use with scindra-engine.

This script exports the YOLOX checkpoint to ONNX and writes the sidecar
metadata JSON that the engine's ModelResolver expects.

Usage:
    python scripts/export_yolox_onnx.py \
        -f scripts/yolox_mouse_exp.py \
        -c YOLOX_outputs/yolox_mouse_nano/best_ckpt.pth \
        --out models/yolox_mouse_640.onnx

The script produces two files:
    models/yolox_mouse_640.onnx   - The ONNX model
    models/yolox_mouse_640.json   - Sidecar metadata for scindra-engine
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOX checkpoint to ONNX.")
    parser.add_argument("-f", "--exp-file", type=str, required=True, help="Path to YOLOX experiment file.")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--out", type=str, default="models/yolox_mouse_640.onnx", help="Output ONNX file path.")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640], help="Input H W (default: 640 640).")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--no-decode", action="store_true", help="Export without decode head (raw grid output).")
    args = parser.parse_args()

    # Import the experiment
    sys.path.insert(0, str(Path(args.exp_file).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_module", args.exp_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    exp = mod.Exp()

    # Build model
    model = exp.get_model()
    model.eval()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Set the model to deploy mode (decode in forward pass)
    if not args.no_decode:
        model.head.decode_in_inference = True

    # Create dummy input
    h, w = args.input_size
    dummy = torch.randn(1, 3, h, w)

    # Export
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes=None,  # Fixed input size for determinism
    )
    print(f"Exported ONNX model: {out_path}")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        import numpy as np
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: np.random.randn(1, 3, h, w).astype(np.float32)})
        print(f"ONNX verification passed. Output shape: {out[0].shape}")
        # Expected: (1, N, 5 + num_classes) where N = sum of grid cells
    except ImportError:
        print("onnxruntime not installed; skipping ONNX verification.")

    # Write sidecar metadata JSON
    meta = {
        "input_size": [h, w],
        "num_classes": exp.num_classes,
        "class_names": ["mouse"],
        "score_thresh": 0.25,
        "nms_iou": 0.45,
        "description": "YOLOX-Nano fine-tuned on Kumar Lab OFA mouse dataset",
        "training": {
            "epochs": exp.max_epoch,
            "input_size": [h, w],
            "dataset": "Kumar Lab Single Mouse Tracking (OFA)",
            "base_model": "yolox_nano",
        },
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote sidecar metadata: {meta_path}")

    print(f"\n--- Ready to use with scindra-engine ---")
    print(f"  scindra-engine track-centroid --detector --detector-model {out_path} ...")
    print(f"  Or set: $env:SCINDRA_YOLOX_ONNX_PATH = '{out_path.resolve()}'")


if __name__ == "__main__":
    main()
