"""Prepare a CVAT COCO export for YOLOX training.

CVAT exports a zip with annotations/instances_default.json and images.
This script:
  - Loads the annotation JSON
  - By default keeps only images that have at least one bbox (--annotated-only; use --all-frames to include all)
  - Finds images (in <cvat_export>/images/ or <cvat_export>/)
  - Splits into train/val (default 85/15)
  - Writes a YOLOX-style dataset: annotations/train.json, val.json, images/train/, images/val/
  - Uses category "mouse" (id 1) for compatibility with the Kumar-trained model.

Usage:
  python scripts/prepare_cvat_coco.py \\
      --cvat-export inputs/EZM_Dataset \\
      --out-dir datasets/ezm_coco \\
      [--val-ratio 0.15] [--symlink] [--all-frames]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def find_image_dir(export_dir: Path, file_name: str) -> Path | None:
    """Return directory that contains the image, or None."""
    candidates = [
        export_dir / "images",
        export_dir / "images" / "default",
        export_dir,
    ]
    for candidate in candidates:
        if (candidate / file_name).exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CVAT COCO export for YOLOX (train/val split)."
    )
    parser.add_argument(
        "--cvat-export",
        type=Path,
        default=Path("inputs/EZM_Dataset"),
        help="Path to extracted CVAT export (contains annotations/ and images).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/ezm_coco"),
        help="Output dataset directory (YOLOX layout).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of images for validation (default 0.15).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        default=True,
        help="Include only images that have at least one bounding box (default: True).",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Include all images, even with no annotations (overrides --annotated-only).",
    )
    args = parser.parse_args()

    export_dir = args.cvat_export.resolve()
    out_dir = args.out_dir.resolve()

    # Find annotation file (CVAT uses instances_default.json)
    ann_file = export_dir / "annotations" / "instances_default.json"
    if not ann_file.exists():
        ann_file = export_dir / "annotations" / "train.json"
    if not ann_file.exists():
        raise SystemExit(f"Annotation file not found under {export_dir}/annotations/")

    data = json.loads(ann_file.read_text())
    images = data["images"]
    annotations = data["annotations"]
    categories = data.get("categories", [{"id": 1, "name": "mouse", "supercategory": "animal"}])

    # Normalize category to id=1 "mouse" for compatibility with Kumar-trained model
    cat_id_map = {}
    for c in categories:
        cat_id_map[c["id"]] = 1
    new_categories = [{"id": 1, "name": "mouse", "supercategory": "animal"}]
    for ann in annotations:
        ann["category_id"] = cat_id_map.get(ann["category_id"], 1)

    # Optionally keep only images that have at least one annotation
    if args.annotated_only and not args.all_frames:
        annotated_image_ids = {a["image_id"] for a in annotations}
        images = [im for im in images if im["id"] in annotated_image_ids]
        if not images:
            raise SystemExit(
                "No images have annotations. Annotate some frames in CVAT or use --all-frames."
            )
        print(f"  Kept {len(images)} images with at least one annotation (dropped the rest).")

    # Resolve image paths
    if not images:
        raise SystemExit("No images in the annotation file.")

    sample_name = images[0]["file_name"]
    img_dir = find_image_dir(export_dir, sample_name)
    if img_dir is None:
        raise SystemExit(
            f"Images not found. Looked for {sample_name} in "
            f"{export_dir / 'images'} and {export_dir}. "
            "Ensure the CVAT export was extracted with images."
        )

    # Train/val split by image
    random.seed(args.seed)
    image_ids = [im["id"] for im in images]
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * args.val_ratio))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    def split_data(ids: set) -> tuple[list, list]:
        imgs = [im for im in images if im["id"] in ids]
        anns = [a for a in annotations if a["image_id"] in ids]
        return imgs, anns

    train_images, train_anns = split_data(train_ids)
    val_images, val_anns = split_data(val_ids)

    out_ann = out_dir / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)

    for split_name, imgs, anns in [
        ("train", train_images, train_anns),
        ("val", val_images, val_anns),
    ]:
        coco = {
            "images": imgs,
            "annotations": anns,
            "categories": new_categories,
        }
        (out_ann / f"{split_name}.json").write_text(json.dumps(coco, indent=2))
        dst_dir = out_dir / "images" / split_name
        for im in imgs:
            src = img_dir / im["file_name"]
            if not src.exists():
                raise SystemExit(f"Missing image: {src}")
            dst = dst_dir / im["file_name"]
            if dst.exists():
                continue
            if args.symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        print(f"  {split_name}: {len(imgs)} images, {len(anns)} annotations -> {dst_dir}")

    print(f"\nDone! Dataset written to: {out_dir}")
    print(f"  annotations/train.json, val.json")
    print(f"  images/train/, images/val/")
    print(f"\nFine-tune from Kumar checkpoint:")
    print(f"  .\\.venv-train\\Scripts\\Activate.ps1")
    print(f"  $env:YOLOX_DATA_DIR = '{out_dir}'; $env:YOLOX_OUTPUT_DIR = 'YOLOX_outputs'")
    print(f"  python -m yolox.tools.train -f scripts/yolox_mouse_exp.py -d 1 -b 16 --fp16 -c YOLOX_outputs/yolox_mouse_nano/best_ckpt.pth")


if __name__ == "__main__":
    main()
