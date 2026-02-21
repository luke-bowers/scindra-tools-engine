"""Convert Kumar Lab ellipse annotations to COCO-format JSON for YOLOX training.

Usage:
    python scripts/convert_kumar_to_coco.py \
        --dataset-dir path/to/OFA_Dataset \
        --out-dir path/to/coco_dataset

The script reads ellipse .txt files (angle, major, minor, cy, cx) next to
each image, computes axis-aligned bounding boxes from the ellipses, and
writes COCO-format annotation JSON files.

Directory structure produced:
    coco_dataset/
        images/
            train/
            val/
        annotations/
            train.json
            val.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def ellipse_to_bbox(
    cx: float,
    cy: float,
    major: float,
    minor: float,
    angle_deg: float,
    padding: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounding box from an ellipse.

    Args:
        cx, cy: Centre of the ellipse (pixels).
        major: Full major axis length (pixels). Halved to get semi-axis.
        minor: Full minor axis length (pixels). Halved to get semi-axis.
        angle_deg: Rotation angle in degrees (0 = downward, CCW positive).
        padding: Extra pixels (or fraction if < 1) to add to each side for a
                looser box (Kumar ellipses are tight fits, sometimes tail excluded).

    Returns:
        (x_min, y_min, width, height) in COCO format.
    """
    a = major / 2.0
    b = minor / 2.0
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half_w = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
    half_h = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
    # Optional padding (use same value on all sides; if < 1 treat as fraction of min(half_w,half_h))
    if padding > 0:
        p = padding if padding >= 1.0 else padding * min(half_w, half_h)
        half_w += p
        half_h += p
    x_min = cx - half_w
    y_min = cy - half_h
    width = 2.0 * half_w
    height = 2.0 * half_h
    return (x_min, y_min, width, height)


def parse_ellipse_file(path: Path) -> tuple[float, float, float, float, float] | None:
    """Parse a Kumar Lab ellipse annotation file.

    The Kumar Lab web page documents: Angle, Major, Minor, Y_center, X_center.
    In practice the OFA_Dataset .txt files appear to be: X_center, Y_center, Major, Minor, Angle
    (center first, in image coords with Y from top). We use the empirical order so that
    the first two values (0..width, 0..height) are center and the next two (~20--60) are
    axis lengths. Returns (cx, cy, major, minor, angle_deg) or None if unreadable.
    """
    try:
        text = path.read_text().strip()
        if not text:
            return None
        parts = text.split()
        if len(parts) < 5:
            return None
        # Empirical: file order is X, Y, Major, Minor, Angle (image coords, Y from top)
        x_center = float(parts[0])
        y_center = float(parts[1])
        major = float(parts[2])
        minor = float(parts[3])
        angle_deg = float(parts[4])
        return (x_center, y_center, major, minor, angle_deg)
    except (ValueError, OSError):
        return None


def build_coco_json(
    image_dir: Path,
    ell_dir: Path,
    split_prefix: str,
    bbox_padding: float = 0.0,
) -> tuple[dict, list[Path]]:
    """Build a COCO-format annotation dict for one split.

    Args:
        image_dir: Directory containing the images (e.g. Img/).
        ell_dir: Directory containing the ellipse .txt files (e.g. Ell/).
        split_prefix: Filename prefix filter ("Training" or "Validation").
        bbox_padding: Extra pixels added to ellipse bbox (0 = tight fit).

    Returns:
        (coco_dict, list_of_image_paths)
    """
    images_list: list[dict] = []
    annotations_list: list[dict] = []
    image_paths: list[Path] = []

    ann_id = 1
    img_id = 1

    # Gather matching images
    img_files = sorted(
        [f for f in image_dir.iterdir() if f.stem.startswith(split_prefix) and f.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.name,
    )

    for img_path in img_files:
        # Corresponding ellipse file
        ell_path = ell_dir / (img_path.stem + ".txt")
        if not ell_path.is_file():
            continue

        parsed = parse_ellipse_file(ell_path)
        if parsed is None:
            continue

        cx, cy, major, minor, angle_deg = parsed

        # Get image dimensions
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except ImportError:
            img_w, img_h = 640, 480

        # Parsed cy is already image row (Y from top).
        x_min, y_min, bbox_w, bbox_h = ellipse_to_bbox(
            cx, cy, major, minor, angle_deg, padding=bbox_padding
        )

        # Clamp to image bounds
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        bbox_w = min(bbox_w, img_w - x_min)
        bbox_h = min(bbox_h, img_h - y_min)

        if bbox_w <= 0 or bbox_h <= 0:
            continue

        images_list.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        annotations_list.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [round(x_min, 2), round(y_min, 2), round(bbox_w, 2), round(bbox_h, 2)],
            "area": round(bbox_w * bbox_h, 2),
            "iscrowd": 0,
        })

        image_paths.append(img_path)
        img_id += 1
        ann_id += 1

    coco = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": [
            {"id": 1, "name": "mouse", "supercategory": "animal"},
        ],
    }

    return coco, image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kumar Lab dataset to COCO format.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to extracted OFA_Dataset (contains Img/ and Ell/ subdirs).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for COCO dataset.")
    parser.add_argument("--bbox-padding", type=float, default=0.0, metavar="PIX", help="Add PIX pixels to each side of ellipse bbox (default: 0).")
    parser.add_argument("--copy-images", action="store_true", default=True, help="Copy images to output directory (default: True).")
    parser.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying (saves disk space).")
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    out_dir: Path = args.out_dir

    # Kumar Lab datasets use Ref/ for images and Ell/ for annotations.
    # Also support Img/ (alternative naming) or flat layout.
    img_dir: Path | None = None
    ell_dir: Path | None = None

    for img_candidate in ("Ref", "Img"):
        candidate = dataset_dir / img_candidate
        if candidate.is_dir():
            img_dir = candidate
            break

    if img_dir is None:
        # Try flat layout (images and .txt files in the same directory)
        if any(dataset_dir.glob("Training_*.png")) or any(dataset_dir.glob("Training_*.jpg")):
            img_dir = dataset_dir
        else:
            print(f"ERROR: Cannot find images in {dataset_dir}.")
            print(f"  Expected one of:")
            print(f"    {dataset_dir / 'Ref'}/  (Kumar Lab standard)")
            print(f"    {dataset_dir / 'Img'}/")
            print(f"    Training_*.png files directly in {dataset_dir}")
            raise SystemExit(1)

    ell_dir = dataset_dir / "Ell"
    if not ell_dir.is_dir():
        # Ellipse files might be alongside images in flat layout
        ell_dir = img_dir

    print(f"  Image directory: {img_dir}")
    print(f"  Ellipse directory: {ell_dir}")

    bbox_padding = getattr(args, "bbox_padding", 0.0)
    # Build COCO annotations
    print("Processing training split...")
    train_coco, train_imgs = build_coco_json(img_dir, ell_dir, "Training", bbox_padding)
    print(f"  Found {len(train_imgs)} training images with annotations.")

    print("Processing validation split...")
    val_coco, val_imgs = build_coco_json(img_dir, ell_dir, "Validation", bbox_padding)
    print(f"  Found {len(val_imgs)} validation images with annotations.")

    if len(train_imgs) == 0:
        print("WARNING: No training images found. Check --dataset-dir points to the extracted dataset.")

    # Write output
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    (ann_dir / "train.json").write_text(json.dumps(train_coco, indent=2))
    print(f"  Wrote {ann_dir / 'train.json'}")

    (ann_dir / "val.json").write_text(json.dumps(val_coco, indent=2))
    print(f"  Wrote {ann_dir / 'val.json'}")

    # Copy/symlink images
    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = out_dir / "images" / split_name
        img_out.mkdir(parents=True, exist_ok=True)

        for src in split_imgs:
            dst = img_out / src.name
            if dst.exists():
                continue
            if args.symlink_images:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        print(f"  {'Linked' if args.symlink_images else 'Copied'} {len(split_imgs)} images to {img_out}")

    print(f"\nDone! COCO dataset written to: {out_dir}")
    print(f"\nNext step: use this dataset to train YOLOX-Nano.")
    print(f"  Train JSON: {ann_dir / 'train.json'}")
    print(f"  Val JSON:   {ann_dir / 'val.json'}")


if __name__ == "__main__":
    main()
