"""Draw COCO bboxes on a few Kumar training images to verify conversion.

Run from repo root:
    python scripts/verify_training_labels.py

Writes out/verify_labels/ with sample images; check that the green box
tightly wraps the mouse. If boxes are wrong, the ellipse->bbox conversion
or COCO build is at fault.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Install opencv-python to run this script.")
    raise SystemExit(1)


def main() -> None:
    coco_dir = Path("datasets/kumar_mouse_coco")
    ann_path = coco_dir / "annotations" / "train.json"
    img_dir = coco_dir / "images" / "train"
    out_dir = Path("out/verify_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in data["images"]}
    anns = data["annotations"]

    # Draw first 12 (image_id 1..12 or so)
    seen = set()
    n = 0
    for ann in anns:
        if n >= 12:
            break
        img_id = ann["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        im = images[img_id]
        path = img_dir / im["file_name"]
        if not path.exists():
            continue
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        img = cv2.imread(str(path))
        if img is None:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, f"id={img_id}", (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), img)
        n += 1

    print(f"Wrote {n} samples to {out_dir}")
    print("Check that the green box wraps the mouse. If not, conversion is wrong.")


if __name__ == "__main__":
    main()
