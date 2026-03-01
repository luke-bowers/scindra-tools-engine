from PIL import Image
import numpy as np
import json
import sys
from pathlib import Path

def find_white_bbox(img, thresh=200):
    gray = np.array(img.convert("L"))
    mask = gray >= thresh
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (int(x1), int(y1), int(x2), int(y2))

def expand_bbox(bbox, img_w, img_h, pad=8):
    x1,y1,x2,y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w-1, x2 + pad)
    y2 = min(img_h-1, y2 + pad)
    return (x1,y1,x2,y2)

def main():
    src = Path("out/debug_arena/arena_crop_static.png")
    out = Path("out/debug_arena/arena_crop_cropped_from_ai.png")
    if not src.exists():
        print(json.dumps({"error":"source not found"}))
        return
    img = Image.open(src)
    w,h = img.size
    # try a couple thresholds if needed
    bbox = find_white_bbox(img, thresh=210)
    if bbox is None:
        bbox = find_white_bbox(img, thresh=180)
    if bbox is None:
        print(json.dumps({"error":"no white region found"}))
        return
    bbox = expand_bbox(bbox, w, h, pad=6)
    x1,y1,x2,y2 = bbox
    crop = img.crop((x1,y1,x2+1,y2+1))
    crop.save(out)
    result = {
        "path": str(out.as_posix()),
        "coords": {
            "x": x1,
            "y": y1,
            "width": x2 - x1 + 1,
            "height": y2 - y1 + 1
        },
        "bbox": [x1,y1,x2,y2],
        "image_size": {"width": w, "height": h}
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
