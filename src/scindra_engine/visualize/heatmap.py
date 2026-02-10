from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scindra_engine.tracking import TrackPoint


def write_heatmap_png(
    width: int,
    height: int,
    track_points: list[TrackPoint],
    out_path: str,
    *,
    blur_ksize: int = 31,
) -> None:
    """Write a heatmap PNG showing spatial density of centroids.

    Args:
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        track_points: List of TrackPoint objects for the video.
        out_path: Path to the output PNG file.
        blur_ksize: Odd kernel size for Gaussian blur.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if blur_ksize <= 0 or blur_ksize % 2 == 0:
        raise ValueError("blur_ksize must be a positive odd integer")

    dst_path = Path(out_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    accum = np.zeros((height, width), dtype=np.float32)

    for point in track_points:
        if point.x is None or point.y is None:
            continue
        cx = int(round(point.x))
        cy = int(round(point.y))
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            # Clamp to bounds instead of skipping entirely.
            cx = min(max(cx, 0), width - 1)
            cy = min(max(cy, 0), height - 1)
        accum[cy, cx] += 1.0

    blurred = cv2.GaussianBlur(accum, (blur_ksize, blur_ksize), 0)

    if float(blurred.max()) <= 0.0:
        heatmap_uint8 = np.zeros_like(blurred, dtype=np.uint8)
    else:
        # Normalize into the same array to satisfy type checkers.
        heatmap_norm = cv2.normalize(
            blurred,
            blurred,
            alpha=0.0,
            beta=255.0,
            norm_type=cv2.NORM_MINMAX,
        )
        heatmap_uint8 = heatmap_norm.astype(np.uint8)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if not cv2.imwrite(str(dst_path), colored):
        raise RuntimeError(f"Failed to write heatmap PNG to {dst_path}")

