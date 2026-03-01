from __future__ import annotations

from pathlib import Path
from typing import Callable

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
    display_aspect_ratio: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Write a heatmap PNG showing spatial density of centroids.

    Args:
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        track_points: List of TrackPoint objects for the video.
        out_path: Path to the output PNG file.
        blur_ksize: Odd kernel size for Gaussian blur.
        display_aspect_ratio: If set (e.g. '16:9'), resize heatmap to match so proportions match the video.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if blur_ksize <= 0 or blur_ksize % 2 == 0:
        raise ValueError("blur_ksize must be a positive odd integer")

    dst_path = Path(out_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    accum = np.zeros((height, width), dtype=np.float32)

    total_points = len(track_points)
    processed = 0

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
        processed += 1
        if progress_callback is not None and total_points > 0:
            progress_callback(processed, total_points)

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

    if display_aspect_ratio:
        from scindra_engine.video_io import get_display_dimensions

        new_w, new_h = get_display_dimensions(width, height, display_aspect_ratio)
        if (new_w, new_h) != (width, height):
            colored = cv2.resize(colored, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if not cv2.imwrite(str(dst_path), colored):
        raise RuntimeError(f"Failed to write heatmap PNG to {dst_path}")

