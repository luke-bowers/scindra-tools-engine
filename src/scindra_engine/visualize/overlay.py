from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from scindra_engine.tracking import TrackPoint
from scindra_engine.video_io import VideoReader


def write_overlay_video(
    video_path: str,
    track_points: list[TrackPoint],
    out_path: str,
    *,
    draw_radius: int = 6,
    draw_thickness: int = -1,
    draw_trail: bool = True,
    trail_length: int = 30,
    progress_every_n: int = 30,
    scale: float = 1.0,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Write an overlay video with centroid markers and optional trail.

    Args:
        video_path: Path to the source video.
        track_points: List of TrackPoint objects (coordinates in cropped space when crop_xyxy is set).
        out_path: Path to the output MP4 file.
        draw_radius: Circle radius in pixels for the centroid marker.
        draw_thickness: Thickness passed to cv2.circle (-1 for filled).
        draw_trail: Whether to draw a short trail of recent centroids.
        trail_length: Number of recent centroids to keep in the trail.
        progress_every_n: Emit a PROGRESS line every N frames.
        crop_xyxy: When set, crop each frame to (x1, y1, x2, y2) and write at cropped size.
    """
    if trail_length <= 0:
        trail_length = 1
    if progress_every_n <= 0:
        progress_every_n = 1
    if scale <= 0.0:
        raise ValueError("scale must be a positive floating-point value")

    src_path = Path(video_path)
    dst_path = Path(out_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Map frame index to TrackPoint for quick lookup.
    points_by_frame = _build_points_index(track_points)

    with VideoReader(src_path) as reader:
        fps = reader.fps
        total_frames = reader.frame_count
        # Use actual decoded frame dimensions (handles rotation metadata mismatch)
        full_width, full_height = reader.get_actual_dimensions()

        if crop_xyxy is not None:
            x1, y1, x2, y2 = crop_xyxy
            width = x2 - x1
            height = y2 - y1
        else:
            width = full_width
            height = full_height

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions for overlay: {width}x{height}")

        out_width = width
        out_height = height
        if scale != 1.0:
            out_width = max(1, int(round(width * scale)))
            out_height = max(1, int(round(height * scale)))

        fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*"mp4v"))
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (out_width, out_height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {dst_path}")

        trail: list[tuple[int, int]] = []
        last_progress_printed = 0

        try:
            for frame_idx, frame_bgr in reader.iter_frames():
                if crop_xyxy is not None:
                    x1, y1, x2, y2 = crop_xyxy
                    frame_bgr = frame_bgr[y1:y2, x1:x2]
                # Resize frame to output resolution first so drawing uses correct coordinate space
                if frame_bgr.shape[1] != out_width or frame_bgr.shape[0] != out_height:
                    frame_bgr = cv2.resize(frame_bgr, (out_width, out_height), interpolation=cv2.INTER_AREA)

                point = points_by_frame.get(frame_idx)
                if point is not None and point.x is not None and point.y is not None:
                    # Centroid is in source (width x height) space; scale to output space
                    sx = float(point.x) * (float(out_width) / float(width))
                    sy = float(point.y) * (float(out_height) / float(height))
                    cx, cy = _clamp_point(sx, sy, out_width, out_height)

                    # Draw the current centroid as a filled circle (red in BGR).
                    # Scale radius with output but keep a visible minimum (so dot doesn't disappear when scale < 1)
                    effective_radius = max(3, int(round(draw_radius * (out_width / max(width, 1)))))
                    cv2.circle(
                        frame_bgr,
                        (cx, cy),
                        effective_radius,
                        (0, 0, 255),
                        draw_thickness,
                    )

                    trail.append((cx, cy))
                    if len(trail) > trail_length:
                        # Keep only the most recent N points.
                        trail = trail[-trail_length:]

                if draw_trail and len(trail) >= 2:
                    # Scale trail thickness with output so it doesn't dominate at low resolution
                    trail_thickness = max(1, int(round(2 * (out_width / max(width, 1)))))
                    _draw_trail(frame_bgr, trail, thickness=trail_thickness)

                writer.write(frame_bgr)

                done = frame_idx + 1
                if progress_callback is not None and total_frames > 0:
                    progress_callback(done, total_frames)
                elif (
                    progress_every_n > 0
                    and done % progress_every_n == 0
                    and done != last_progress_printed
                ):
                    print(f"PROGRESS {done}/{total_frames} overlay")
                    last_progress_printed = done

            if total_frames > 0 and progress_callback is not None:
                progress_callback(total_frames, total_frames)
            elif total_frames > 0 and last_progress_printed != total_frames:
                print(f"PROGRESS {total_frames}/{total_frames} overlay")
        finally:
            writer.release()


def _build_points_index(
    track_points: Iterable[TrackPoint],
) -> dict[int, TrackPoint]:
    """Build a mapping from frame index to TrackPoint.

    Later points for the same frame overwrite earlier ones, but the current
    pipeline generates at most one TrackPoint per frame.
    """
    result: dict[int, TrackPoint] = {}
    for point in track_points:
        result[point.frame_idx] = point
    return result


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    cx = int(round(x))
    cy = int(round(y))
    if width > 0:
        cx = min(max(cx, 0), width - 1)
    if height > 0:
        cy = min(max(cy, 0), height - 1)
    return cx, cy


def _draw_trail(
    frame_bgr: np.ndarray,
    trail: list[tuple[int, int]],
    *,
    thickness: int = 2,
) -> None:
    """Draw a simple line trail connecting recent centroid positions."""
    color = (0, 255, 255)  # yellow in BGR
    for i in range(1, len(trail)):
        p0 = trail[i - 1]
        p1 = trail[i]
        cv2.line(frame_bgr, p0, p1, color, thickness)

