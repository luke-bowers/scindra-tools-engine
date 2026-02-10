from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
) -> None:
    """Write an overlay video with centroid markers and optional trail.

    Args:
        video_path: Path to the source video.
        track_points: List of TrackPoint objects for the video.
        out_path: Path to the output MP4 file.
        draw_radius: Circle radius in pixels for the centroid marker.
        draw_thickness: Thickness passed to cv2.circle (-1 for filled).
        draw_trail: Whether to draw a short trail of recent centroids.
        trail_length: Number of recent centroids to keep in the trail.
        progress_every_n: Emit a PROGRESS line every N frames.
    """
    if trail_length <= 0:
        trail_length = 1
    if progress_every_n <= 0:
        progress_every_n = 1

    src_path = Path(video_path)
    dst_path = Path(out_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Map frame index to TrackPoint for quick lookup.
    points_by_frame = _build_points_index(track_points)

    with VideoReader(src_path) as reader:
        fps = reader.fps
        width = reader.width
        height = reader.height
        total_frames = reader.frame_count

        fourcc = int(getattr(cv2, "VideoWriter_fourcc")(*"mp4v"))
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {dst_path}")

        trail: list[tuple[int, int]] = []
        last_progress_printed = 0

        try:
            for frame_idx, frame_bgr in reader.iter_frames():
                point = points_by_frame.get(frame_idx)
                if point is not None and point.x is not None and point.y is not None:
                    cx, cy = _clamp_point(point.x, point.y, width, height)

                    # Draw the current centroid as a filled circle (red in BGR).
                    cv2.circle(
                        frame_bgr,
                        (cx, cy),
                        draw_radius,
                        (0, 0, 255),
                        draw_thickness,
                    )

                    trail.append((cx, cy))
                    if len(trail) > trail_length:
                        # Keep only the most recent N points.
                        trail = trail[-trail_length:]

                if draw_trail and len(trail) >= 2:
                    _draw_trail(frame_bgr, trail)

                writer.write(frame_bgr)

                done = frame_idx + 1
                if (
                    progress_every_n > 0
                    and done % progress_every_n == 0
                    and done != last_progress_printed
                ):
                    print(f"PROGRESS {done}/{total_frames} overlay")
                    last_progress_printed = done

            if total_frames > 0 and last_progress_printed != total_frames:
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


def _draw_trail(frame_bgr: np.ndarray, trail: list[tuple[int, int]]) -> None:
    """Draw a simple line trail connecting recent centroid positions."""
    # Use a fixed yellow color in BGR for the trail.
    color = (0, 255, 255)
    thickness = 2
    for i in range(1, len(trail)):
        p0 = trail[i - 1]
        p1 = trail[i]
        cv2.line(frame_bgr, p0, p1, color, thickness)

