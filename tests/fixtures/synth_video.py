from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np


def write_synth_video(
    path: Path,
    num_frames: int = 16,
    size: tuple[int, int] = (64, 48),
    fps: float = 10.0,
) -> None:
    """Write a small deterministic MP4 video for testing.

    Args:
        path: Output file path.
        num_frames: Number of frames to generate.
        size: (width, height) in pixels.
        fps: Frames per second to encode.
    """

    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    try:
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Deterministic but varying pattern across frames.
            base = np.linspace(0, 255, width, dtype=np.float32)
            x_row = np.tile(base, (height, 1))
            frame[..., 0] = np.clip(x_row + i * 5.0, 0.0, 255.0).astype(
                np.uint8
            )  # Blue
            frame[..., 1] = np.tile(
                np.linspace(0, 255, height, dtype=np.uint8).reshape(
                    height, 1
                ),
                (1, width),
            )  # Green
            frame[..., 2] = (x_row / 2.0).astype(np.uint8)  # Red
            writer.write(frame)
    finally:
        writer.release()


def make_synth_video(
    tmp_dir: Path,
    num_frames: int = 16,
    size: tuple[int, int] = (64, 48),
    fps: float = 10.0,
) -> Path:
    """Create a synthetic MP4 video under a temporary directory."""

    video_path = tmp_dir / "synth.mp4"
    write_synth_video(video_path, num_frames=num_frames, size=size, fps=fps)
    return video_path

