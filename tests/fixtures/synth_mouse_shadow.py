from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np


def write_synth_mouse_shadow_video(
    path: Path,
    num_frames: int = 30,
    size: tuple[int, int] = (160, 120),
    fps: float = 15.0,
) -> None:
    """Write a synthetic video with a moving mouse and a shadow blob."""
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    background_color = 200
    mouse_color = 40
    shadow_color = 110

    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), background_color, dtype=np.uint8)
            cx = int(20 + (width - 40) * (i / max(1, num_frames - 1)))
            cy = int(height / 2 + 10 * np.sin(i / 3.0))

            cv2.ellipse(
                frame,
                (cx, cy),
                (10, 6),
                0,
                0,
                360,
                (mouse_color, mouse_color, mouse_color),
                -1,
            )

            shadow_offset = 18
            cv2.ellipse(
                frame,
                (cx + shadow_offset, cy + 5),
                (20, 6),
                -10,
                0,
                360,
                (shadow_color, shadow_color, shadow_color),
                -1,
            )

            if i % 10 == 0:
                cv2.circle(
                    frame,
                    (cx + 30, cy - 12),
                    5,
                    (mouse_color, mouse_color, mouse_color),
                    -1,
                )

            writer.write(frame)
    finally:
        writer.release()


def make_synth_mouse_shadow_video(
    out_dir: Path,
    num_frames: int = 30,
    size: tuple[int, int] = (160, 120),
    fps: float = 15.0,
) -> Path:
    video_path = out_dir / "synth_mouse_shadow.mp4"
    write_synth_mouse_shadow_video(
        video_path, num_frames=num_frames, size=size, fps=fps
    )
    return video_path
