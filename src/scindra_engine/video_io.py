from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np


class VideoIOError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic video metadata."""

    fps: float
    frame_count: int
    width: int
    height: int


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with deterministic iteration."""

    _FRAME_POS_PROP: Final[int] = cv2.CAP_PROP_POS_FRAMES

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cap: cv2.VideoCapture = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise VideoIOError(f"Could not open video: {self._path}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self._metadata = VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    @property
    def path(self) -> Path:
        """Path to the underlying video file."""

        return self._path

    @property
    def fps(self) -> float:
        """Frames per second reported by the backend."""

        return self._metadata.fps

    @property
    def frame_count(self) -> int:
        """Total number of frames reported by the backend."""

        return self._metadata.frame_count

    @property
    def width(self) -> int:
        """Frame width in pixels."""

        return self._metadata.width

    @property
    def height(self) -> int:
        """Frame height in pixels."""

        return self._metadata.height

    def close(self) -> None:
        """Release the underlying VideoCapture."""

        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.close()

    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames as (index, frame_bgr) pairs.

        Frames are returned as NumPy arrays in BGR color order, matching
        OpenCV conventions.
        """

        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if step <= 0:
            raise ValueError("step must be a positive integer")

        total = self.frame_count
        if total <= 0:
            return
        if end_frame is None or end_frame > total:
            end = total
        else:
            end = end_frame

        if start_frame >= end:
            return

        # Seek to the starting frame; some backends may return False even
        # when seeking is best-effort, so we do not treat the return value
        # as fatal and rely on subsequent reads to detect errors.
        self._cap.set(self._FRAME_POS_PROP, float(start_frame))

        current = start_frame
        while current < end:
            success, frame = self._cap.read()
            if not success:
                raise VideoIOError(
                    f"Failed to read frame {current} from {self._path}"
                )

            if (current - start_frame) % step == 0:
                yield current, frame

            current += 1


class FrameSampler:
    """Evenly sample frames from a VideoReader for background modeling."""

    def __init__(self, reader: VideoReader) -> None:
        self._reader = reader

    def sample(self, num_frames: int) -> list[tuple[int, np.ndarray]]:
        """Return evenly spaced (frame_index, frame_bgr) pairs.

        The sampling is deterministic given the reader's reported frame_count.
        """

        if num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")

        total = self._reader.frame_count
        if total <= 0:
            return []
        if num_frames >= total:
            return list(self._reader.iter_frames())

        indices = self._compute_indices(total, num_frames)
        result: list[tuple[int, np.ndarray]] = []

        target_iter = iter(indices)
        try:
            next_target = next(target_iter)
        except StopIteration:
            return []

        for idx, frame in self._reader.iter_frames():
            while idx > next_target:
                try:
                    next_target = next(target_iter)
                except StopIteration:
                    return result

            if idx == next_target:
                result.append((idx, frame))
                try:
                    next_target = next(target_iter)
                except StopIteration:
                    return result

        return result

    @staticmethod
    def _compute_indices(total: int, num_frames: int) -> list[int]:
        if num_frames == 1:
            return [total // 2]
        # Evenly spaced across [0, total-1]
        return [
            round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
        ]

