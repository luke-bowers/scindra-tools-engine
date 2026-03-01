from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Callable

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

    def get_actual_dimensions(self) -> tuple[int, int]:
        """Return (width, height) from the first decoded frame.

        Use this instead of .width/.height when the file may have rotation or
        display metadata that changes decoded dimensions (e.g. CAP_PROP reports
        1920x1080 but decoded frames are 1080x1920). Restores read position to 0.
        """
        self._cap.set(self._FRAME_POS_PROP, 0.0)
        success, frame = self._cap.read()
        self._cap.set(self._FRAME_POS_PROP, 0.0)
        if not success or frame is None:
            return (self._metadata.width, self._metadata.height)
        height, width = frame.shape[:2]
        return (int(width), int(height))

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

    def sample(
        self,
        num_frames: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Return evenly spaced (frame_index, frame_bgr) pairs.

        The sampling is deterministic given the reader's reported frame_count.
        """

        if num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")

        total = self._reader.frame_count
        if total <= 0:
            return []
        effective_total = min(num_frames, total)
        indices = (
            self._compute_indices(total, num_frames)
            if num_frames < total
            else list(range(total))
        )
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
                if progress_callback is not None and effective_total > 0:
                    progress_callback(len(result), effective_total)
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


def require_ffmpeg_available() -> None:
    """Ensure ffmpeg and ffprobe are on PATH. Raise VideoIOError if either is missing.

    The engine requires these for overlay output (setting display aspect ratio so
    the overlay matches the input video). Install ffmpeg and ensure it is on PATH:
    - https://ffmpeg.org/download.html
    - Windows: choco install ffmpeg; or add ffmpeg bin to PATH
    """
    missing: list[str] = []
    for cmd in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [cmd, "-version"],
                capture_output=True,
                timeout=5,
            )
        except FileNotFoundError:
            missing.append(cmd)
        except subprocess.TimeoutExpired:
            pass  # found but slow; treat as available
    if missing:
        raise VideoIOError(
            "Overlay output requires ffmpeg and ffprobe on PATH. "
            "Missing: {}. Install ffmpeg from https://ffmpeg.org/download.html".format(
                ", ".join(missing)
            )
        )


def get_display_dimensions(width: int, height: int, dar: str | None) -> tuple[int, int]:
    """Return (new_width, new_height) so that new_w/new_h matches DAR (avoids distortion).

    When video has non-square pixels, decoded frame size (width, height) may not match
    display aspect ratio. Use this to get dimensions for resizing so PNG/video output
    displays with correct proportions. Returns (width, height) unchanged if dar is None
    or invalid.
    """
    if not dar or width <= 0 or height <= 0:
        return (width, height)
    try:
        parts = dar.strip().split(":")
        if len(parts) != 2:
            return (width, height)
        w_ratio = float(parts[0])
        h_ratio = float(parts[1])
    except (ValueError, AttributeError):
        return (width, height)
    if w_ratio <= 0 or h_ratio <= 0:
        return (width, height)
    target_ratio = w_ratio / h_ratio
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 0.01:
        return (width, height)
    if current_ratio > target_ratio:
        new_w = int(round(height * target_ratio))
        return (new_w, height)
    new_h = int(round(width / target_ratio))
    return (width, new_h)


def resize_to_display_aspect(frame: np.ndarray, dar: str | None) -> np.ndarray:
    """Resize frame so width/height matches display aspect ratio (prevents stretched PNGs).

    Use when writing frame buffers to PNG so they match the video's intended display
    proportions. Returns frame unchanged if dar is None or invalid.
    """
    if not dar or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = get_display_dimensions(w, h, dar)
    if (new_w, new_h) == (w, h):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_to_display_aspect(
    frame: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
    dar: str | None,
) -> np.ndarray:
    """Crop the frame to crop_xyxy and return an image with correct display aspect ratio.

    When the video has non-square pixels, cropping raw frames yields distorted crops.
    This resizes the full frame to DAR, maps crop_xyxy into the resized image, then
    crops there so the written PNG displays with correct proportions.

    Args:
        frame: Full BGR frame at decoded resolution.
        crop_xyxy: (x1, y1, x2, y2) in original frame coordinates.
        dar: Display aspect ratio (e.g. '16:9') or None to crop without resizing.

    Returns:
        Cropped region with correct aspect ratio (no stretch).
    """
    if not dar or frame.size == 0:
        x1, y1, x2, y2 = crop_xyxy
        return frame[y1:y2, x1:x2].copy()
    h, w = frame.shape[:2]
    resized = resize_to_display_aspect(frame, dar)
    rw = resized.shape[1]
    rh = resized.shape[0]
    scale_x = rw / w
    scale_y = rh / h
    x1, y1, x2, y2 = crop_xyxy
    x1_s = int(round(x1 * scale_x))
    y1_s = int(round(y1 * scale_y))
    x2_s = int(round(x2 * scale_x))
    y2_s = int(round(y2 * scale_y))
    x1_s = max(0, min(x1_s, rw - 1))
    y1_s = max(0, min(y1_s, rh - 1))
    x2_s = max(x1_s + 1, min(x2_s, rw))
    y2_s = max(y1_s + 1, min(y2_s, rh))
    return resized[y1_s:y2_s, x1_s:x2_s].copy()


def draw_crop_box_for_display(
    frame: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
    dar: str | None,
    label: str = "Arena crop (detected)",
) -> np.ndarray:
    """Draw the crop box on the frame for display. When dar is set, draws in DAR space. If the crop is square in source pixels, draws a square in output space so it appears square (avoids stretch when sx != sy)."""
    x1, y1, x2, y2 = crop_xyxy
    if dar and frame.size > 0:
        h, w = frame.shape[:2]
        out = resize_to_display_aspect(frame.copy(), dar)
        rw, rh = out.shape[1], out.shape[0]
        sx, sy = rw / w, rh / h
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w == crop_h:
            # Crop is square in source: draw a square in output so it looks square
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            side_out = min(crop_w * sx, crop_h * sy)
            x1_d = int(round(cx * sx - side_out / 2))
            y1_d = int(round(cy * sy - side_out / 2))
            x2_d = int(round(cx * sx + side_out / 2))
            y2_d = int(round(cy * sy + side_out / 2))
        else:
            x1_d = int(round(x1 * sx))
            y1_d = int(round(y1 * sy))
            x2_d = int(round(x2 * sx))
            y2_d = int(round(y2 * sy))
        cv2.rectangle(out, (x1_d, y1_d), (x2_d, y2_d), (0, 255, 0), 2)
        cv2.putText(out, label, (x1_d, max(20, y1_d - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return out
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def ensure_square_image(img: np.ndarray) -> np.ndarray:
    """Return a centered square crop of img (side = min(width, height)). No-op if already square."""
    h, w = img.shape[:2]
    if w == h:
        return img
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side].copy()


def get_video_display_aspect_ratio(video_path: str | Path) -> str | None:
    """Return the display aspect ratio (e.g. '9:16') from the first video stream via ffprobe.

    Returns None if ffprobe is unavailable, the stream has no DAR, or on error.
    """
    path = Path(video_path)
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=display_aspect_ratio",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        dar = result.stdout.strip().strip('"')
        return dar if dar and ":" in dar else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def fix_video_display_aspect_ratio(
    video_path: Path,
    reference_path: Path,
) -> bool:
    """Set the display aspect ratio of *video_path* to match *reference_path* using ffmpeg.

    Uses stream copy (no re-encode). Returns True if the fix was applied, False if
    skipped (ffmpeg unavailable, no DAR from reference, or already matching).
    """
    dar = get_video_display_aspect_ratio(reference_path)
    if not dar:
        return False
    try:
        # ffmpeg cannot overwrite in place; write to temp then replace
        temp_path = video_path.with_suffix(".tmp" + video_path.suffix)
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-c",
                "copy",
                "-aspect",
                dar,
                str(temp_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            if temp_path.exists():
                temp_path.unlink()
            return False
        temp_path.replace(video_path)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

