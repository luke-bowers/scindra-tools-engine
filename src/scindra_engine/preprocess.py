from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scindra_engine.schemas import PreprocessingConfig


@dataclass(frozen=True)
class BackgroundModel:
    image: np.ndarray


def build_background(
    frames_bgr: list[np.ndarray], config: PreprocessingConfig
) -> BackgroundModel | None:
    if config.background_model == "none" or not frames_bgr:
        return None

    gray_frames = [
        _to_grayscale(frame, config) for frame in frames_bgr if frame is not None
    ]
    if not gray_frames:
        return None

    stack = np.stack(gray_frames, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)
    return BackgroundModel(image=median)


def preprocess_frame(
    frame_bgr: np.ndarray,
    config: PreprocessingConfig,
    background: BackgroundModel | None = None,
) -> np.ndarray:
    gray = _to_grayscale(frame_bgr, config)

    if background is not None:
        diff = cv2.absdiff(gray, background.image)
        gray = diff

    if config.clahe:
        clahe = cv2.createCLAHE(
            clipLimit=float(config.clahe_clip_limit), tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

    if config.gamma is not None:
        gray = _apply_gamma(gray, config.gamma)

    if config.denoise == "gaussian":
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif config.denoise == "bilateral":
        gray = cv2.bilateralFilter(gray, 7, 50, 50)

    if config.illumination_correction == "rolling_ball":
        kernel = _rolling_ball_kernel(gray.shape)
        background_est = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        gray = cv2.subtract(gray, background_est)
    elif config.illumination_correction == "morph_open":
        kernel = _rolling_ball_kernel(gray.shape, scale=0.08)
        background_est = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        gray = cv2.subtract(gray, background_est)

    return gray


def _to_grayscale(frame_bgr: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    if frame_bgr.ndim == 2:
        gray = frame_bgr
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def _rolling_ball_kernel(shape: tuple[int, int], scale: float = 0.12) -> np.ndarray:
    height, width = shape
    size = max(3, int(min(height, width) * scale))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
