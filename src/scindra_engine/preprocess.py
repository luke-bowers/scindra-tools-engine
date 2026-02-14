from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scindra_engine.schemas import PreprocessingConfig


@dataclass
class BackgroundModel:
    """Background model for frame preprocessing.

    For *median_n*: ``image`` holds the static median background.
    For *mog2*: ``mog2_subtractor`` wraps a pre-trained
    ``cv2.BackgroundSubtractorMOG2``.
    """

    image: np.ndarray | None = None
    image_bgr: np.ndarray | None = None  # colour median (for chroma filtering)
    mog2_subtractor: object | None = None  # cv2.BackgroundSubtractorMOG2


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

    if config.background_model == "mog2":
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.mog2_history,
            varThreshold=config.mog2_var_threshold,
            detectShadows=config.mog2_detect_shadows,
        )
        # Pre-train on the sampled frames so the model starts warm
        for g in gray_frames:
            subtractor.apply(g, learningRate=0.05)
        return BackgroundModel(mog2_subtractor=subtractor)

    # median_n
    stack = np.stack(gray_frames, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)

    # Also compute BGR median for chrominance-based shadow suppression
    bgr_frames = [f for f in frames_bgr if f is not None and f.ndim == 3]
    median_bgr: np.ndarray | None = None
    if bgr_frames:
        bgr_stack = np.stack(bgr_frames, axis=0).astype(np.float32)
        median_bgr = np.median(bgr_stack, axis=0).astype(np.uint8)

    return BackgroundModel(image=median, image_bgr=median_bgr)


def preprocess_frame(
    frame_bgr: np.ndarray,
    config: PreprocessingConfig,
    background: BackgroundModel | None = None,
) -> np.ndarray:
    gray = _to_grayscale(frame_bgr, config)

    if background is not None:
        if background.mog2_subtractor is not None:
            # MOG2: apply returns foreground mask (0/127/255)
            fg = background.mog2_subtractor.apply(gray, learningRate=0)  # type: ignore[union-attr]
            # Keep only definite foreground (value 255), suppress shadows
            _, gray = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        elif background.image is not None:
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
