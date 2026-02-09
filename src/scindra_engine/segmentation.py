from __future__ import annotations

import cv2
import numpy as np

from scindra_engine.schemas import MorphologyConfig, SegmentationConfig


def segment_frame(
    frame_gray: np.ndarray,
    segmentation: SegmentationConfig,
    morphology: MorphologyConfig,
) -> np.ndarray:
    mask = _threshold(frame_gray, segmentation)
    mask = _apply_morphology(mask, morphology)
    return mask


def _threshold(
    frame_gray: np.ndarray, segmentation: SegmentationConfig
) -> np.ndarray:
    if segmentation.threshold == "manual":
        if segmentation.manual_value is None:
            raise ValueError("manual_value is required when threshold is 'manual'")
        _, mask = cv2.threshold(
            frame_gray, int(segmentation.manual_value), 255, cv2.THRESH_BINARY
        )
    elif segmentation.threshold == "adaptive":
        mask = cv2.adaptiveThreshold(
            frame_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(segmentation.adaptive_block_size),
            int(segmentation.adaptive_C),
        )
    else:
        _, mask = cv2.threshold(
            frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if segmentation.invert:
        mask = cv2.bitwise_not(mask)
    return mask


def _apply_morphology(mask: np.ndarray, morphology: MorphologyConfig) -> np.ndarray:
    result = mask
    if morphology.open_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology.open_ksize, morphology.open_ksize),
        )
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    if morphology.close_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology.close_ksize, morphology.close_ksize),
        )
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    if morphology.erode_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.erode(result, kernel, iterations=int(morphology.erode_iters))

    if morphology.dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.dilate(result, kernel, iterations=int(morphology.dilate_iters))

    return result
