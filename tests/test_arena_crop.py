"""Tests for arena_crop module."""

from __future__ import annotations

import numpy as np

from scindra_engine.arena_crop import (
    crop_frame,
    detect_arena_crop_xyxy,
)


def test_crop_frame() -> None:
    """crop_frame returns the correct region."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[10:50, 20:120, :] = 255
    cropped = crop_frame(frame, (20, 10, 120, 50))
    assert cropped.shape == (40, 100, 3)
    assert np.all(cropped == 255)


def test_detect_arena_crop_xyxy_white_rect() -> None:
    """Detect bounding box of a white rectangle on black."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[25:75, 50:150, :] = 255
    box, _ = detect_arena_crop_xyxy(img, margin_px=0, min_area_ratio=0.01)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 <= 55
    assert y1 <= 30
    assert x2 >= 145
    assert y2 >= 70


def test_detect_arena_crop_xyxy_margin() -> None:
    """Margin expands the box (clamped to image)."""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[10:50, 10:70, :] = 255
    box, _ = detect_arena_crop_xyxy(img, margin_px=5, min_area_ratio=0.01)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 <= 10
    assert y1 <= 10
    assert x2 >= 70
    assert y2 >= 50


def test_detect_arena_crop_xyxy_debug_callback() -> None:
    """Debug callback is invoked at each pipeline step with expected data."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[25:75, 50:150, :] = 255
    steps: list[str] = []
    payloads: list[dict] = []

    def collect(step: str, data: dict) -> None:
        steps.append(step)
        payloads.append({k: v for k, v in data.items() if k != "image"})

    box, _ = detect_arena_crop_xyxy(img, min_area_ratio=0.01, debug_callback=collect)
    assert box is not None
    assert "edges" in steps
    assert steps[-1] == "contour"
    assert len(payloads) >= 2
    assert "morph_close_ksize" in payloads[0]
    assert "num_candidates" in payloads[-1]
    assert "chosen_bbox" in payloads[-1]
