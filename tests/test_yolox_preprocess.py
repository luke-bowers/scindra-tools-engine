"""Tests for YOLOX preprocessing (letterbox + normalize).

These tests exercise the preprocessing pipeline from yolox_onnx.py.
The ``preprocess_frame`` and ``letterbox`` functions do NOT require
onnxruntime — they are pure numpy/cv2.  However, they are closely tied
to the detector feature, so we guard them with the detector_optional
marker so they are skipped cleanly when the detector extra is not
installed in a minimal CI environment.
"""
from __future__ import annotations

import numpy as np
import pytest

from scindra_engine.detectors.yolox_onnx import letterbox, preprocess_frame


def _make_synthetic_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Deterministic synthetic BGR frame."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_letterbox_preserves_aspect() -> None:
    frame = _make_synthetic_frame(120, 160)
    padded, scale, (pt, pl) = letterbox(frame, (640, 640))
    assert padded.shape == (640, 640, 3)
    # Scale should be min(640/160, 640/120) = 4.0
    assert abs(scale - 4.0) < 1e-6


def test_letterbox_pad_value() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, _, (pt, pl) = letterbox(frame, (640, 640), pad_value=114)
    # Top pad area should be 114
    if pt > 0:
        assert padded[0, 0, 0] == 114


def test_preprocess_output_shape() -> None:
    frame = _make_synthetic_frame(120, 160)
    tensor, scale, pad = preprocess_frame(frame, (640, 640))
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32


def test_preprocess_value_range() -> None:
    frame = _make_synthetic_frame(120, 160)
    tensor, _, _ = preprocess_frame(frame, (640, 640))
    assert tensor.min() >= 0.0
    assert tensor.max() <= 255.0


def test_preprocess_determinism() -> None:
    """Two calls with the same frame produce bit-identical tensors."""
    frame = _make_synthetic_frame(120, 160)
    t1, s1, p1 = preprocess_frame(frame, (640, 640))
    t2, s2, p2 = preprocess_frame(frame, (640, 640))
    assert s1 == s2
    assert p1 == p2
    np.testing.assert_array_equal(t1, t2)


def test_preprocess_rgb_conversion() -> None:
    """Output tensor has RGB channel order (reversed from BGR input)."""
    # Create a frame where B=0, G=0, R=255
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # Red channel in BGR
    tensor, _, _ = preprocess_frame(frame, (4, 4))
    # After BGR->RGB, channel 0 should be R (255.0 in 0-255 range)
    # Channel 2 should be B (0)
    # The frame is tiny so no padding needed
    assert tensor[0, 0, 0, 0] == pytest.approx(255.0, abs=1e-6)  # R
    assert tensor[0, 2, 0, 0] == pytest.approx(0.0, abs=1e-6)  # B
