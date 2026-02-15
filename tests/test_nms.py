"""Tests for deterministic NMS implementation."""
from __future__ import annotations

import numpy as np

from scindra_engine.detectors.nms import nms


def test_nms_empty_input() -> None:
    boxes = np.empty((0, 4), dtype=np.float64)
    scores = np.empty(0, dtype=np.float64)
    keep = nms(boxes, scores, 0.5)
    assert len(keep) == 0


def test_nms_single_box() -> None:
    boxes = np.array([[10, 20, 50, 60]], dtype=np.float64)
    scores = np.array([0.9], dtype=np.float64)
    keep = nms(boxes, scores, 0.5)
    assert list(keep) == [0]


def test_nms_no_overlap() -> None:
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [100, 100, 110, 110],
            [200, 200, 210, 210],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float64)
    keep = nms(boxes, scores, 0.5)
    # All kept, ordered by descending score
    assert list(keep) == [1, 0, 2]


def test_nms_full_overlap_keeps_best() -> None:
    boxes = np.array(
        [
            [10, 10, 50, 50],
            [10, 10, 50, 50],
            [10, 10, 50, 50],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.5, 0.9, 0.3], dtype=np.float64)
    keep = nms(boxes, scores, 0.5)
    assert list(keep) == [1]  # only the highest-score survives


def test_nms_partial_overlap() -> None:
    # Two boxes with ~50% overlap and one far away
    boxes = np.array(
        [
            [0, 0, 20, 20],   # area=400
            [10, 10, 30, 30],  # area=400, overlaps with box 0
            [100, 100, 120, 120],  # no overlap
        ],
        dtype=np.float64,
    )
    scores = np.array([0.8, 0.85, 0.7], dtype=np.float64)
    # Overlap between box 0 and 1: intersection = 10*10 = 100,
    # union = 400+400-100 = 700, IoU = 100/700 ≈ 0.143 < 0.5
    keep = nms(boxes, scores, 0.5)
    # All kept since IoU < threshold
    assert 1 in keep
    assert 0 in keep
    assert 2 in keep


def test_nms_suppresses_high_iou() -> None:
    boxes = np.array(
        [
            [0, 0, 20, 20],   # area = 400
            [2, 2, 22, 22],   # area = 400, high overlap with box 0
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8], dtype=np.float64)
    # Intersection: 18*18 = 324, Union: 400+400-324 = 476, IoU ≈ 0.68
    keep = nms(boxes, scores, 0.5)
    assert list(keep) == [0]  # box 1 suppressed


def test_nms_deterministic_tie_breaking() -> None:
    """When scores are equal, original order is preserved (stable sort)."""
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [100, 100, 110, 110],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.5, 0.5], dtype=np.float64)
    keep1 = nms(boxes, scores, 0.5)
    keep2 = nms(boxes, scores, 0.5)
    assert list(keep1) == list(keep2)
    # Both should be kept (no overlap), order is stable
    assert list(keep1) == [0, 1]
