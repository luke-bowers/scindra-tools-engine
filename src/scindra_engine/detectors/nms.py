"""Deterministic pure-numpy Non-Maximum Suppression."""
from __future__ import annotations

import numpy as np


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
) -> np.ndarray:
    """Greedy NMS with deterministic tie-breaking.

    Args:
        boxes_xyxy: ``(N, 4)`` array of ``[x1, y1, x2, y2]`` boxes.
        scores: ``(N,)`` array of confidence scores.
        iou_thresh: IoU threshold for suppression.

    Returns:
        1-D int array of kept indices in descending score order.
    """
    if len(scores) == 0:
        return np.empty(0, dtype=np.intp)

    # Stable descending sort (mergesort is stable in numpy)
    order = np.argsort(-scores, kind="mergesort")

    x1 = boxes_xyxy[:, 0].astype(np.float64)
    y1 = boxes_xyxy[:, 1].astype(np.float64)
    x2 = boxes_xyxy[:, 2].astype(np.float64)
    y2 = boxes_xyxy[:, 3].astype(np.float64)
    areas = (x2 - x1) * (y2 - y1)

    keep: list[int] = []
    suppressed = np.zeros(len(scores), dtype=np.bool_)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))

        # Compute IoU of remaining boxes with the kept box
        xx1 = np.maximum(x1[idx], x1)
        yy1 = np.maximum(y1[idx], y1)
        xx2 = np.minimum(x2[idx], x2)
        yy2 = np.minimum(y2[idx], y2)

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[idx] + areas - inter_area
        iou = np.where(union_area > 0, inter_area / union_area, 0.0)

        suppressed |= iou > iou_thresh

    return np.array(keep, dtype=np.intp)
