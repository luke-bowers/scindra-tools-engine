"""Render debug frames showing centroid blobs color-coded by status (selected, plausible, excluded)."""

from __future__ import annotations

import cv2
import numpy as np

from scindra_engine.tracking import TrackFrameDebug


def render_debug_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    debug_info: TrackFrameDebug,
) -> np.ndarray:
    """Draw blob overlays on the frame: green=selected, yellow=plausible, red=excluded (with reason).

    All drawing is at processing resolution (same as mask and frame). Returns a new BGR image
    with a faint mask overlay and color-coded blob rectangles plus a legend.
    """
    out = frame_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Optional: faint mask overlay (cyan tint where mask is white)
    if mask is not None and mask.size > 0:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 1.0, mask_3ch, 0.15, 0)

    # Build sets for quick lookup: selected, plausible (without selected), excluded with reason
    selected = debug_info.selected
    plausible_set = {id(c) for c in debug_info.plausible}
    excluded_by_candidate = {id(c): reason for c, reason in debug_info.excluded}

    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4

    for candidate in debug_info.all_candidates:
        x, y, w, h = candidate.bbox
        cx, cy = int(round(candidate.centroid[0])), int(round(candidate.centroid[1]))
        if selected is not None and id(candidate) == id(selected):
            color = (0, 255, 0)  # BGR green
            label = "sel"
        elif id(candidate) in plausible_set:
            color = (0, 255, 255)  # BGR yellow
            label = "plaus"
        else:
            reason = excluded_by_candidate.get(id(candidate), "?")
            color = (0, 0, 255)  # BGR red
            label = reason
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(out, (cx, cy), 3, color, -1)
        cv2.putText(out, label, (x, max(0, y - 2)), font, font_scale, color, 1, cv2.LINE_AA)

    # Legend
    legend_lines = [
        "green=selected  yellow=plausible  red=excluded",
        "excluded: area_low | area_high | spatial",
    ]
    y_off = 20
    for line in legend_lines:
        cv2.putText(out, line, (8, y_off), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, line, (10, y_off), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y_off += 18
    return out
