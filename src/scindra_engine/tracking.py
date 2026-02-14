from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot

import cv2
import numpy as np

from scindra_engine.schemas import TrackingConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackPoint:
    frame_idx: int
    x: float | None
    y: float | None
    area: float | None
    confidence: float
    flags: list[str]


@dataclass(frozen=True)
class Candidate:
    label: int
    centroid: tuple[float, float]
    area: int
    bbox: tuple[int, int, int, int]
    solidity: float
    aspect_ratio: float
    mean_intensity: float = 128.0  # default for backwards compat


# ---------------------------------------------------------------------------
# Adaptive area filter  (P2)
# ---------------------------------------------------------------------------

class AdaptiveAreaFilter:
    """Maintains a running median of recent detection areas and narrows
    the acceptance window around it.
    """

    def __init__(self, ratio: float = 3.0, history_len: int = 30) -> None:
        self._ratio = ratio
        self._history: deque[float] = deque(maxlen=history_len)

    @property
    def initialized(self) -> bool:
        """Need at least 3 samples before the filter kicks in."""
        return len(self._history) >= 3

    def update(self, area: float) -> None:
        """Record a new detection area."""
        self._history.append(area)

    def get_bounds(self) -> tuple[float, float] | None:
        """Return ``(min_area, max_area)`` or *None* if not yet ready."""
        if not self.initialized:
            return None
        median = float(np.median(list(self._history)))
        return (median / self._ratio, median * self._ratio)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track_frame(
    mask: np.ndarray,
    frame_idx: int,
    tracking: TrackingConfig,
    previous: TrackPoint | None,
    ambiguity_confidence: float,
    shadow_confidence: float,
    *,
    gray_frame: np.ndarray | None = None,
    kalman: object | None = None,
    adaptive_area: AdaptiveAreaFilter | None = None,
) -> TrackPoint:
    """Detect candidates in *mask* and select the best one.

    New optional parameters (all backwards-compatible):

    * *gray_frame* – preprocessed grayscale frame, used for mean-intensity
      scoring of candidates.
    * *kalman* – a ``KalmanPointTracker`` (from ``kalman_tracker.py``).
      When provided and initialised, Kalman gating replaces the raw
      ``max_jump_px`` distance filter in ``_filter_plausible``, and
      Kalman distance is used in ``_select_candidate``.
    * *adaptive_area* – an ``AdaptiveAreaFilter``.  When provided and
      initialised, the area acceptance window is narrowed to track the
      running median of recent detections.
    """
    candidates = _find_candidates(mask, gray_frame=gray_frame)
    plausible = _filter_plausible(
        candidates,
        tracking,
        previous,
        kalman=kalman,
        adaptive_area=adaptive_area,
    )

    if not plausible:
        return TrackPoint(
            frame_idx=frame_idx,
            x=None,
            y=None,
            area=None,
            confidence=0.0,
            flags=["NO_DETECTION"],
        )

    ambiguous = len(plausible) > 1
    selected = _select_candidate(
        plausible, tracking, previous, kalman=kalman,
    )

    flags: list[str] = []
    confidence = 0.9

    if ambiguous:
        flags.append("AMBIGUOUS_TARGET")
        confidence = min(confidence, ambiguity_confidence)

    if _is_shadow_like(selected):
        flags.append("SHADOW_LIKE")
        confidence = min(confidence, shadow_confidence)

    return TrackPoint(
        frame_idx=frame_idx,
        x=selected.centroid[0],
        y=selected.centroid[1],
        area=float(selected.area),
        confidence=float(confidence),
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _find_candidates(
    mask: np.ndarray,
    *,
    gray_frame: np.ndarray | None = None,
) -> list[Candidate]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates: list[Candidate] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid = (float(centroids[label][0]), float(centroids[label][1]))
        solidity = _compute_solidity(labels, label)
        aspect_ratio = _compute_aspect_ratio(w, h)

        # Mean intensity of the blob in the original gray frame (P3)
        if gray_frame is not None:
            blob_pixels = gray_frame[labels == label]
            mean_intensity = float(blob_pixels.mean()) if blob_pixels.size > 0 else 128.0
        else:
            mean_intensity = 128.0

        candidates.append(
            Candidate(
                label=label,
                centroid=centroid,
                area=area,
                bbox=(x, y, w, h),
                solidity=solidity,
                aspect_ratio=aspect_ratio,
                mean_intensity=mean_intensity,
            )
        )
    return candidates


def _compute_solidity(labels: np.ndarray, label: int) -> float:
    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return min(1.0, area / hull_area)


def _compute_aspect_ratio(width: int, height: int) -> float:
    if height <= 0:
        return 0.0
    return width / float(height)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_plausible(
    candidates: list[Candidate],
    tracking: TrackingConfig,
    previous: TrackPoint | None,
    *,
    kalman: object | None = None,
    adaptive_area: AdaptiveAreaFilter | None = None,
) -> list[Candidate]:
    # Resolve area bounds ------------------------------------------------
    area_lo: float = float(tracking.min_area_px)
    area_hi: float = float(tracking.max_area_px)

    if adaptive_area is not None:
        bounds = adaptive_area.get_bounds()
        if bounds is not None:
            # Tighten the window (never loosen beyond config limits)
            area_lo = max(area_lo, bounds[0])
            area_hi = min(area_hi, bounds[1])

    filtered: list[Candidate] = []
    for candidate in candidates:
        # Area gate
        if candidate.area < area_lo or candidate.area > area_hi:
            continue

        # Spatial gate: prefer Kalman, fall back to max_jump_px
        if kalman is not None and getattr(kalman, "initialized", False):
            if not kalman.is_within_gate(*candidate.centroid):  # type: ignore[union-attr]
                continue
        elif previous is not None and previous.x is not None and previous.y is not None:
            dist = _distance(candidate.centroid, (previous.x, previous.y))
            if dist > tracking.max_jump_px:
                continue

        filtered.append(candidate)
    return filtered


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_candidate(
    candidates: list[Candidate],
    tracking: TrackingConfig,
    previous: TrackPoint | None,
    *,
    kalman: object | None = None,
) -> Candidate:
    if len(candidates) == 1:
        return candidates[0]

    # Precompute the predicted position from Kalman (if available)
    kalman_pred: tuple[float, float] | None = None
    if kalman is not None and getattr(kalman, "initialized", False):
        kalman_pred = getattr(kalman, "predicted_position", None)

    # Normalise intensity for scoring: compute range across candidates
    intensities = [c.mean_intensity for c in candidates]
    i_min, i_max = min(intensities), max(intensities)
    i_range = i_max - i_min if i_max > i_min else 1.0

    def score(candidate: Candidate) -> float:
        solidity_score = candidate.solidity
        shadow_penalty = 0.35 if _is_shadow_like(candidate) else 0.0

        # Intensity: lower = more likely the (dark) mouse  →  higher score
        intensity_score = 1.0 - (candidate.mean_intensity - i_min) / i_range

        if kalman_pred is not None:
            # Use Kalman-predicted distance instead of raw previous
            dist = _distance(candidate.centroid, kalman_pred)
            # Scale by the Kalman gate radius for a [0..1] score
            gate_sigma = getattr(tracking, "kalman_gate_sigma", 4.0)
            gate_dist = kalman.gate_distance(*candidate.centroid)  # type: ignore[union-attr]
            distance_score = max(0.0, 1.0 - gate_dist / gate_sigma)
        elif previous is not None and previous.x is not None and previous.y is not None:
            dist = _distance(candidate.centroid, (previous.x, previous.y))
            distance_score = max(0.0, 1.0 - dist / tracking.max_jump_px)
        else:
            # Bootstrap: no previous → no distance score
            return (
                0.35 * solidity_score
                + 0.25 * intensity_score
                - shadow_penalty
            )

        area_score = 1.0
        if previous is not None and previous.area is not None and previous.area > 0:
            ratio = candidate.area / previous.area
            if ratio > 1.0:
                ratio = 1.0 / ratio
            area_score = max(0.0, min(1.0, float(ratio)))

        return (
            0.30 * distance_score
            + 0.15 * solidity_score
            + 0.15 * area_score
            + 0.25 * intensity_score
            - shadow_penalty
        )

    return max(candidates, key=score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_shadow_like(candidate: Candidate) -> bool:
    if candidate.solidity < 0.4:
        return True
    if candidate.aspect_ratio > 3.0 or candidate.aspect_ratio < 0.33:
        return True
    return False


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])
