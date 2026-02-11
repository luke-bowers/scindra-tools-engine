from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

from scindra_engine.schemas import TrackingConfig


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


def track_frame(
    mask: np.ndarray,
    frame_idx: int,
    tracking: TrackingConfig,
    previous: TrackPoint | None,
    ambiguity_confidence: float,
    shadow_confidence: float,
) -> TrackPoint:
    candidates = _find_candidates(mask)
    plausible = _filter_plausible(candidates, tracking, previous)

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
    selected = _select_candidate(plausible, tracking, previous)

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


def _find_candidates(mask: np.ndarray) -> list[Candidate]:
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
        candidates.append(
            Candidate(
                label=label,
                centroid=centroid,
                area=area,
                bbox=(x, y, w, h),
                solidity=solidity,
                aspect_ratio=aspect_ratio,
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


def _filter_plausible(
    candidates: list[Candidate],
    tracking: TrackingConfig,
    previous: TrackPoint | None,
) -> list[Candidate]:
    filtered: list[Candidate] = []
    for candidate in candidates:
        if candidate.area < tracking.min_area_px:
            continue
        if candidate.area > tracking.max_area_px:
            continue
        if previous is not None and previous.x is not None and previous.y is not None:
            dist = _distance(
                candidate.centroid, (previous.x, previous.y)
            )
            if dist > tracking.max_jump_px:
                continue
        filtered.append(candidate)
    return filtered


def _select_candidate(
    candidates: list[Candidate],
    tracking: TrackingConfig,
    previous: TrackPoint | None,
) -> Candidate:
    if len(candidates) == 1:
        return candidates[0]

    def score(candidate: Candidate) -> float:
        solidity_score = candidate.solidity
        shadow_penalty = 0.35 if _is_shadow_like(candidate) else 0.0

        if previous is None or previous.x is None or previous.y is None:
            # Bootstrap: prefer compact, non-shadow-like shapes.
            return solidity_score - shadow_penalty

        dist = _distance(candidate.centroid, (previous.x, previous.y))
        distance_score = max(0.0, 1.0 - dist / tracking.max_jump_px)

        area_score = 1.0
        if previous.area is not None and previous.area > 0:
            ratio = candidate.area / previous.area
            if ratio > 1.0:
                ratio = 1.0 / ratio
            area_score = max(0.0, min(1.0, float(ratio)))

        # Bias toward consistent, compact motion and away from likely shadows.
        return (
            0.50 * distance_score
            + 0.30 * solidity_score
            + 0.20 * area_score
            - shadow_penalty
        )

    return max(candidates, key=score)


def _is_shadow_like(candidate: Candidate) -> bool:
    if candidate.solidity < 0.4:
        return True
    if candidate.aspect_ratio > 3.0 or candidate.aspect_ratio < 0.33:
        return True
    return False


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])
