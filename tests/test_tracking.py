from __future__ import annotations

import numpy as np

from scindra_engine.schemas import TrackCentroidConfig, TrackingConfig
from scindra_engine.tracking import (
    Candidate,
    TrackFrameDebug,
    TrackPoint,
    _filter_plausible_with_reasons,
    _select_candidate,
    track_frame,
)


def test_track_centroid_default_segmentation_not_inverted() -> None:
    config = TrackCentroidConfig()
    assert config.segmentation.invert is False


def test_select_candidate_prefers_non_shadow_with_consistent_area() -> None:
    tracking = TrackingConfig(max_jump_px=80.0)
    previous = TrackPoint(
        frame_idx=4,
        x=40.0,
        y=30.0,
        area=180.0,
        confidence=0.9,
        flags=[],
    )

    mouse_like = Candidate(
        label=1,
        centroid=(47.0, 33.0),
        area=176,
        bbox=(0, 0, 10, 10),
        solidity=0.95,
        aspect_ratio=1.3,
    )
    shadow_like = Candidate(
        label=2,
        centroid=(46.0, 34.0),
        area=420,
        bbox=(0, 0, 24, 6),
        solidity=0.65,
        aspect_ratio=4.0,
    )

    selected = _select_candidate([shadow_like, mouse_like], tracking, previous)

    assert selected == mouse_like


def test_select_candidate_uses_compactness_when_bootstrapping() -> None:
    tracking = TrackingConfig(max_jump_px=80.0)
    compact = Candidate(
        label=1,
        centroid=(20.0, 20.0),
        area=150,
        bbox=(0, 0, 12, 10),
        solidity=0.90,
        aspect_ratio=1.2,
    )
    elongated = Candidate(
        label=2,
        centroid=(22.0, 21.0),
        area=150,
        bbox=(0, 0, 30, 6),
        solidity=0.92,
        aspect_ratio=5.0,
    )

    selected = _select_candidate([elongated, compact], tracking, previous=None)

    assert selected == compact


def test_filter_plausible_with_reasons_area_low() -> None:
    """Candidates below min_area_px are excluded with reason 'area_low'."""
    tracking = TrackingConfig(min_area_px=100, max_area_px=1000, max_jump_px=80.0)
    small = Candidate(
        label=1,
        centroid=(50.0, 50.0),
        area=50,
        bbox=(40, 40, 10, 10),
        solidity=0.9,
        aspect_ratio=1.0,
    )
    plausible, excluded = _filter_plausible_with_reasons([small], tracking, None)
    assert plausible == []
    assert len(excluded) == 1
    assert excluded[0][1] == "area_low"


def test_filter_plausible_with_reasons_area_high() -> None:
    """Candidates above max_area_px are excluded with reason 'area_high'."""
    tracking = TrackingConfig(min_area_px=100, max_area_px=1000, max_jump_px=80.0)
    large = Candidate(
        label=1,
        centroid=(50.0, 50.0),
        area=2000,
        bbox=(0, 0, 100, 20),
        solidity=0.9,
        aspect_ratio=5.0,
    )
    plausible, excluded = _filter_plausible_with_reasons([large], tracking, None)
    assert plausible == []
    assert len(excluded) == 1
    assert excluded[0][1] == "area_high"


def test_filter_plausible_with_reasons_spatial() -> None:
    """Candidates beyond max_jump_px from previous are excluded with reason 'spatial'."""
    tracking = TrackingConfig(min_area_px=50, max_area_px=10000, max_jump_px=50.0)
    previous = TrackPoint(
        frame_idx=0,
        x=100.0,
        y=100.0,
        area=200.0,
        confidence=0.9,
        flags=[],
    )
    far = Candidate(
        label=1,
        centroid=(200.0, 200.0),
        area=200,
        bbox=(180, 180, 40, 40),
        solidity=0.9,
        aspect_ratio=1.0,
    )
    plausible, excluded = _filter_plausible_with_reasons([far], tracking, previous)
    assert plausible == []
    assert len(excluded) == 1
    assert excluded[0][1] == "spatial"


def test_track_frame_debug_sink_appends_one_entry() -> None:
    """With debug_sink, track_frame appends one TrackFrameDebug per call."""
    tracking = TrackingConfig(min_area_px=10, max_area_px=100000, max_jump_px=500.0)
    # Mask: one white blob (connected component)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    debug_sink: list[TrackFrameDebug] = []
    point = track_frame(
        mask,
        frame_idx=0,
        tracking=tracking,
        previous=None,
        ambiguity_confidence=0.55,
        shadow_confidence=0.6,
        debug_sink=debug_sink,
    )
    assert len(debug_sink) == 1
    info = debug_sink[0]
    assert len(info.all_candidates) >= 1
    assert info.selected is not None
    assert point.x is not None and point.y is not None
