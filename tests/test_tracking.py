from __future__ import annotations

from scindra_engine.schemas import TrackCentroidConfig, TrackingConfig
from scindra_engine.tracking import Candidate, TrackPoint, _select_candidate


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
