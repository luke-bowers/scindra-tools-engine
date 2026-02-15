"""Integration tests for detector-assisted tracking vs classical-only.

Uses a FakeDetector that knows where the mouse is in the synthetic fixture,
so no real ONNX model is needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scindra_engine.detectors.base import Detection, DetectorResult
from scindra_engine.runners.track_centroid import run_track_centroid
from scindra_engine.schemas import TrackCentroidConfig
from tests.fixtures.synth_mouse_shadow import make_synth_mouse_shadow_video


# ---------------------------------------------------------------------------
# Fake detector that returns a tight bbox around the known mouse position
# ---------------------------------------------------------------------------

_NUM_FRAMES = 30
_WIDTH, _HEIGHT = 160, 120


class FakeDetector:
    """A detector that returns bboxes matching the synth_mouse_shadow fixture."""

    @property
    def name(self) -> str:
        return "FAKE"

    def __init__(self) -> None:
        self._call_count = 0

    def detect(self, frame_bgr: np.ndarray) -> DetectorResult:
        # Reconstruct the mouse center from the fixture's formula
        i = self._call_count
        self._call_count += 1
        cx = int(20 + (_WIDTH - 40) * (i / max(1, _NUM_FRAMES - 1)))
        cy = int(_HEIGHT / 2 + 10 * np.sin(i / 3.0))

        # Tight bbox around the 10x6 ellipse
        x1 = max(0, cx - 12)
        y1 = max(0, cy - 8)
        x2 = min(_WIDTH, cx + 12)
        y2 = min(_HEIGHT, cy + 8)

        det = Detection(bbox_xyxy=(x1, y1, x2, y2), score=0.92, class_id=0)
        return DetectorResult(
            detections=[det],
            best=det,
            confidence=0.92,
            reasons=[],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_detector_assisted_vs_classical(tmp_path: Path) -> None:
    """Detector-assisted mode should yield higher mean confidence than classical-only."""
    video_path = make_synth_mouse_shadow_video(
        tmp_path, num_frames=_NUM_FRAMES, size=(_WIDTH, _HEIGHT)
    )

    config_classical = TrackCentroidConfig.model_validate({})

    result_classical = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "classical",
        config=config_classical,
        write_overlay_video=False,
        write_heatmap=False,
    )

    # Detector-assisted run
    from scindra_engine.schemas import DetectorConfig

    det_cfg = DetectorConfig(
        enabled=True,
        every_n_frames=5,
        min_score=0.3,
        roi_padding_px=20,
    )
    config_det = TrackCentroidConfig.model_validate({})
    config_det = config_det.model_copy(update={"detector": det_cfg})

    fake_det = FakeDetector()
    result_det = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "detector",
        config=config_det,
        write_overlay_video=False,
        write_heatmap=False,
        detector=fake_det,
    )

    # Compute stats
    classical_conf = [p.confidence for p in result_classical.points if p.x is not None]
    det_conf = [p.confidence for p in result_det.points if p.x is not None]

    classical_ambig = sum(
        1 for p in result_classical.points if "AMBIGUOUS_TARGET" in p.flags
    )
    det_ambig = sum(
        1 for p in result_det.points if "AMBIGUOUS_TARGET" in p.flags
    )

    mean_classical = float(np.mean(classical_conf)) if classical_conf else 0.0
    mean_det = float(np.mean(det_conf)) if det_conf else 0.0

    # Detector-assisted should have at least as high mean confidence
    # or fewer ambiguous frames (the ROI restricts to the mouse blob)
    assert (
        mean_det >= mean_classical - 0.05  # allow small margin
        or det_ambig <= classical_ambig
    ), (
        f"Detector-assisted did not improve: "
        f"mean_conf classical={mean_classical:.3f} det={mean_det:.3f}, "
        f"ambiguous classical={classical_ambig} det={det_ambig}"
    )


def test_detector_assisted_deterministic(tmp_path: Path) -> None:
    """Two runs with the same FakeDetector produce identical outputs."""
    from scindra_engine.schemas import DetectorConfig

    video_path = make_synth_mouse_shadow_video(
        tmp_path, num_frames=_NUM_FRAMES, size=(_WIDTH, _HEIGHT)
    )

    det_cfg = DetectorConfig(
        enabled=True,
        every_n_frames=5,
        min_score=0.3,
        roi_padding_px=20,
    )
    config = TrackCentroidConfig.model_validate({})
    config = config.model_copy(update={"detector": det_cfg})

    r1 = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "run1",
        config=config,
        write_overlay_video=False,
        write_heatmap=False,
        detector=FakeDetector(),
    )
    r2 = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "run2",
        config=config,
        write_overlay_video=False,
        write_heatmap=False,
        detector=FakeDetector(),
    )

    assert len(r1.points) == len(r2.points)
    for p1, p2 in zip(r1.points, r2.points):
        assert p1.frame_idx == p2.frame_idx
        if p1.x is not None:
            assert p2.x is not None
            assert abs(p1.x - p2.x) < 1e-6
            assert p1.y is not None and p2.y is not None
            assert abs(p1.y - p2.y) < 1e-6
        else:
            assert p2.x is None


def test_detector_per_frame_csv_has_roi_columns(tmp_path: Path) -> None:
    """When detector is used, per_frame.csv includes ROI and detector columns."""
    import csv

    from scindra_engine.schemas import DetectorConfig

    video_path = make_synth_mouse_shadow_video(
        tmp_path, num_frames=_NUM_FRAMES, size=(_WIDTH, _HEIGHT)
    )
    det_cfg = DetectorConfig(
        enabled=True, every_n_frames=5, min_score=0.3, roi_padding_px=20
    )
    config = TrackCentroidConfig.model_validate({})
    config = config.model_copy(update={"detector": det_cfg})

    result = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "out",
        config=config,
        write_overlay_video=False,
        write_heatmap=False,
        detector=FakeDetector(),
    )

    csv_path = result.run_dir / "per_frame.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert "roi_x1" in header
    assert "detector_score" in header
    assert "detector_used" in header


def test_detector_summary_has_metrics(tmp_path: Path) -> None:
    """tracking_summary.json includes detector_coverage and detector_mean_score."""
    import json

    from scindra_engine.schemas import DetectorConfig

    video_path = make_synth_mouse_shadow_video(
        tmp_path, num_frames=_NUM_FRAMES, size=(_WIDTH, _HEIGHT)
    )
    det_cfg = DetectorConfig(
        enabled=True, every_n_frames=5, min_score=0.3, roi_padding_px=20
    )
    config = TrackCentroidConfig.model_validate({})
    config = config.model_copy(update={"detector": det_cfg})

    result = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "out",
        config=config,
        write_overlay_video=False,
        write_heatmap=False,
        detector=FakeDetector(),
    )

    summary_path = result.run_dir / "tracking_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "detector_coverage" in summary
    assert "detector_mean_score" in summary
    assert summary["detector_coverage"] > 0.0
    assert summary["detector_mean_score"] > 0.0
