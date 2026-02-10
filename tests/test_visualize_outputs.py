from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore[import-untyped]

from scindra_engine.runners import run_track_centroid
from scindra_engine.schemas import TrackCentroidConfig
from scindra_engine.video_io import VideoReader
from scindra_engine.visualize import write_heatmap_png, write_overlay_video
from tests.fixtures.synth_mouse_shadow import make_synth_mouse_shadow_video


def test_visualize_overlay_and_heatmap(tmp_path: Path) -> None:
    video_path = make_synth_mouse_shadow_video(tmp_path)

    config = TrackCentroidConfig.model_validate({})

    result = run_track_centroid(
        video_path=video_path,
        out_dir=tmp_path / "track_out",
        config=config,
    )

    overlay_path = tmp_path / "overlay.mp4"
    heatmap_path = tmp_path / "heatmap.png"

    # Use a large progress interval to avoid noisy test output while still
    # exercising the overlay progress path.
    write_overlay_video(
        str(video_path),
        result.points,
        str(overlay_path),
        trail_length=30,
        progress_every_n=1000,
    )

    with VideoReader(video_path) as reader:
        width = reader.width
        height = reader.height

    write_heatmap_png(
        width=width,
        height=height,
        track_points=result.points,
        out_path=str(heatmap_path),
    )

    assert overlay_path.exists()
    assert overlay_path.stat().st_size > 0

    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(overlay_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    assert frame_count > 0

