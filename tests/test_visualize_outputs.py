from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np

from scindra_engine.runners import run_track_centroid
from scindra_engine.schemas import TrackCentroidConfig
from scindra_engine.tracking import Candidate, TrackFrameDebug
from scindra_engine.video_io import VideoReader
from scindra_engine.visualize import render_debug_frame, write_heatmap_png, write_overlay_video
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


def test_render_debug_frame_produces_bgr_image() -> None:
    """render_debug_frame returns a BGR image with correct shape and does not raise."""
    h, w = 60, 80
    frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    candidate = Candidate(
        label=1,
        centroid=(40.0, 30.0),
        area=400,
        bbox=(30, 20, 20, 20),
        solidity=0.9,
        aspect_ratio=1.0,
    )
    debug_info = TrackFrameDebug(
        all_candidates=(candidate,),
        excluded=(),
        plausible=(candidate,),
        selected=candidate,
    )
    out = render_debug_frame(frame_bgr, mask, debug_info)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8

