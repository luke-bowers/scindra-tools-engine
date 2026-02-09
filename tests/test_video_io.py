from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scindra_engine.hash_utils import file_bytes, sha256_file
from scindra_engine.video_io import FrameSampler, VideoReader
from tests.fixtures.synth_video import make_synth_video


def test_video_reader_metadata_and_iteration(tmp_path: Path) -> None:
    video_path = make_synth_video(
        tmp_path, num_frames=10, size=(32, 24), fps=15.0
    )

    with VideoReader(video_path) as reader:
        assert reader.frame_count == 10
        assert reader.width == 32
        assert reader.height == 24

        # Some backends may not report FPS accurately; tolerate minor drift.
        fps = reader.fps
        assert fps >= 0.0
        if fps > 0.0:
            assert fps == pytest.approx(15.0, rel=0.2)

        frames = list(reader.iter_frames())

    indices = [idx for idx, _ in frames]
    assert indices == list(range(10))
    for _, frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (24, 32, 3)
        assert frame.dtype == np.uint8


def test_iter_frames_with_window_and_step(tmp_path: Path) -> None:
    video_path = make_synth_video(
        tmp_path, num_frames=10, size=(32, 24), fps=15.0
    )

    with VideoReader(video_path) as reader:
        frames = list(reader.iter_frames(start_frame=2, end_frame=8, step=2))

    indices = [idx for idx, _ in frames]
    assert indices == [2, 4, 6]


def test_frame_sampler_even_spread(tmp_path: Path) -> None:
    video_path = make_synth_video(
        tmp_path, num_frames=10, size=(32, 24), fps=15.0
    )

    with VideoReader(video_path) as reader:
        sampler = FrameSampler(reader)
        sampled = sampler.sample(3)

    indices = [idx for idx, _ in sampled]
    total = 10
    expected = [
        round(i * (total - 1) / (3 - 1)) for i in range(3)
    ]  # [0, 4 or 5, 9]
    assert indices == expected


def test_hash_utils_roundtrip_on_video(tmp_path: Path) -> None:
    video_path = make_synth_video(
        tmp_path, num_frames=10, size=(32, 24), fps=15.0
    )

    size = file_bytes(video_path)
    assert size == video_path.stat().st_size

    h1 = sha256_file(video_path)
    h2 = sha256_file(video_path)
    assert h1 == h2

