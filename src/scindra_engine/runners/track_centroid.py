from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

from scindra_engine.preprocess import BackgroundModel, build_background, preprocess_frame
from scindra_engine.schemas import TrackCentroidConfig
from scindra_engine.segmentation import segment_frame
from scindra_engine.tracking import TrackPoint, track_frame
from scindra_engine.video_io import FrameSampler, VideoReader


@dataclass(frozen=True)
class TrackCentroidResult:
    run_dir: Path
    points: list[TrackPoint]
    summary: dict[str, float]


def run_track_centroid(
    video_path: Path,
    out_dir: Path,
    config: TrackCentroidConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TrackCentroidResult:
    run_id = _make_run_id()
    run_dir = out_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with VideoReader(video_path) as reader:
        background = _build_background(reader, config)
        points = _track_video(reader, config, background, progress_callback)

    per_frame_path = run_dir / "per_frame.csv"
    _write_per_frame(per_frame_path, points)

    summary = _summarize(points, config)
    summary_path = run_dir / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return TrackCentroidResult(run_dir=run_dir, points=points, summary=summary)


def _make_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:8]
    return f"{timestamp}_{token}"


def _build_background(
    reader: VideoReader, config: TrackCentroidConfig
) -> BackgroundModel | None:
    if config.preprocessing.background_model == "none":
        return None
    sampler = FrameSampler(reader)
    frames = [frame for _, frame in sampler.sample(config.preprocessing.background_n)]
    return build_background(frames, config.preprocessing)


def _track_video(
    reader: VideoReader,
    config: TrackCentroidConfig,
    background: BackgroundModel | None,
    progress_callback: Callable[[int, int], None] | None,
) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    previous: TrackPoint | None = None
    ema_point: tuple[float, float] | None = None
    total = reader.frame_count

    for idx, frame in reader.iter_frames():
        gray = preprocess_frame(frame, config.preprocessing, background)
        mask = segment_frame(gray, config.segmentation, config.morphology)
        point = track_frame(
            mask,
            frame_idx=idx,
            tracking=config.tracking,
            previous=previous,
            ambiguity_confidence=config.ambiguity_confidence,
            shadow_confidence=config.shadow_confidence,
        )

        if point.x is not None and point.y is not None:
            if config.tracking.smoothing == "ema":
                ema_point = _update_ema(ema_point, point, config.tracking.ema_alpha)
                if ema_point is not None:
                    point = TrackPoint(
                        frame_idx=point.frame_idx,
                        x=ema_point[0],
                        y=ema_point[1],
                        area=point.area,
                        confidence=point.confidence,
                        flags=point.flags,
                    )
        points.append(point)

        if point.x is not None and point.y is not None:
            previous = point

        if progress_callback and (idx + 1) % config.progress_interval == 0:
            progress_callback(idx + 1, total)

    if progress_callback:
        progress_callback(total, total)
    return points


def _update_ema(
    previous: tuple[float, float] | None,
    point: TrackPoint,
    alpha: float,
) -> tuple[float, float] | None:
    if point.x is None or point.y is None:
        return previous
    if previous is None:
        return (point.x, point.y)
    return (
        alpha * point.x + (1.0 - alpha) * previous[0],
        alpha * point.y + (1.0 - alpha) * previous[1],
    )


def _write_per_frame(path: Path, points: list[TrackPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "x", "y", "area", "confidence", "flags"])
        for point in points:
            flags = ";".join(point.flags) if point.flags else ""
            writer.writerow(
                [
                    point.frame_idx,
                    "" if point.x is None else f"{point.x:.3f}",
                    "" if point.y is None else f"{point.y:.3f}",
                    "" if point.area is None else f"{point.area:.1f}",
                    f"{point.confidence:.3f}",
                    flags,
                ]
            )


def _summarize(
    points: list[TrackPoint], config: TrackCentroidConfig
) -> dict[str, float]:
    total = len(points)
    tracked = [p for p in points if p.x is not None and p.y is not None]
    coverage = (len(tracked) / total) if total > 0 else 0.0
    mean_conf = (
        float(np.mean([p.confidence for p in tracked])) if tracked else 0.0
    )
    jump_rate = _jump_rate(tracked, config)
    return {
        "coverage": float(coverage),
        "mean_conf": float(mean_conf),
        "jump_rate": float(jump_rate),
    }


def _jump_rate(
    points: list[TrackPoint], config: TrackCentroidConfig
) -> float:
    if len(points) < 2:
        return 0.0
    jumps = 0
    total = 0
    prev = points[0]
    for point in points[1:]:
        if (
            prev.x is not None
            and prev.y is not None
            and point.x is not None
            and point.y is not None
        ):
            dist = np.hypot(point.x - prev.x, point.y - prev.y)
            if dist > config.tracking.max_jump_px:
                jumps += 1
            total += 1
        prev = point
    return (jumps / total) if total > 0 else 0.0
