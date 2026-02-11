from __future__ import annotations

import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from scindra_engine.preprocess import BackgroundModel, build_background, preprocess_frame
from scindra_engine.schemas import TrackCentroidConfig
from scindra_engine.segmentation import segment_frame
from scindra_engine.tracking import TrackPoint, track_frame
from scindra_engine.video_io import FrameSampler, VideoReader
from scindra_engine.visualize import write_heatmap_png, write_overlay_video as _write_overlay_video


@dataclass(frozen=True)
class TrackCentroidResult:
    run_dir: Path
    points: list[TrackPoint]
    summary: dict[str, float]


class ThreadSafeProgressTracker:
    """Thread-safe progress tracker for parallel chunk processing."""

    def __init__(self, total: int, callback: Callable[[int, int], None] | None = None) -> None:
        self._total = total
        self._callback = callback
        self._lock = threading.Lock()
        self._processed = 0
        self._start_time = time.time()

    def update(self, increment: int) -> None:
        """Update progress by incrementing processed count."""
        with self._lock:
            self._processed = min(self._processed + increment, self._total)
            if self._callback:
                self._callback(self._processed, self._total)

    def get_progress(self) -> tuple[int, int]:
        """Get current progress (processed, total)."""
        with self._lock:
            return (self._processed, self._total)


def run_track_centroid(
    video_path: Path,
    out_dir: Path,
    config: TrackCentroidConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    write_overlay_video: bool | None = None,
    write_heatmap: bool | None = None,
    trail_length: int | None = None,
    parallel_workers: int | None = None,
    chunk_size: int | None = None,
) -> TrackCentroidResult:
    run_id = _make_run_id()
    run_dir = out_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    width = 0
    height = 0

    with VideoReader(video_path) as reader:
        background = _build_background(reader, config)
        width = reader.width
        height = reader.height

    # Use parallel processing if enabled (default: enabled for videos with sufficient frames)
    # Fall back to sequential for small videos or if parallel processing fails
    effective_workers = parallel_workers if parallel_workers is not None else config.parallel_workers
    effective_chunk_size = chunk_size if chunk_size is not None else config.chunk_size

    try:
        points = _track_video_parallel(
            video_path,
            config,
            background,
            progress_callback,
            num_workers=effective_workers,
            chunk_size=effective_chunk_size,
        )
    except Exception:
        # Fallback to sequential processing on any error
        with VideoReader(video_path) as reader:
            points = _track_video(reader, config, background, progress_callback)

    per_frame_path = run_dir / "per_frame.csv"
    _write_per_frame(per_frame_path, points)

    summary = _summarize(points, config)
    summary_path = run_dir / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    overlay_enabled, heatmap_enabled, effective_trail_length = (
        _resolve_visualization_options(
            config=config,
            write_overlay_video_override=write_overlay_video,
            write_heatmap_override=write_heatmap,
            trail_length_override=trail_length,
        )
    )

    if overlay_enabled:
        overlay_path = run_dir / "overlay.mp4"
        _write_overlay_video(
            str(video_path),
            points,
            str(overlay_path),
            trail_length=effective_trail_length,
        )

    if heatmap_enabled:
        if width <= 0 or height <= 0:
            raise RuntimeError("Video dimensions are not available for heatmap output")
        heatmap_path = run_dir / "heatmap.png"
        write_heatmap_png(
            width=width,
            height=height,
            track_points=points,
            out_path=str(heatmap_path),
        )

    return TrackCentroidResult(run_dir=run_dir, points=points, summary=summary)


def _make_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:8]
    return f"{timestamp}_{token}"


def _downsample_frame(frame: np.ndarray, factor: float) -> np.ndarray:
    """Downsample a frame by the given factor.
    
    Args:
        frame: Input frame in BGR format.
        factor: Downsampling factor (e.g., 2.0 = half resolution).
        
    Returns:
        Downsampled frame.
    """
    if factor <= 1.0:
        return frame
    new_w = int(frame.shape[1] / factor)
    new_h = int(frame.shape[0] / factor)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _create_scaled_tracking_config(
    config: TrackCentroidConfig,
) -> TrackCentroidConfig:
    """Create a config with scaled tracking parameters for downsampled frames.
    
    Args:
        config: Original config.
        
    Returns:
        Config with scaled tracking parameters if downsampling is enabled.
    """
    from scindra_engine.schemas import TrackingConfig
    
    if config.downsample_factor is None or config.downsample_factor <= 1.0:
        return config
    
    factor = config.downsample_factor
    scaled_tracking = TrackingConfig(
        min_area_px=int(config.tracking.min_area_px / (factor * factor)),
        max_area_px=int(config.tracking.max_area_px / (factor * factor)),
        max_jump_px=config.tracking.max_jump_px / factor,
        smoothing=config.tracking.smoothing,
        ema_alpha=config.tracking.ema_alpha,
    )
    
    # Create a new config with scaled tracking
    return TrackCentroidConfig(
        preprocessing=config.preprocessing,
        segmentation=config.segmentation,
        morphology=config.morphology,
        tracking=scaled_tracking,
        progress_interval=config.progress_interval,
        ambiguity_confidence=config.ambiguity_confidence,
        shadow_confidence=config.shadow_confidence,
        parallel_workers=config.parallel_workers,
        chunk_size=config.chunk_size,
        downsample_factor=config.downsample_factor,
    )


def _build_background(
    reader: VideoReader, config: TrackCentroidConfig
) -> BackgroundModel | None:
    if config.preprocessing.background_model == "none":
        return None
    sampler = FrameSampler(reader)
    frames = [frame for _, frame in sampler.sample(config.preprocessing.background_n)]
    
    # Downsample frames if downsampling is enabled
    if config.downsample_factor is not None and config.downsample_factor > 1.0:
        frames = [_downsample_frame(frame, config.downsample_factor) for frame in frames]
    
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
    
    # Create scaled config if downsampling is enabled
    effective_config = _create_scaled_tracking_config(config)
    downsample_factor = config.downsample_factor if config.downsample_factor is not None and config.downsample_factor > 1.0 else None

    for idx, frame in reader.iter_frames():
        # Downsample frame if downsampling is enabled
        if downsample_factor is not None:
            frame = _downsample_frame(frame, downsample_factor)
        
        gray = preprocess_frame(frame, effective_config.preprocessing, background)
        mask = segment_frame(gray, effective_config.segmentation, effective_config.morphology)
        point = track_frame(
            mask,
            frame_idx=idx,
            tracking=effective_config.tracking,
            previous=previous,
            ambiguity_confidence=effective_config.ambiguity_confidence,
            shadow_confidence=effective_config.shadow_confidence,
        )

        # Scale coordinates back to original resolution
        if downsample_factor is not None and point.x is not None and point.y is not None:
            scaled_x = point.x * downsample_factor
            scaled_y = point.y * downsample_factor
            scaled_area = point.area * (downsample_factor * downsample_factor) if point.area is not None else None
            point = TrackPoint(
                frame_idx=point.frame_idx,
                x=scaled_x,
                y=scaled_y,
                area=scaled_area,
                confidence=point.confidence,
                flags=point.flags,
            )
            # Keep previous in downsampled space for next iteration's tracking
            if point.x is not None and point.y is not None:
                previous = TrackPoint(
                    frame_idx=point.frame_idx,
                    x=point.x / downsample_factor,
                    y=point.y / downsample_factor,
                    area=point.area / (downsample_factor * downsample_factor) if point.area is not None else None,
                    confidence=point.confidence,
                    flags=point.flags,
                )
        else:
            # No downsampling, use point directly for previous
            if point.x is not None and point.y is not None:
                previous = point

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


def _process_chunk(
    video_path: Path,
    chunk_start: int,
    chunk_end: int,
    config: TrackCentroidConfig,
    background: BackgroundModel | None,
    previous_point: TrackPoint | None,
    *,
    apply_ema: bool = False,
    progress_tracker: ThreadSafeProgressTracker | None = None,
    progress_interval: int = 10,
) -> list[TrackPoint]:
    """Process a chunk of frames from a video.

    Args:
        video_path: Path to the video file.
        chunk_start: Starting frame index (inclusive).
        chunk_end: Ending frame index (exclusive).
        config: Tracking configuration.
        background: Background model (shared across chunks).
        previous_point: Last valid point from previous chunk (for continuity).
        apply_ema: Whether to apply EMA smoothing within the chunk (False for parallel processing).
        progress_tracker: Optional thread-safe progress tracker for real-time updates.
        progress_interval: Report progress every N frames within the chunk.

    Returns:
        List of TrackPoints for frames in [chunk_start, chunk_end).
    """
    points: list[TrackPoint] = []
    
    # Create scaled config if downsampling is enabled
    effective_config = _create_scaled_tracking_config(config)
    downsample_factor = config.downsample_factor if config.downsample_factor is not None and config.downsample_factor > 1.0 else None
    
    # Scale down previous_point if downsampling is enabled
    if downsample_factor is not None and previous_point is not None and previous_point.x is not None and previous_point.y is not None:
        scaled_previous = TrackPoint(
            frame_idx=previous_point.frame_idx,
            x=previous_point.x / downsample_factor,
            y=previous_point.y / downsample_factor,
            area=previous_point.area / (downsample_factor * downsample_factor) if previous_point.area is not None else None,
            confidence=previous_point.confidence,
            flags=previous_point.flags,
        )
        previous: TrackPoint | None = scaled_previous
    else:
        previous: TrackPoint | None = previous_point
    
    ema_point: tuple[float, float] | None = None
    frames_in_chunk = chunk_end - chunk_start

    with VideoReader(video_path) as reader:
        frame_count = 0
        for idx, frame in reader.iter_frames(start_frame=chunk_start, end_frame=chunk_end):
            # Downsample frame if downsampling is enabled
            if downsample_factor is not None:
                frame = _downsample_frame(frame, downsample_factor)
            
            gray = preprocess_frame(frame, effective_config.preprocessing, background)
            mask = segment_frame(gray, effective_config.segmentation, effective_config.morphology)
            point = track_frame(
                mask,
                frame_idx=idx,
                tracking=effective_config.tracking,
                previous=previous,
                ambiguity_confidence=effective_config.ambiguity_confidence,
                shadow_confidence=effective_config.shadow_confidence,
            )

            # Scale coordinates back to original resolution
            if downsample_factor is not None and point.x is not None and point.y is not None:
                scaled_x = point.x * downsample_factor
                scaled_y = point.y * downsample_factor
                scaled_area = point.area * (downsample_factor * downsample_factor) if point.area is not None else None
                point = TrackPoint(
                    frame_idx=point.frame_idx,
                    x=scaled_x,
                    y=scaled_y,
                    area=scaled_area,
                    confidence=point.confidence,
                    flags=point.flags,
                )
                # Keep previous in downsampled space for next iteration's tracking
                if point.x is not None and point.y is not None:
                    previous = TrackPoint(
                        frame_idx=point.frame_idx,
                        x=point.x / downsample_factor,
                        y=point.y / downsample_factor,
                        area=point.area / (downsample_factor * downsample_factor) if point.area is not None else None,
                        confidence=point.confidence,
                        flags=point.flags,
                    )
            else:
                # No downsampling, use point directly for previous
                if point.x is not None and point.y is not None:
                    previous = point

            if point.x is not None and point.y is not None:
                if apply_ema and config.tracking.smoothing == "ema":
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

            frame_count += 1
            # Report progress periodically during chunk processing
            if progress_tracker and frame_count % progress_interval == 0:
                progress_tracker.update(progress_interval)

        # Report any remaining frames
        if progress_tracker:
            remaining = frame_count % progress_interval
            if remaining > 0:
                progress_tracker.update(remaining)

    return points


def _reapply_ema(
    points: list[TrackPoint],
    alpha: float,
) -> list[TrackPoint]:
    """Apply EMA smoothing across the entire point list sequentially.

    This ensures consistent EMA smoothing across chunk boundaries.
    Assumes points do not already have EMA applied (raw tracking points).

    Args:
        points: List of TrackPoints (raw, without EMA smoothing).
        alpha: EMA alpha parameter.

    Returns:
        List of TrackPoints with EMA smoothing applied globally.
    """
    if not points:
        return points

    result: list[TrackPoint] = []
    ema_point: tuple[float, float] | None = None

    for point in points:
        if point.x is not None and point.y is not None:
            ema_point = _update_ema(ema_point, point, alpha)
            if ema_point is not None:
                point = TrackPoint(
                    frame_idx=point.frame_idx,
                    x=ema_point[0],
                    y=ema_point[1],
                    area=point.area,
                    confidence=point.confidence,
                    flags=point.flags,
                )
        result.append(point)

    return result


def _track_video_parallel(
    video_path: Path,
    config: TrackCentroidConfig,
    background: BackgroundModel | None,
    progress_callback: Callable[[int, int], None] | None,
    *,
    num_workers: int | None = None,
    chunk_size: int = 200,
) -> list[TrackPoint]:
    """Track video frames using parallel chunk processing.

    Args:
        video_path: Path to the video file.
        config: Tracking configuration.
        background: Background model.
        progress_callback: Optional callback for progress updates.
        num_workers: Number of parallel workers (default: CPU count).
        chunk_size: Number of frames per chunk.

    Returns:
        List of TrackPoints for all frames.
    """
    # Get video metadata
    with VideoReader(video_path) as reader:
        total_frames = reader.frame_count

    if total_frames <= 0:
        return []

    # Use sequential processing for small videos
    if total_frames <= chunk_size:
        with VideoReader(video_path) as reader:
            return _track_video(reader, config, background, progress_callback)

    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, min(num_workers, (total_frames + chunk_size - 1) // chunk_size))

    # Create chunks
    chunks: list[tuple[int, int]] = []
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))

    # Process chunks in parallel
    # Note: Chunks after the first won't have the correct previous_point,
    # but this is acceptable because:
    # 1. The previous_point is mainly used for filtering (max_jump_px) and selection
    # 2. Missing previous_point just means the first frame of a chunk won't filter
    #    by distance, which is fine for parallelization
    # 3. EMA smoothing will be re-applied globally after merging
    all_points: list[TrackPoint] = []

    # Create thread-safe progress tracker for real-time updates from worker threads
    progress_tracker = ThreadSafeProgressTracker(total_frames, progress_callback) if progress_callback else None

    # Progress reporting interval within chunks (report every N frames)
    # Use a reasonable granularity - not too frequent to avoid overhead
    progress_interval = max(1, min(config.progress_interval, chunk_size // 4))

    # Start a background thread to periodically update progress display
    # This ensures progress updates even when chunks are processing slowly
    progress_update_thread: threading.Thread | None = None
    stop_progress_updates = threading.Event()

    if progress_tracker and progress_callback:
        def progress_updater() -> None:
            """Periodically update progress display."""
            update_interval = 0.5  # Update every 0.5 seconds
            while not stop_progress_updates.is_set():
                processed, total = progress_tracker.get_progress()
                if processed < total:
                    progress_callback(processed, total)
                stop_progress_updates.wait(update_interval)

        progress_update_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_update_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks in parallel
            future_to_chunk = {}
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                # Only pass previous_point for the first chunk
                # Other chunks will process without it (acceptable trade-off for parallelism)
                future = executor.submit(
                    _process_chunk,
                    video_path,
                    chunk_start,
                    chunk_end,
                    config,
                    background,
                    None,  # Don't pass previous_point to maintain true parallelism
                    apply_ema=False,  # Don't apply EMA in chunks, will apply globally after merging
                    progress_tracker=progress_tracker,
                    progress_interval=progress_interval,
                )
                future_to_chunk[future] = chunk_idx

            # Collect results and merge in order
            chunk_results: dict[int, list[TrackPoint]] = {}

            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_points = future.result()
                    chunk_results[chunk_idx] = chunk_points
                except Exception:
                    # If parallel processing fails, fall back to sequential
                    executor.shutdown(wait=False, cancel_futures=True)
                    if progress_callback:
                        progress_callback(0, total_frames)
                    with VideoReader(video_path) as reader:
                        return _track_video(reader, config, background, progress_callback)

            # Merge chunks in order
            for chunk_idx in sorted(chunk_results.keys()):
                all_points.extend(chunk_results[chunk_idx])

    finally:
        # Stop the progress update thread
        if progress_update_thread:
            stop_progress_updates.set()
            progress_update_thread.join(timeout=1.0)

    # Re-apply EMA smoothing globally to ensure consistency across chunk boundaries
    if config.tracking.smoothing == "ema":
        all_points = _reapply_ema(all_points, config.tracking.ema_alpha)

    if progress_callback:
        progress_callback(total_frames, total_frames)

    return all_points


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


def _resolve_visualization_options(
    *,
    config: TrackCentroidConfig,
    write_overlay_video_override: bool | None,
    write_heatmap_override: bool | None,
    trail_length_override: int | None,
) -> tuple[bool, bool, int]:
    """Resolve overlay/heatmap settings from config and overrides.

    Defaults are overlay and heatmap enabled, with a short trail. If the
    config later gains an `outputs` section, its fields will be respected but
    still overridden by explicit function arguments (e.g. CLI flags).
    """
    overlay_default = True
    heatmap_default = True
    trail_default = 30

    outputs = getattr(config, "outputs", None)
    if outputs is not None:
        overlay_default = getattr(outputs, "write_overlay_video", overlay_default)
        heatmap_default = getattr(outputs, "write_heatmap", heatmap_default)
        trail_default = getattr(outputs, "trail_length", trail_default)

    overlay_enabled = (
        overlay_default if write_overlay_video_override is None else write_overlay_video_override
    )
    heatmap_enabled = (
        heatmap_default if write_heatmap_override is None else write_heatmap_override
    )

    trail_value = trail_default if trail_length_override is None else trail_length_override
    if trail_value <= 0:
        trail_value = 1

    return overlay_enabled, heatmap_enabled, trail_value

