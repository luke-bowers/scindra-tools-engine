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

from scindra_engine.arena_crop import (
    build_static_image,
    crop_frame,
    detect_arena_crop_xyxy,
    expand_arena_box_xyxy,
    get_arena_detection_edges,
    get_arena_detection_edges_with_close,
)
from scindra_engine.detectors.base import Detector
from scindra_engine.detectors.state import DetectorState, FrameDetectorInfo
from scindra_engine.kalman_tracker import KalmanPointTracker
from scindra_engine.motion import MotionAccumulator
from scindra_engine.preprocess import BackgroundModel, build_background, preprocess_frame
from scindra_engine.schemas import KeyFrameInterpolationConfig, TrackCentroidConfig
from scindra_engine.segmentation import segment_frame
from scindra_engine.tracking import AdaptiveAreaFilter, TrackFrameDebug, TrackPoint, track_frame
from scindra_engine.video_io import (
    FrameSampler,
    VideoReader,
    crop_to_display_aspect,
    draw_crop_box_for_display,
    ensure_square_image,
    fix_video_display_aspect_ratio,
    get_video_display_aspect_ratio,
    require_ffmpeg_available,
    resize_to_display_aspect,
)
from scindra_engine.visualize import write_heatmap_png, write_overlay_video as _write_overlay_video
from scindra_engine.visualize.debug_blobs import render_debug_frame


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


class _GlobalProgress:
    """Aggregate progress across multiple sequential phases.

    Translates per-phase (done, total) updates into a single global (done, total)
    stream suitable for the CLI progress callback. Updates are throttled in time
    to avoid overwhelming the terminal.
    """

    def __init__(
        self,
        total_units: int,
        callback: Callable[[int, int], None] | None,
        min_interval_sec: float = 0.25,
    ) -> None:
        self._total_units = max(1, total_units)
        self._callback = callback
        self._min_interval = min_interval_sec
        self._last_reported: int = -1
        self._last_time: float = 0.0
        self._phase_offset: int = 0

    def start_phase(self, units: int) -> "_PhaseProgress":
        units = max(0, units)
        phase = _PhaseProgress(self, self._phase_offset, units)
        self._phase_offset += units
        return phase

    def report(
        self,
        offset_units: int,
        phase_units: int,
        local_done: int,
        local_total: int,
    ) -> None:
        if self._callback is None:
            return
        if phase_units <= 0 or local_total <= 0:
            return

        # Map local [0, local_total] -> global [offset_units, offset_units + phase_units]
        frac = max(0.0, min(float(local_done) / float(local_total), 1.0))
        global_done = offset_units + int(round(frac * phase_units))
        global_done = max(0, min(global_done, self._total_units))

        if global_done <= self._last_reported:
            return

        now = time.time()
        # Throttle updates while still in-flight to avoid spamming the terminal.
        if global_done < self._total_units and self._last_time > 0.0:
            if (now - self._last_time) < self._min_interval:
                return

        self._last_reported = global_done
        self._last_time = now
        self._callback(global_done, self._total_units)

    def finalize(self) -> None:
        """Ensure a final completed update is sent."""
        if self._callback is None:
            return
        if self._last_reported < self._total_units:
            self._callback(self._total_units, self._total_units)
            self._last_reported = self._total_units


class _PhaseProgress:
    """Adapter that converts per-phase progress into global progress units."""

    def __init__(
        self,
        global_progress: _GlobalProgress,
        offset_units: int,
        phase_units: int,
    ) -> None:
        self._global = global_progress
        self._offset_units = offset_units
        self._phase_units = phase_units

    def callback(self, done: int, total: int) -> None:
        self._global.report(self._offset_units, self._phase_units, done, total)


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
    detector: Detector | None = None,
    command: str | None = None,
) -> TrackCentroidResult:
    run_id = _make_run_id()
    run_dir = out_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    width = 0
    height = 0
    total_frames = 0

    # Probe video metadata once up front (needed for progress and arena mask)
    with VideoReader(video_path) as meta_reader:
        total_frames = meta_reader.frame_count
        # Use actual decoded frame dimensions (handles rotation metadata mismatch)
        width, height = meta_reader.get_actual_dimensions()

    # --- Arena crop: compute crop_xyxy and effective dimensions ---
    crop_xyxy: tuple[int, int, int, int] | None = None
    chosen_circle: tuple[int, int, int] | None = None  # (cx, cy, r) when crop came from Hough
    eff_width = width
    eff_height = height
    arena_crop_config = getattr(config, "arena_crop", None)
    if arena_crop_config is not None and arena_crop_config.enabled:
        if arena_crop_config.mode == "MANUAL" and arena_crop_config.manual_crop_xyxy is not None:
            crop_xyxy = arena_crop_config.manual_crop_xyxy
            eff_width = crop_xyxy[2] - crop_xyxy[0]
            eff_height = crop_xyxy[3] - crop_xyxy[1]
        else:
            # AUTO: will be computed in arena_crop_phase below (after progress is set up)
            pass

    # --- Detector + debug configuration (does not require frames) ---
    detector_state: DetectorState | None = None
    if config.detector.enabled and detector is not None:
        detector_state = DetectorState(detector, config.detector)

    debug_frames_dir: Path | None = None
    if getattr(config, "debug_mode", False):
        debug_frames_dir = run_dir / "debug_frames"
        debug_frames_dir.mkdir(parents=True, exist_ok=True)

    use_precompute_parallel = (
        detector_state is not None
        and getattr(config.detector, "precompute_roi_parallel", False)
        and debug_frames_dir is None
    )

    # Resolve overlay / heatmap settings and trail length once
    overlay_enabled, heatmap_enabled, effective_trail_length = _resolve_visualization_options(
        config=config,
        write_overlay_video_override=write_overlay_video,
        write_heatmap_override=write_heatmap,
        trail_length_override=trail_length,
    )

    # Estimate work units for each phase for global progress aggregation
    if config.preprocessing.background_model == "none":
        background_samples = 0
    else:
        n_sample = config.preprocessing.background_n
        if config.preprocessing.background_model == "mog2":
            n_sample = max(n_sample, 50)
        background_samples = min(n_sample, total_frames) if total_frames > 0 else n_sample

    arena_crop_units = 0
    if arena_crop_config is not None and arena_crop_config.enabled and arena_crop_config.mode == "AUTO":
        arena_crop_units = min(arena_crop_config.n_frames_static, total_frames) if total_frames > 0 else arena_crop_config.n_frames_static

    detector_pre_units = total_frames if use_precompute_parallel and total_frames > 0 else 0
    tracking_units = total_frames if total_frames > 0 else 0
    overlay_units = total_frames if overlay_enabled and total_frames > 0 else 0
    heatmap_units = total_frames if heatmap_enabled and total_frames > 0 else 0

    total_units = arena_crop_units + background_samples + detector_pre_units + tracking_units + overlay_units + heatmap_units

    global_progress: _GlobalProgress | None = None
    if progress_callback is not None and total_units > 0:
        global_progress = _GlobalProgress(total_units=total_units, callback=progress_callback)

    # Arena crop AUTO phase: build static image and detect crop box
    if arena_crop_units > 0 and arena_crop_config is not None:
        arena_crop_phase = global_progress.start_phase(arena_crop_units) if global_progress is not None else None
        with VideoReader(video_path) as crop_reader:
            static_img = build_static_image(
                crop_reader,
                arena_crop_units,
                method=arena_crop_config.static_method,
                progress_callback=arena_crop_phase.callback if arena_crop_phase is not None else None,
            )
        # Save static image and edges so user can see what the pipeline is analyzing (resize to DAR to avoid distortion)
        dar = get_video_display_aspect_ratio(video_path)
        static_for_png = resize_to_display_aspect(static_img, dar)
        cv2.imwrite(str(run_dir / "arena_crop_static.png"), static_for_png)
        canny_low = getattr(arena_crop_config, "canny_low", 50)
        canny_high = getattr(arena_crop_config, "canny_high", 150)
        blur_ksize = getattr(arena_crop_config, "blur_ksize", 5)
        edges = get_arena_detection_edges(static_img, canny_low=canny_low, canny_high=canny_high, blur_ksize=blur_ksize)
        edges_for_png = resize_to_display_aspect(edges, dar)
        cv2.imwrite(str(run_dir / "arena_crop_edges.png"), edges_for_png)
        morph_close_ksize = getattr(arena_crop_config, "morph_close_ksize", 0)
        if morph_close_ksize > 0:
            edges_closed = get_arena_detection_edges_with_close(
                static_img,
                canny_low=canny_low,
                canny_high=canny_high,
                blur_ksize=blur_ksize,
                morph_close_ksize=morph_close_ksize,
            )
            edges_closed_png = resize_to_display_aspect(edges_closed, dar)
            cv2.imwrite(str(run_dir / "arena_crop_edges_closed.png"), edges_closed_png)
        _debug_out = getattr(arena_crop_config, "debug_output_dir", None)
        if _debug_out:
            arena_debug_dir = Path(_debug_out).resolve()
        elif debug_frames_dir is not None:
            arena_debug_dir = run_dir / "arena_debug"
        else:
            arena_debug_dir = None
        arena_debug_manifest: list[dict] = []

        def _arena_debug_cb(step: str, data: dict) -> None:
            img = data.get("image")
            if img is not None and arena_debug_dir is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                out_img = resize_to_display_aspect(img, dar)
                cv2.imwrite(str(arena_debug_dir / f"arena_debug_{step}.png"), out_img)
            if step == "open_field" and arena_debug_dir is not None:
                mask = data.get("mask")
                if mask is not None and isinstance(mask, np.ndarray):
                    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_out = resize_to_display_aspect(mask_vis, dar)
                    cv2.imwrite(str(arena_debug_dir / "arena_debug_open_field_mask.png"), mask_out)
            entry = {k: v for k, v in data.items() if k not in ("image", "mask")}
            entry["step"] = step
            arena_debug_manifest.append(entry)

        if arena_debug_dir is not None:
            arena_debug_dir.mkdir(parents=True, exist_ok=True)
        arena_type_val = getattr(arena_crop_config, "arena_type", "elevated_zero")
        box, chosen_circle = detect_arena_crop_xyxy(
            static_img,
            margin_px=arena_crop_config.margin_px,
            min_area_ratio=arena_crop_config.min_area_ratio,
            canny_low=canny_low,
            canny_high=canny_high,
            blur_ksize=blur_ksize,
            morph_close_ksize=getattr(arena_crop_config, "morph_close_ksize", 0),
            use_hough_circle=getattr(arena_crop_config, "use_hough_circle", True),
            hough_min_radius_ratio=getattr(arena_crop_config, "hough_min_radius_ratio", 0.08),
            hough_max_radius_ratio=getattr(arena_crop_config, "hough_max_radius_ratio", 0.48),
            hough_center_margin_ratio=getattr(arena_crop_config, "hough_center_margin_ratio", 0.15),
            hough_acc_threshold=getattr(arena_crop_config, "hough_acc_threshold", 25),
            circle_only=getattr(arena_crop_config, "circle_only", False),
            circle_padding_ratio=getattr(arena_crop_config, "circle_padding_ratio", 0.03),
            force_square_crop=getattr(arena_crop_config, "force_square_crop", True),
            debug_callback=_arena_debug_cb if arena_debug_dir is not None else None,
            dar=dar,
            arena_type=arena_type_val,
            open_field_white_threshold=getattr(arena_crop_config, "open_field_white_threshold", 200),
            open_field_min_area_ratio=getattr(arena_crop_config, "open_field_min_area_ratio", 0.02),
            open_field_rectangularity_min=getattr(arena_crop_config, "open_field_rectangularity_min", 0.6),
        )
        if arena_debug_dir is not None:
            (arena_debug_dir / "arena_debug_manifest.json").write_text(
                json.dumps(arena_debug_manifest, indent=2), encoding="utf-8"
            )
        if box is not None:
            expand_ratio = getattr(arena_crop_config, "crop_expand_ratio", 0.0)
            if expand_ratio > 0:
                h, w = static_img.shape[:2]
                box = expand_arena_box_xyxy(box, w, h, expand_ratio)
            crop_xyxy = box
            eff_width = crop_xyxy[2] - crop_xyxy[0]
            eff_height = crop_xyxy[3] - crop_xyxy[1]
            if not getattr(arena_crop_config, "use_circle_mask", True):
                chosen_circle = None
            # Output: full frame with detected box (drawn in DAR space so box shape is correct), closed edges (if used), and cropped frame
            box_for_png = draw_crop_box_for_display(static_img, crop_xyxy, dar)
            cv2.imwrite(str(run_dir / "arena_crop_box.png"), box_for_png)
            cropped = crop_to_display_aspect(static_img, crop_xyxy, dar)
            if chosen_circle is not None and getattr(arena_crop_config, "force_square_crop", True):
                cropped = ensure_square_image(cropped)
            cv2.imwrite(str(run_dir / "arena_crop_cropped.png"), cropped)

    # Background model (may emit its own phase progress)
    background_phase: _PhaseProgress | None = None
    if global_progress is not None and background_samples > 0:
        background_phase = global_progress.start_phase(background_samples)

    with VideoReader(video_path) as reader:
        background = _build_background(
            reader,
            config,
            crop_xyxy=crop_xyxy,
            progress_callback=background_phase.callback if background_phase is not None else None,
        )

    # Write arena_crop.json so the user can see whether crop was applied and the box
    _write_arena_crop_info(run_dir, crop_xyxy, eff_width, eff_height, arena_crop_config)

    # When debug mode and crop applied, write a preview image (full frame 0 with crop box)
    if debug_frames_dir is not None and crop_xyxy is not None:
        _write_arena_crop_preview(video_path, crop_xyxy, debug_frames_dir, dar)

    # Build arena ROI mask once (at processing resolution; use effective dimensions when cropped)
    arena_mask = _load_or_build_arena_mask(config, eff_height, eff_width, crop_xyxy=crop_xyxy)
    if chosen_circle is not None and crop_xyxy is not None:
        circle_mask = _build_circle_arena_mask(
            crop_xyxy,
            chosen_circle,
            eff_width,
            eff_height,
            config.downsample_factor,
        )
        if arena_mask is not None:
            arena_mask = cv2.bitwise_and(arena_mask, circle_mask)
        else:
            arena_mask = circle_mask

    det_infos: list[FrameDetectorInfo] | None = None

    # Phase adapters for detector pre-pass and tracking
    detector_phase: _PhaseProgress | None = None
    tracking_phase: _PhaseProgress | None = None
    if global_progress is not None:
        if detector_pre_units > 0:
            detector_phase = global_progress.start_phase(detector_pre_units)
        if tracking_units > 0:
            tracking_phase = global_progress.start_phase(tracking_units)

    # Choose which callback to use for tracking depending on whether we are aggregating phases
    tracking_progress_cb: Callable[[int, int], None] | None
    if tracking_phase is not None:
        tracking_progress_cb = tracking_phase.callback
    else:
        tracking_progress_cb = progress_callback

    dar = get_video_display_aspect_ratio(video_path)

    if debug_frames_dir is not None or (detector_state is not None and not use_precompute_parallel):
        # Sequential: required for debug frames or (by default) detector-assisted mode
        with VideoReader(video_path) as reader:
            points, det_infos = _track_video(
                reader,
                config,
                background,
                tracking_progress_cb,
                arena_mask=arena_mask,
                crop_xyxy=crop_xyxy,
                debug_frames_dir=debug_frames_dir,
                debug_frame_interval=getattr(config, "debug_frame_interval", 30),
                debug_max_frames=getattr(config, "debug_max_frames", 100),
                detector_state=detector_state,
                display_aspect_ratio=dar,
            )
    else:
        # Parallel processing (classical-only, or detector with precomputed ROIs)
        effective_workers = parallel_workers if parallel_workers is not None else config.parallel_workers
        effective_chunk_size = chunk_size if chunk_size is not None else config.chunk_size

        # When using detector-assisted parallel mode, first build a per-frame ROI schedule
        if use_precompute_parallel and detector_state is not None:
            batch_size = getattr(
                config.detector, "detector_precompute_batch_size", 8
            )
            stride = getattr(
                config.detector, "precompute_detector_stride", 1
            )
            det_infos = _precompute_detector_infos(
                video_path,
                detector_state,
                progress_callback=detector_phase.callback if detector_phase is not None else None,
                batch_size=batch_size,
                stride=stride,
                crop_xyxy=crop_xyxy,
            )

        try:
            points = _track_video_parallel(
                video_path,
                config,
                background,
                tracking_progress_cb,
                num_workers=effective_workers,
                chunk_size=effective_chunk_size,
                arena_mask=arena_mask,
                det_infos=det_infos,
                crop_xyxy=crop_xyxy,
                display_aspect_ratio=dar,
            )
        except Exception:
            # Fallback to sequential processing on error
            with VideoReader(video_path) as reader:
                points, det_infos = _track_video(
                    reader,
                    config,
                    background,
                    tracking_progress_cb,
                    arena_mask=arena_mask,
                    crop_xyxy=crop_xyxy,
                    detector_state=detector_state,
                    display_aspect_ratio=dar,
                )

    # Post-processing: key-frame interpolation
    if config.key_frame_interpolation.enabled:
        points = _key_frame_interpolate(points, config.key_frame_interpolation)

    per_frame_path = run_dir / "per_frame.csv"
    _write_per_frame(per_frame_path, points, det_infos=det_infos)

    summary = _summarize(points, config, det_infos=det_infos)
    # Add run context for replication: command, config, and input video
    run_meta: dict[str, object] = {
        "input_video": os.path.abspath(video_path),
        **summary,
    }
    if command is not None:
        run_meta["command"] = command
    if crop_xyxy is not None:
        run_meta["crop_xyxy"] = list(crop_xyxy)
    run_meta["config"] = config.model_dump(mode="json")
    summary_path = run_dir / "tracking_summary.json"
    summary_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # Phases for overlay and heatmap (if enabled)
    overlay_phase: _PhaseProgress | None = None
    heatmap_phase: _PhaseProgress | None = None
    if global_progress is not None:
        if overlay_units > 0:
            overlay_phase = global_progress.start_phase(overlay_units)
        if heatmap_units > 0:
            heatmap_phase = global_progress.start_phase(heatmap_units)

    # Run overlay and heatmap in parallel when both are enabled; otherwise run singly
    if overlay_enabled and heatmap_enabled:
        if eff_width <= 0 or eff_height <= 0:
            raise RuntimeError("Video dimensions are not available for heatmap output")
        require_ffmpeg_available()
        overlay_path = run_dir / "overlay.mp4"
        heatmap_path = run_dir / "heatmap.png"
        overlay_scale = getattr(config, "overlay_scale", 0.25)
        heatmap_blur_ksize = getattr(config, "heatmap_blur_ksize", 51)
        with ThreadPoolExecutor(max_workers=2) as ex:
            future_overlay = ex.submit(
                _write_overlay_video,
                str(video_path),
                points,
                str(overlay_path),
                trail_length=effective_trail_length,
                scale=overlay_scale,
                crop_xyxy=crop_xyxy,
                progress_callback=overlay_phase.callback if overlay_phase is not None else None,
            )
            future_heatmap = ex.submit(
                write_heatmap_png,
                eff_width,
                eff_height,
                points,
                str(heatmap_path),
                blur_ksize=heatmap_blur_ksize,
                display_aspect_ratio=dar,
                progress_callback=heatmap_phase.callback if heatmap_phase is not None else None,
            )
            future_overlay.result()
            future_heatmap.result()
        fix_video_display_aspect_ratio(overlay_path, video_path)
        if overlay_phase is not None:
            overlay_phase.callback(overlay_units, overlay_units)
        if heatmap_phase is not None:
            heatmap_phase.callback(heatmap_units, heatmap_units)
    else:
        if overlay_enabled:
            require_ffmpeg_available()
            overlay_path = run_dir / "overlay.mp4"
            _write_overlay_video(
                str(video_path),
                points,
                str(overlay_path),
                trail_length=effective_trail_length,
                scale=getattr(config, "overlay_scale", 0.25),
                crop_xyxy=crop_xyxy,
                progress_callback=overlay_phase.callback if overlay_phase is not None else None,
            )
            fix_video_display_aspect_ratio(overlay_path, video_path)
        if heatmap_enabled:
            if eff_width <= 0 or eff_height <= 0:
                raise RuntimeError("Video dimensions are not available for heatmap output")
            heatmap_path = run_dir / "heatmap.png"
            write_heatmap_png(
                width=eff_width,
                height=eff_height,
                track_points=points,
                out_path=str(heatmap_path),
                blur_ksize=getattr(config, "heatmap_blur_ksize", 51),
                display_aspect_ratio=dar,
                progress_callback=heatmap_phase.callback if heatmap_phase is not None else None,
            )

    # Detector debug frames (sampled bbox overlays)
    if (
        detector_state is not None
        and det_infos is not None
        and config.detector.write_detector_debug_frames
    ):
        _write_detector_debug_frames(
            video_path=video_path,
            det_infos=det_infos,
            points=points,
            out_dir=run_dir / "debug_frames",
            max_frames=config.detector.detector_debug_frame_count,
            min_score=config.detector.min_score,
        )

    if global_progress is not None:
        global_progress.finalize()

    return TrackCentroidResult(run_dir=run_dir, points=points, summary=run_meta)


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
        adaptive_area=config.tracking.adaptive_area,
        adaptive_area_ratio=config.tracking.adaptive_area_ratio,
        adaptive_area_history=config.tracking.adaptive_area_history,
        use_kalman=config.tracking.use_kalman,
        kalman_process_noise=config.tracking.kalman_process_noise / (factor * factor),
        kalman_measurement_noise=config.tracking.kalman_measurement_noise / (factor * factor),
        kalman_gate_sigma=config.tracking.kalman_gate_sigma,
        kalman_coast_frames=config.tracking.kalman_coast_frames,
    )
    
    # Create a new config with scaled tracking
    return TrackCentroidConfig(
        preprocessing=config.preprocessing,
        segmentation=config.segmentation,
        morphology=config.morphology,
        tracking=scaled_tracking,
        motion_mask=config.motion_mask,
        chroma_filter=config.chroma_filter,
        arena_roi=config.arena_roi,
        key_frame_interpolation=config.key_frame_interpolation,
        detector=config.detector,
        progress_interval=config.progress_interval,
        ambiguity_confidence=config.ambiguity_confidence,
        shadow_confidence=config.shadow_confidence,
        parallel_workers=config.parallel_workers,
        chunk_size=config.chunk_size,
        downsample_factor=config.downsample_factor,
    )


def _build_background(
    reader: VideoReader,
    config: TrackCentroidConfig,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BackgroundModel | None:
    if config.preprocessing.background_model == "none":
        return None
    sampler = FrameSampler(reader)

    # For MOG2 we sample more frames for a richer pre-training set
    n_sample = config.preprocessing.background_n
    if config.preprocessing.background_model == "mog2":
        n_sample = max(n_sample, 50)

    frames = [frame for _, frame in sampler.sample(n_sample, progress_callback=progress_callback)]

    # Crop to arena box before downsampling when arena crop is enabled
    if crop_xyxy is not None:
        frames = [crop_frame(f, crop_xyxy) for f in frames]

    # Downsample frames if downsampling is enabled
    if config.downsample_factor is not None and config.downsample_factor > 1.0:
        frames = [_downsample_frame(frame, config.downsample_factor) for frame in frames]

    return build_background(frames, config.preprocessing)


def _interpolate_roi(
    roi_a: tuple[int, int, int, int],
    roi_b: tuple[int, int, int, int],
    t: float,
) -> tuple[int, int, int, int]:
    """Linear interpolation between two ROIs (x1, y1, x2, y2). t in [0, 1]."""
    x1 = int(round(roi_a[0] + t * (roi_b[0] - roi_a[0])))
    y1 = int(round(roi_a[1] + t * (roi_b[1] - roi_a[1])))
    x2 = int(round(roi_a[2] + t * (roi_b[2] - roi_a[2])))
    y2 = int(round(roi_a[3] + t * (roi_b[3] - roi_a[3])))
    return (x1, y1, x2, y2)


def _precompute_detector_infos(
    video_path: Path,
    detector_state: DetectorState,
    progress_callback: Callable[[int, int], None] | None = None,
    batch_size: int = 8,
    stride: int = 1,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> list[FrameDetectorInfo]:
    """Run the detector in a pre-pass to build a per-frame ROI schedule.

    When batch_size > 1, runs batched inference for speed. When stride > 1,
    runs detector only every stride frames and interpolates ROIs in between.
    """
    detector = detector_state._detector
    has_batch = hasattr(detector, "detect_batch") and batch_size > 1

    # Pass 1: collect frames where we should run detector (optionally strided); run in batches
    stored_infos: dict[int, FrameDetectorInfo] = {}
    buffer: list[tuple[int, np.ndarray, tuple[int, int]]] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        frames = [b[1] for b in buffer]
        if has_batch:
            results = detector.detect_batch(frames)
            for (idx, _, frame_hw), result in zip(buffer, results):
                info = detector_state.apply_detection_result(result, frame_hw)
                stored_infos[idx] = info
        else:
            for (idx, frame_bgr, _) in buffer:
                info = detector_state.step(
                    frame_bgr=frame_bgr,
                    frame_idx=idx,
                    tracking_conf=1.0,
                    has_centroid=True,
                )
                stored_infos[idx] = info
        buffer.clear()

    with VideoReader(video_path) as reader:
        total = reader.frame_count
        for idx, frame in reader.iter_frames():
            if crop_xyxy is not None:
                frame = crop_frame(frame, crop_xyxy)
            if detector_state.should_run(idx, 1.0, True) and (
                stride <= 1 or idx % stride == 0
            ):
                buffer.append((idx, frame.copy(), frame.shape[:2]))
                if len(buffer) >= batch_size:
                    flush_buffer()
            if progress_callback is not None and total > 0:
                progress_callback(idx + 1, total)
        flush_buffer()

    detector_frame_indices = sorted(stored_infos.keys()) if stored_infos else []

    # Pass 2: build det_infos in frame order; interpolate ROI when stride > 1
    det_infos = []
    with VideoReader(video_path) as reader:
        for idx, frame in reader.iter_frames():
            if crop_xyxy is not None:
                frame = crop_frame(frame, crop_xyxy)
            frame_hw = frame.shape[:2]
            if idx in stored_infos:
                det_infos.append(stored_infos[idx])
            else:
                detector_state.frames_since_detect += 1
                if stride > 1 and len(detector_frame_indices) >= 2:
                    # Interpolate between nearest detector frames
                    prev_list = [i for i in detector_frame_indices if i < idx]
                    next_list = [i for i in detector_frame_indices if i > idx]
                    prev_idx = max(prev_list) if prev_list else None
                    next_idx = min(next_list) if next_list else None
                    if prev_idx is not None and next_idx is not None:
                        roi_prev = stored_infos[prev_idx].roi_xyxy
                        roi_next = stored_infos[next_idx].roi_xyxy
                        if roi_prev is not None and roi_next is not None:
                            t = (idx - prev_idx) / (next_idx - prev_idx)
                            interp = _interpolate_roi(roi_prev, roi_next, t)
                            det_infos.append(
                                FrameDetectorInfo(
                                    roi_xyxy=interp,
                                    detector_used=False,
                                )
                            )
                        else:
                            det_infos.append(
                                detector_state.info_for_frame_without_run(
                                    frame_hw
                                )
                            )
                    elif prev_idx is not None and stored_infos[prev_idx].roi_xyxy is not None:
                        det_infos.append(
                            FrameDetectorInfo(
                                roi_xyxy=stored_infos[prev_idx].roi_xyxy,
                                detector_used=False,
                            )
                        )
                    elif next_idx is not None and stored_infos[next_idx].roi_xyxy is not None:
                        det_infos.append(
                            FrameDetectorInfo(
                                roi_xyxy=stored_infos[next_idx].roi_xyxy,
                                detector_used=False,
                            )
                        )
                    else:
                        det_infos.append(
                            detector_state.info_for_frame_without_run(
                                frame_hw
                            )
                        )
                else:
                    det_infos.append(
                        detector_state.info_for_frame_without_run(frame_hw)
                    )
    return det_infos


def _build_circle_arena_mask(
    crop_xyxy: tuple[int, int, int, int],
    chosen_circle: tuple[int, int, int],
    orig_width: int,
    orig_height: int,
    downsample_factor: float | None,
) -> np.ndarray:
    """Build a binary mask (255 inside circle) at processing resolution from detected Hough circle.

    chosen_circle is (cx, cy, r) in full-frame coordinates; crop_xyxy is (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = crop_xyxy
    cx, cy, r = chosen_circle
    # Circle center and radius in cropped frame
    cx_crop = cx - x1
    cy_crop = cy - y1
    if downsample_factor is not None and downsample_factor > 1.0:
        proc_w = int(orig_width / downsample_factor)
        proc_h = int(orig_height / downsample_factor)
        cx_p = cx_crop / downsample_factor
        cy_p = cy_crop / downsample_factor
        r_p = r / downsample_factor
    else:
        proc_w = orig_width
        proc_h = orig_height
        cx_p, cy_p, r_p = float(cx_crop), float(cy_crop), float(r)
    mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx_p)), int(round(cy_p))), int(round(r_p)), 255, -1)
    return mask


def _load_or_build_arena_mask(
    config: TrackCentroidConfig,
    orig_height: int,
    orig_width: int,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Load or build the arena ROI mask at processing resolution.

    orig_height/orig_width are the effective frame dimensions (cropped when
    crop_xyxy is set). When crop_xyxy is set, geometric ROI params are
    converted from full-frame to cropped coordinates.

    The returned mask is a single-channel ``uint8`` image where 255 marks
    pixels *inside* the arena.  It is sized to the processing resolution
    (i.e. already accounting for ``downsample_factor``).

    Returns ``None`` when arena ROI is disabled.
    """
    roi = config.arena_roi
    if not roi.enabled:
        return None

    factor = config.downsample_factor
    if factor is not None and factor > 1.0:
        proc_w = int(orig_width / factor)
        proc_h = int(orig_height / factor)
    else:
        factor = None  # normalise so later checks are simpler
        proc_w = orig_width
        proc_h = orig_height

    offset_x = crop_xyxy[0] if crop_xyxy is not None else 0
    offset_y = crop_xyxy[1] if crop_xyxy is not None else 0

    # --- mask from image file ---
    if roi.mask_path is not None:
        raw = cv2.imread(roi.mask_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Arena mask image not found: {roi.mask_path}")
        if crop_xyxy is not None:
            raw = raw[crop_xyxy[1]:crop_xyxy[3], crop_xyxy[0]:crop_xyxy[2]]
        if raw.shape[:2] != (proc_h, proc_w):
            raw = cv2.resize(raw, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
        return mask

    # --- geometric shape (params in full-frame coords; convert to cropped when applicable) ---
    if roi.kind is not None and roi.params is not None:
        mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        p = roi.params

        if roi.kind == "CIRCLE":
            cx, cy, r = p["center_x"] - offset_x, p["center_y"] - offset_y, p["radius"]
            if factor is not None:
                cx, cy, r = cx / factor, cy / factor, r / factor
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)

        elif roi.kind == "RECT":
            x, y, w, h = p["x"] - offset_x, p["y"] - offset_y, p["w"], p["h"]
            if factor is not None:
                x, y, w, h = x / factor, y / factor, w / factor, h / factor
            x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        return mask

    # Shouldn't reach here due to schema validator, but be defensive.
    raise ValueError("Arena ROI is enabled but no mask_path or kind+params provided.")


def _track_video(
    reader: VideoReader,
    config: TrackCentroidConfig,
    background: BackgroundModel | None,
    progress_callback: Callable[[int, int], None] | None,
    arena_mask: np.ndarray | None = None,
    *,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    debug_frames_dir: Path | None = None,
    debug_frame_interval: int = 30,
    debug_max_frames: int | None = 100,
    detector_state: DetectorState | None = None,
    display_aspect_ratio: str | None = None,
) -> tuple[list[TrackPoint], list[FrameDetectorInfo] | None]:
    points: list[TrackPoint] = []
    det_infos: list[FrameDetectorInfo] | None = [] if detector_state is not None else None
    previous: TrackPoint | None = None
    ema_point: tuple[float, float] | None = None
    total = reader.frame_count
    
    # Create scaled config if downsampling is enabled
    effective_config = _create_scaled_tracking_config(config)
    downsample_factor = config.downsample_factor if config.downsample_factor is not None and config.downsample_factor > 1.0 else None

    # --- NEW: initialise motion / Kalman / adaptive-area state -----------
    motion: MotionAccumulator | None = None
    if effective_config.motion_mask.enabled:
        mc = effective_config.motion_mask
        motion = MotionAccumulator(
            history_len=mc.history_len,
            threshold=mc.threshold,
            dilate_ksize=mc.dilate_ksize,
            dilate_iters=mc.dilate_iters,
        )

    kalman: KalmanPointTracker | None = None
    if effective_config.tracking.use_kalman:
        tc = effective_config.tracking
        kalman = KalmanPointTracker(
            process_noise=tc.kalman_process_noise,
            measurement_noise=tc.kalman_measurement_noise,
            gate_sigma=tc.kalman_gate_sigma,
        )

    area_filter: AdaptiveAreaFilter | None = None
    if effective_config.tracking.adaptive_area:
        area_filter = AdaptiveAreaFilter(
            ratio=effective_config.tracking.adaptive_area_ratio,
            history_len=effective_config.tracking.adaptive_area_history,
        )
    # ---------------------------------------------------------------------

    coast_limit = effective_config.tracking.kalman_coast_frames

    debug_sink: list[TrackFrameDebug] = []
    debug_frames_written = 0

    # Tracking confidence for detector scheduling (previous frame's)
    last_tracking_conf: float = 0.0
    last_has_centroid: bool = False

    # Start after first detection: only emit non-null points once detector finds mouse
    tracking_started: bool = False
    if detector_state is not None and config.detector.start_tracking_after_first_detection:
        tracking_started = False
    else:
        tracking_started = True  # no deferral when detector is off or option disabled

    for idx, frame in reader.iter_frames():
        # Crop to arena box first (before detector and pipeline)
        if crop_xyxy is not None:
            frame = crop_frame(frame, crop_xyxy)
        # --- Detector-assisted ROI (runs on full-res frame BEFORE downsampling) ---
        det_info: FrameDetectorInfo | None = None
        roi_xyxy: tuple[int, int, int, int] | None = None
        if detector_state is not None:
            det_info = detector_state.step(
                frame_bgr=frame,
                frame_idx=idx,
                tracking_conf=last_tracking_conf,
                has_centroid=last_has_centroid,
            )
            roi_xyxy = det_info.roi_xyxy

        # --- Detector-first early exit (when detector enabled) ---
        # 1. Detector ran and found mouse with confidence -> use bbox center, skip pipeline
        # 2. Detector ran and found nothing, waiting for first detection -> emit null, skip pipeline
        point: TrackPoint | None = None
        if detector_state is not None and det_info is not None and det_info.detector_used:
            if (
                det_info.detector_score is not None
                and det_info.detector_score >= config.detector.min_score
                and roi_xyxy is not None
            ):
                # Use detector bbox center directly (original video coords)
                cx = (roi_xyxy[0] + roi_xyxy[2]) / 2.0
                cy = (roi_xyxy[1] + roi_xyxy[3]) / 2.0
                point = TrackPoint(
                    frame_idx=idx,
                    x=cx,
                    y=cy,
                    area=None,
                    confidence=det_info.detector_score,
                    flags=["DETECTOR_DIRECT"],
                )
                tracking_started = True
            elif (
                config.detector.start_tracking_after_first_detection
                and not tracking_started
            ):
                # No confident detection, still waiting for first -> emit null, skip pipeline
                point = TrackPoint(
                    frame_idx=idx,
                    x=None,
                    y=None,
                    area=None,
                    confidence=0.0,
                    flags=[],
                )

        # Full pipeline only when we didn't take a detector-first shortcut
        if point is None:
            # Downsample frame if downsampling is enabled
            if downsample_factor is not None:
                frame = _downsample_frame(frame, downsample_factor)
                # Scale ROI to downsampled space
                if roi_xyxy is not None:
                    roi_xyxy = (
                        int(roi_xyxy[0] / downsample_factor),
                        int(roi_xyxy[1] / downsample_factor),
                        int(roi_xyxy[2] / downsample_factor),
                        int(roi_xyxy[3] / downsample_factor),
                    )

            # Raw grayscale (original intensity — for candidate scoring)
            raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

            gray = preprocess_frame(frame, effective_config.preprocessing, background)
            mask = segment_frame(gray, effective_config.segmentation, effective_config.morphology)

            # Apply arena ROI mask — discard foreground outside the arena
            if arena_mask is not None:
                mask = cv2.bitwise_and(mask, arena_mask)

            # Chrominance / luminance filter — suppress shadows while keeping
            # objects that differ from the background in *either* colour or
            # brightness.  A shadow has the same hue AND a small brightness
            # change; a real object (the mouse) will differ in at least one.
            if (
                effective_config.chroma_filter.enabled
                and background is not None
                and background.image_bgr is not None
                and frame.ndim == 3
            ):
                lab_f = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                lab_bg = cv2.cvtColor(background.image_bgr, cv2.COLOR_BGR2Lab)
                chroma_diff = cv2.max(
                    cv2.absdiff(lab_f[:, :, 1], lab_bg[:, :, 1]),
                    cv2.absdiff(lab_f[:, :, 2], lab_bg[:, :, 2]),
                )
                luma_diff = cv2.absdiff(lab_f[:, :, 0], lab_bg[:, :, 0])
                # Keep pixel if chrominance differs OR luminance differs a lot
                _, chroma_ok = cv2.threshold(
                    chroma_diff,
                    effective_config.chroma_filter.threshold,
                    255,
                    cv2.THRESH_BINARY,
                )
                _, luma_ok = cv2.threshold(
                    luma_diff,
                    effective_config.chroma_filter.luma_threshold,
                    255,
                    cv2.THRESH_BINARY,
                )
                chroma_mask = cv2.bitwise_or(chroma_ok, luma_ok)
                mask = cv2.bitwise_and(mask, chroma_mask)

            # Apply motion mask — discard static foreground
            if motion is not None:
                motion_mask = motion.update(gray)
                # Kalman search window: allow detection near predicted position
                # even when the mouse is stationary (motion mask would suppress it)
                if kalman is not None and kalman.initialized:
                    pred = kalman.predicted_position
                    if pred is not None:
                        r = int(kalman.search_radius())
                        cv2.circle(
                            motion_mask,
                            (int(round(pred[0])), int(round(pred[1]))),
                            r, 255, -1,
                        )
                mask = cv2.bitwise_and(mask, motion_mask)

            # --- Crop mask and gray to detector ROI (if available) ---
            roi_offset_x = 0
            roi_offset_y = 0
            if roi_xyxy is not None:
                rx1, ry1, rx2, ry2 = roi_xyxy
                # Ensure valid crop region
                if rx2 > rx1 and ry2 > ry1:
                    mask = mask[ry1:ry2, rx1:rx2]
                    raw_gray = raw_gray[ry1:ry2, rx1:rx2]
                    roi_offset_x = rx1
                    roi_offset_y = ry1

            # Kalman predict (before track_frame so gating uses the prediction)
            # Adjust Kalman prediction to ROI space if we have a detector ROI
            if kalman is not None and kalman.initialized:
                kalman.predict()

            # When using ROI, adjust 'previous' to ROI-local coordinates for tracking
            roi_previous = previous
            if roi_xyxy is not None and previous is not None and previous.x is not None and previous.y is not None:
                roi_previous = TrackPoint(
                    frame_idx=previous.frame_idx,
                    x=previous.x - roi_offset_x,
                    y=previous.y - roi_offset_y,
                    area=previous.area,
                    confidence=previous.confidence,
                    flags=previous.flags,
                )

            if debug_frames_dir is not None:
                debug_sink.clear()
            point = track_frame(
                mask,
                frame_idx=idx,
                tracking=effective_config.tracking,
                previous=roi_previous,
                ambiguity_confidence=effective_config.ambiguity_confidence,
                shadow_confidence=effective_config.shadow_confidence,
                gray_frame=raw_gray,
                kalman=kalman if roi_xyxy is None else None,  # skip Kalman gating inside ROI
                adaptive_area=area_filter,
                debug_sink=debug_sink if debug_frames_dir is not None else None,
            )

            # Convert ROI-local coords back to global (processing-resolution)
            if roi_xyxy is not None and point.x is not None and point.y is not None:
                point = TrackPoint(
                    frame_idx=point.frame_idx,
                    x=point.x + roi_offset_x,
                    y=point.y + roi_offset_y,
                    area=point.area,
                    confidence=point.confidence,
                    flags=point.flags,
                )

            if (
                debug_frames_dir is not None
                and debug_sink
                and (idx % debug_frame_interval == 0)
                and (debug_max_frames is None or debug_frames_written < debug_max_frames)
            ):
                # Use full frame; when mask is cropped to ROI, expand it and pass offset for blob coords
                debug_mask = mask
                off_x, off_y = 0, 0
                if roi_xyxy is not None:
                    rx1, ry1, rx2, ry2 = roi_xyxy
                    if rx2 > rx1 and ry2 > ry1:
                        h, w = frame.shape[:2]
                        debug_mask = np.zeros((h, w), dtype=mask.dtype)
                        debug_mask[ry1:ry2, rx1:rx2] = mask
                        off_x, off_y = rx1, ry1
                debug_img = render_debug_frame(
                    frame, debug_mask, debug_sink[-1],
                    roi_offset_x=off_x, roi_offset_y=off_y,
                )
                if crop_xyxy is not None:
                    cw = crop_xyxy[2] - crop_xyxy[0]
                    ch = crop_xyxy[3] - crop_xyxy[1]
                    cv2.putText(
                        debug_img,
                        f"Arena crop: {cw}x{ch}",
                        (8, debug_img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                if display_aspect_ratio:
                    debug_img = resize_to_display_aspect(debug_img, display_aspect_ratio)
                out_path = debug_frames_dir / f"frame_{idx:06d}.png"
                if cv2.imwrite(str(out_path), debug_img):
                    debug_frames_written += 1

            # --- Adjust confidence based on detector state ---
            if detector_state is not None and point.x is not None:
                extra_flags = list(point.flags)
                conf = point.confidence
                if det_info is not None and det_info.detector_used:
                    if det_info.detector_score is not None and det_info.detector_score >= config.detector.min_score:
                        pass  # fresh, good detection: keep full confidence
                    elif det_info.detector_score is not None and det_info.detector_score < config.detector.min_score:
                        conf *= 0.85
                        if "DETECTOR_LOW_SCORE" not in extra_flags:
                            extra_flags.append("DETECTOR_LOW_SCORE")
                elif detector_state.frames_since_detect > config.detector.every_n_frames * 2:
                    conf *= 0.85
                    if "DETECTOR_STALE_ROI" not in extra_flags:
                        extra_flags.append("DETECTOR_STALE_ROI")

                if extra_flags != point.flags or conf != point.confidence:
                    point = TrackPoint(
                        frame_idx=point.frame_idx,
                        x=point.x,
                        y=point.y,
                        area=point.area,
                        confidence=conf,
                        flags=extra_flags,
                    )

            # --- Start-after-first-detection: null out points until detector finds mouse ---
            if not tracking_started:
                if (
                    det_info is not None
                    and det_info.detector_used
                    and det_info.detector_score is not None
                    and det_info.detector_score >= config.detector.min_score
                ):
                    tracking_started = True
                else:
                    point = TrackPoint(
                        frame_idx=idx,
                        x=None,
                        y=None,
                        area=None,
                        confidence=0.0,
                        flags=[],
                    )

            # --- Update Kalman / adaptive area in processing space -----------
            if point.x is not None and point.y is not None:
                if kalman is not None:
                    if not kalman.initialized:
                        kalman.initialize(point.x, point.y)
                    else:
                        cx, cy = kalman.update(point.x, point.y)
                        # Use Kalman-corrected position (smooths jitter)
                        point = TrackPoint(
                            frame_idx=point.frame_idx,
                            x=cx,
                            y=cy,
                            area=point.area,
                            confidence=point.confidence,
                            flags=point.flags,
                        )
                if area_filter is not None and point.area is not None:
                    area_filter.update(point.area)
            else:
                if kalman is not None and kalman.initialized:
                    kalman.mark_no_measurement()
                    # --- Kalman coasting: emit predicted position -----------
                    # Do NOT coast on black/empty frames: no foreground = no valid info
                    mask_foreground_px = cv2.countNonZero(mask)
                    min_foreground_for_coast = effective_config.tracking.min_area_px
                    if (
                        coast_limit > 0
                        and kalman.frames_without_measurement <= coast_limit
                        and mask_foreground_px >= min_foreground_for_coast
                    ):
                        pred = kalman.predicted_position
                        if pred is not None:
                            coast_conf = max(
                                0.30,
                                0.85 - 0.01 * kalman.frames_without_measurement,
                            )
                            point = TrackPoint(
                                frame_idx=idx,
                                x=pred[0],
                                y=pred[1],
                                area=previous.area if previous is not None and previous.area is not None else None,
                                confidence=coast_conf,
                                flags=["KALMAN_COAST"],
                            )
                    # --------------------------------------------------------
            # -----------------------------------------------------------------

        # Scale coordinates back to original resolution (skip for DETECTOR_DIRECT - already original)
        if (
            downsample_factor is not None
            and point.x is not None
            and point.y is not None
            and "DETECTOR_DIRECT" not in point.flags
        ):
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
        elif point.x is not None and point.y is not None:
            # No downsampling, or DETECTOR_DIRECT (already in original coords)
            if downsample_factor is not None and "DETECTOR_DIRECT" in point.flags:
                # Keep previous in processing (downsampled) space for next frame
                previous = TrackPoint(
                    frame_idx=point.frame_idx,
                    x=point.x / downsample_factor,
                    y=point.y / downsample_factor,
                    area=point.area,
                    confidence=point.confidence,
                    flags=point.flags,
                )
            else:
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
        if det_infos is not None:
            det_infos.append(det_info if det_info is not None else FrameDetectorInfo())

        # Update tracking state for detector scheduling
        last_tracking_conf = point.confidence
        last_has_centroid = point.x is not None and point.y is not None

        if progress_callback and (idx + 1) % config.progress_interval == 0:
            progress_callback(idx + 1, total)

    if progress_callback:
        progress_callback(total, total)
    return points, det_infos


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


def _compute_first_detection_frame(
    det_infos: list[FrameDetectorInfo],
    min_score: float,
) -> int | None:
    """Return the first frame index with a valid detector detection, or None if never."""
    for i, di in enumerate(det_infos):
        if (
            di.detector_used
            and di.detector_score is not None
            and di.detector_score >= min_score
        ):
            return i
    return None


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
    arena_mask: np.ndarray | None = None,
    det_infos: list[FrameDetectorInfo] | None = None,
    first_detection_frame: int | None = None,
    crop_xyxy: tuple[int, int, int, int] | None = None,
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
        arena_mask: Optional arena ROI mask at processing resolution (255 = inside).
        first_detection_frame: If set, frames with idx < this are emitted as null (start-after-first-detection).

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

    # --- NEW: per-chunk state for motion / Kalman / adaptive-area --------
    motion: MotionAccumulator | None = None
    if effective_config.motion_mask.enabled:
        mc = effective_config.motion_mask
        motion = MotionAccumulator(
            history_len=mc.history_len,
            threshold=mc.threshold,
            dilate_ksize=mc.dilate_ksize,
            dilate_iters=mc.dilate_iters,
        )

    kalman: KalmanPointTracker | None = None
    if effective_config.tracking.use_kalman:
        tc = effective_config.tracking
        kalman = KalmanPointTracker(
            process_noise=tc.kalman_process_noise,
            measurement_noise=tc.kalman_measurement_noise,
            gate_sigma=tc.kalman_gate_sigma,
        )

    area_filter: AdaptiveAreaFilter | None = None
    if effective_config.tracking.adaptive_area:
        area_filter = AdaptiveAreaFilter(
            ratio=effective_config.tracking.adaptive_area_ratio,
            history_len=effective_config.tracking.adaptive_area_history,
        )
    # ---------------------------------------------------------------------

    coast_limit = effective_config.tracking.kalman_coast_frames

    with VideoReader(video_path) as reader:
        frame_count = 0
        for idx, frame in reader.iter_frames(start_frame=chunk_start, end_frame=chunk_end):
            if crop_xyxy is not None:
                frame = crop_frame(frame, crop_xyxy)
            # Optional detector ROI (full-resolution coords from pre-pass)
            roi_xyxy: tuple[int, int, int, int] | None = None
            di: FrameDetectorInfo | None = None
            if det_infos is not None and idx < len(det_infos):
                di = det_infos[idx]
                roi_xyxy = di.roi_xyxy

            # --- Detector-first early exit (when detector enabled) ---
            point: TrackPoint | None = None
            if (
                det_infos is not None
                and di is not None
                and di.detector_used
                and config.detector.enabled
            ):
                if (
                    di.detector_score is not None
                    and di.detector_score >= config.detector.min_score
                    and roi_xyxy is not None
                ):
                    # Use detector bbox center directly (original video coords)
                    roi_full = det_infos[idx].roi_xyxy  # full-res before we scale
                    if roi_full is not None:
                        cx = (roi_full[0] + roi_full[2]) / 2.0
                        cy = (roi_full[1] + roi_full[3]) / 2.0
                        point = TrackPoint(
                            frame_idx=idx,
                            x=cx,
                            y=cy,
                            area=None,
                            confidence=di.detector_score,
                            flags=["DETECTOR_DIRECT"],
                        )
                elif (
                    config.detector.start_tracking_after_first_detection
                    and first_detection_frame is not None
                    and idx < first_detection_frame
                ):
                    # No confident detection, still waiting for first -> emit null, skip pipeline
                    point = TrackPoint(
                        frame_idx=idx,
                        x=None,
                        y=None,
                        area=None,
                        confidence=0.0,
                        flags=[],
                    )

            # Full pipeline only when we didn't take a detector-first shortcut
            if point is None:
                # Downsample frame if downsampling is enabled
                if downsample_factor is not None:
                    frame = _downsample_frame(frame, downsample_factor)
                    # Scale ROI to downsampled space
                    if roi_xyxy is not None:
                        roi_xyxy = (
                            int(roi_xyxy[0] / downsample_factor),
                            int(roi_xyxy[1] / downsample_factor),
                            int(roi_xyxy[2] / downsample_factor),
                            int(roi_xyxy[3] / downsample_factor),
                        )

                # Raw grayscale (original intensity — for candidate scoring)
                raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

                gray = preprocess_frame(frame, effective_config.preprocessing, background)
                mask = segment_frame(gray, effective_config.segmentation, effective_config.morphology)

                # Apply arena ROI mask — discard foreground outside the arena
                if arena_mask is not None:
                    mask = cv2.bitwise_and(mask, arena_mask)

                # Chrominance / luminance filter — suppress shadows while keeping
                # objects that differ from the background in colour or brightness.
                if (
                    effective_config.chroma_filter.enabled
                    and background is not None
                    and background.image_bgr is not None
                    and frame.ndim == 3
                ):
                    lab_f = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                    lab_bg = cv2.cvtColor(background.image_bgr, cv2.COLOR_BGR2Lab)
                    chroma_diff = cv2.max(
                        cv2.absdiff(lab_f[:, :, 1], lab_bg[:, :, 1]),
                        cv2.absdiff(lab_f[:, :, 2], lab_bg[:, :, 2]),
                    )
                    luma_diff = cv2.absdiff(lab_f[:, :, 0], lab_bg[:, :, 0])
                    # Keep pixel if chrominance differs OR luminance differs a lot
                    _, chroma_ok = cv2.threshold(
                        chroma_diff,
                        effective_config.chroma_filter.threshold,
                        255,
                        cv2.THRESH_BINARY,
                    )
                    _, luma_ok = cv2.threshold(
                        luma_diff,
                        effective_config.chroma_filter.luma_threshold,
                        255,
                        cv2.THRESH_BINARY,
                    )
                    chroma_mask = cv2.bitwise_or(chroma_ok, luma_ok)
                    mask = cv2.bitwise_and(mask, chroma_mask)

                # --- Crop mask and gray to detector ROI (if available) ---
                roi_offset_x = 0
                roi_offset_y = 0
                if roi_xyxy is not None:
                    rx1, ry1, rx2, ry2 = roi_xyxy
                    # Ensure valid crop region
                    if rx2 > rx1 and ry2 > ry1:
                        mask = mask[ry1:ry2, rx1:rx2]
                        raw_gray = raw_gray[ry1:ry2, rx1:rx2]
                        roi_offset_x = rx1
                        roi_offset_y = ry1

                # Apply motion mask — discard static foreground
                if motion is not None:
                    motion_mask = motion.update(gray)
                    # Kalman search window: keep detection near predicted pos
                    if kalman is not None and kalman.initialized:
                        pred = kalman.predicted_position
                        if pred is not None:
                            r = int(kalman.search_radius())
                            cv2.circle(
                                motion_mask,
                                (int(round(pred[0])), int(round(pred[1]))),
                                r, 255, -1,
                            )
                    mask = cv2.bitwise_and(mask, motion_mask)

                # Kalman predict (before track_frame)
                if kalman is not None and kalman.initialized:
                    kalman.predict()

                # When using ROI, adjust 'previous' to ROI-local coordinates for tracking
                roi_previous = previous
                if roi_xyxy is not None and previous is not None and previous.x is not None and previous.y is not None:
                    roi_previous = TrackPoint(
                        frame_idx=previous.frame_idx,
                        x=previous.x - roi_offset_x,
                        y=previous.y - roi_offset_y,
                        area=previous.area,
                        confidence=previous.confidence,
                        flags=previous.flags,
                    )

                point = track_frame(
                    mask,
                    frame_idx=idx,
                    tracking=effective_config.tracking,
                    previous=roi_previous,
                    ambiguity_confidence=effective_config.ambiguity_confidence,
                    shadow_confidence=effective_config.shadow_confidence,
                    gray_frame=raw_gray,
                    kalman=kalman if roi_xyxy is None else None,
                    adaptive_area=area_filter,
                )

                # Convert ROI-local coords back to global (processing-resolution)
                if roi_xyxy is not None and point.x is not None and point.y is not None:
                    point = TrackPoint(
                        frame_idx=point.frame_idx,
                        x=point.x + roi_offset_x,
                        y=point.y + roi_offset_y,
                        area=point.area,
                        confidence=point.confidence,
                        flags=point.flags,
                    )

                # --- Update Kalman / adaptive area in processing space -------
                if point.x is not None and point.y is not None:
                    if kalman is not None:
                        if not kalman.initialized:
                            kalman.initialize(point.x, point.y)
                        else:
                            cx, cy = kalman.update(point.x, point.y)
                            point = TrackPoint(
                                frame_idx=point.frame_idx,
                                x=cx,
                                y=cy,
                                area=point.area,
                                confidence=point.confidence,
                                flags=point.flags,
                            )
                    if area_filter is not None and point.area is not None:
                        area_filter.update(point.area)
                else:
                    if kalman is not None and kalman.initialized:
                        kalman.mark_no_measurement()
                        # --- Kalman coasting ---
                        # Do NOT coast on black/empty frames: no foreground = no valid info
                        mask_foreground_px = cv2.countNonZero(mask)
                        min_foreground_for_coast = effective_config.tracking.min_area_px
                        if (
                            coast_limit > 0
                            and kalman.frames_without_measurement <= coast_limit
                            and mask_foreground_px >= min_foreground_for_coast
                        ):
                            pred = kalman.predicted_position
                            if pred is not None:
                                coast_conf = max(
                                    0.30,
                                    0.85 - 0.01 * kalman.frames_without_measurement,
                                )
                                point = TrackPoint(
                                    frame_idx=idx,
                                    x=pred[0],
                                    y=pred[1],
                                    area=previous.area if previous is not None and previous.area is not None else None,
                                    confidence=coast_conf,
                                    flags=["KALMAN_COAST"],
                                )
                        # -----------------------
                # -------------------------------------------------------------

            # Scale coordinates back to original resolution (skip for DETECTOR_DIRECT - already original)
            if (
                downsample_factor is not None
                and point.x is not None
                and point.y is not None
                and "DETECTOR_DIRECT" not in point.flags
            ):
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
            elif point.x is not None and point.y is not None:
                # No downsampling, or DETECTOR_DIRECT (already in original coords)
                if downsample_factor is not None and "DETECTOR_DIRECT" in point.flags:
                    previous = TrackPoint(
                        frame_idx=point.frame_idx,
                        x=point.x / downsample_factor,
                        y=point.y / downsample_factor,
                        area=point.area,
                        confidence=point.confidence,
                        flags=point.flags,
                    )
                else:
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
            # Start-after-first-detection: null out points before first detector hit
            if (
                first_detection_frame is not None
                and idx < first_detection_frame
                and point.x is not None
                and point.y is not None
            ):
                point = TrackPoint(
                    frame_idx=idx,
                    x=None,
                    y=None,
                    area=None,
                    confidence=0.0,
                    flags=[],
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


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Catmull-Rom spline interpolation between *p1* and *p2* at parameter *t* in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _key_frame_interpolate(
    points: list[TrackPoint],
    config: KeyFrameInterpolationConfig,
) -> list[TrackPoint]:
    """Post-process tracking points using key-frame interpolation.

    Key frames are high-confidence, non-ambiguous frames.  Between consecutive
    key frames that are at most ``max_gap_frames`` apart, NO_DETECTION gaps are
    filled and (optionally) points deviating too far from the interpolated path
    are replaced.

    Args:
        points: Raw (or EMA-smoothed) tracking points.
        config: Key-frame interpolation configuration.

    Returns:
        A new list of ``TrackPoint`` with interpolated values where appropriate.
    """
    if not points or not config.enabled:
        return points

    n = len(points)

    # 1. Identify key-frame indices
    key_indices: list[int] = []
    for i, p in enumerate(points):
        if p.x is None or p.y is None:
            continue
        if p.confidence < config.min_confidence:
            continue
        if "AMBIGUOUS_TARGET" in p.flags:
            continue
        key_indices.append(i)

    if len(key_indices) < 2:
        return points  # not enough anchors

    # 2. Build result as a mutable copy
    result = list(points)

    # 3. Walk consecutive key-frame pairs and interpolate the gaps
    for ki in range(len(key_indices) - 1):
        i_start = key_indices[ki]
        i_end = key_indices[ki + 1]
        gap = i_end - i_start - 1

        if gap <= 0 or gap > config.max_gap_frames:
            continue

        kf_s = points[i_start]
        kf_e = points[i_end]

        # For cubic Catmull-Rom we need the flanking key frames (p0, p3).
        use_cubic = config.method == "cubic" and len(key_indices) >= 2
        if use_cubic:
            if ki > 0:
                p0_idx = key_indices[ki - 1]
                p0_x, p0_y = points[p0_idx].x, points[p0_idx].y
            else:
                # Mirror about kf_s
                p0_x = 2.0 * kf_s.x - kf_e.x  # type: ignore[operator]
                p0_y = 2.0 * kf_s.y - kf_e.y  # type: ignore[operator]

            if ki + 2 < len(key_indices):
                p3_idx = key_indices[ki + 2]
                p3_x, p3_y = points[p3_idx].x, points[p3_idx].y
            else:
                p3_x = 2.0 * kf_e.x - kf_s.x  # type: ignore[operator]
                p3_y = 2.0 * kf_e.y - kf_s.y  # type: ignore[operator]

        for k in range(i_start + 1, i_end):
            orig = points[k]
            t = (k - i_start) / (i_end - i_start)

            if use_cubic:
                interp_x = _catmull_rom(p0_x, kf_s.x, kf_e.x, p3_x, t)  # type: ignore[arg-type]
                interp_y = _catmull_rom(p0_y, kf_s.y, kf_e.y, p3_y, t)  # type: ignore[arg-type]
            else:
                interp_x = kf_s.x + t * (kf_e.x - kf_s.x)  # type: ignore[operator]
                interp_y = kf_s.y + t * (kf_e.y - kf_s.y)  # type: ignore[operator]

            # Decide whether to replace this frame
            should_replace = False
            if orig.x is None or orig.y is None:
                should_replace = True
            elif config.max_deviation_px is not None:
                dev = float(np.hypot(orig.x - interp_x, orig.y - interp_y))
                if dev > config.max_deviation_px:
                    should_replace = True

            if should_replace:
                # Interpolate area between key frames
                if kf_s.area is not None and kf_e.area is not None:
                    interp_area = kf_s.area + t * (kf_e.area - kf_s.area)
                else:
                    interp_area = orig.area

                flags = [f for f in (orig.flags or []) if f != "NO_DETECTION"]
                if "INTERPOLATED" not in flags:
                    flags.append("INTERPOLATED")

                interp_conf = min(kf_s.confidence, kf_e.confidence) * 0.8

                result[k] = TrackPoint(
                    frame_idx=orig.frame_idx,
                    x=interp_x,
                    y=interp_y,
                    area=interp_area,
                    confidence=interp_conf,
                    flags=flags,
                )

    return result


def _track_video_parallel(
    video_path: Path,
    config: TrackCentroidConfig,
    background: BackgroundModel | None,
    progress_callback: Callable[[int, int], None] | None,
    *,
    num_workers: int | None = None,
    chunk_size: int = 200,
    arena_mask: np.ndarray | None = None,
    det_infos: list[FrameDetectorInfo] | None = None,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    display_aspect_ratio: str | None = None,
) -> list[TrackPoint]:
    """Track video frames using parallel chunk processing.

    Args:
        video_path: Path to the video file.
        config: Tracking configuration.
        background: Background model.
        progress_callback: Optional callback for progress updates.
        num_workers: Number of parallel workers (default: CPU count).
        chunk_size: Number of frames per chunk.
        arena_mask: Optional arena ROI mask at processing resolution (255 = inside).

    Returns:
        List of TrackPoints for all frames.
    """
    # Get video metadata
    with VideoReader(video_path) as reader:
        total_frames = reader.frame_count

    if total_frames <= 0:
        return []

    # MOG2 background is stateful — force sequential processing
    if background is not None and background.mog2_subtractor is not None:
        with VideoReader(video_path) as reader:
            pts, _ = _track_video(reader, config, background, progress_callback, arena_mask=arena_mask, crop_xyxy=crop_xyxy, display_aspect_ratio=display_aspect_ratio)
            return pts

    # Use sequential processing for small videos
    if total_frames <= chunk_size:
        with VideoReader(video_path) as reader:
            pts, _ = _track_video(reader, config, background, progress_callback, arena_mask=arena_mask, crop_xyxy=crop_xyxy, display_aspect_ratio=display_aspect_ratio)
            return pts

    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, min(num_workers, (total_frames + chunk_size - 1) // chunk_size))

    # Create chunks
    chunks: list[tuple[int, int]] = []
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))

    # Compute first detection frame for start-after-first-detection (parallel path)
    first_detection_frame: int | None = None
    if (
        det_infos is not None
        and config.detector.enabled
        and config.detector.start_tracking_after_first_detection
    ):
        first_detection_frame = _compute_first_detection_frame(
            det_infos, config.detector.min_score
        )

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
                    arena_mask=arena_mask,
                    det_infos=det_infos,
                    first_detection_frame=first_detection_frame,
                    crop_xyxy=crop_xyxy,
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
                        pts, _ = _track_video(reader, config, background, progress_callback, arena_mask=arena_mask, crop_xyxy=crop_xyxy, display_aspect_ratio=display_aspect_ratio)
                        return pts

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


def _write_arena_crop_info(
    run_dir: Path,
    crop_xyxy: tuple[int, int, int, int] | None,
    eff_width: int,
    eff_height: int,
    arena_crop_config: object | None,
) -> None:
    """Write arena_crop.json so the user can see whether crop was applied, the box, and CV params."""
    if arena_crop_config is None:
        return
    path = run_dir / "arena_crop.json"
    area_total = eff_width * eff_height
    min_area_ratio = getattr(arena_crop_config, "min_area_ratio", 0.05)
    min_area_px = max(100, int(area_total * min_area_ratio))
    params = {
        "arena_type": getattr(arena_crop_config, "arena_type", "elevated_zero"),
        "canny_low": getattr(arena_crop_config, "canny_low", 50),
        "canny_high": getattr(arena_crop_config, "canny_high", 150),
        "blur_ksize": getattr(arena_crop_config, "blur_ksize", 5),
        "morph_close_ksize": getattr(arena_crop_config, "morph_close_ksize", 0),
        "use_hough_circle": getattr(arena_crop_config, "use_hough_circle", True),
        "hough_min_radius_ratio": getattr(arena_crop_config, "hough_min_radius_ratio", 0.08),
        "hough_max_radius_ratio": getattr(arena_crop_config, "hough_max_radius_ratio", 0.48),
        "hough_center_margin_ratio": getattr(arena_crop_config, "hough_center_margin_ratio", 0.15),
        "min_area_ratio": min_area_ratio,
        "min_area_px": min_area_px,
        "margin_px": getattr(arena_crop_config, "margin_px", 0),
    }
    if getattr(arena_crop_config, "arena_type", "elevated_zero") == "open_field":
        params["open_field_white_threshold"] = getattr(arena_crop_config, "open_field_white_threshold", 200)
        params["open_field_min_area_ratio"] = getattr(arena_crop_config, "open_field_min_area_ratio", 0.02)
        params["open_field_rectangularity_min"] = getattr(arena_crop_config, "open_field_rectangularity_min", 0.6)
    if crop_xyxy is not None:
        data = {
            "applied": True,
            "crop_xyxy": list(crop_xyxy),
            "width": eff_width,
            "height": eff_height,
            "params": params,
            "static_image": "arena_crop_static.png",
            "edges_image": "arena_crop_edges.png",
            "box_image": "arena_crop_box.png",
            "cropped_image": "arena_crop_cropped.png",
        }
        if getattr(arena_crop_config, "morph_close_ksize", 0) > 0:
            data["edges_closed_image"] = "arena_crop_edges_closed.png"
    else:
        data = {
            "applied": False,
            "reason": "no_contour",
            "params": params,
            "static_image": "arena_crop_static.png",
            "edges_image": "arena_crop_edges.png",
        }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_arena_crop_preview(
    video_path: Path,
    crop_xyxy: tuple[int, int, int, int],
    debug_frames_dir: Path,
    dar: str | None = None,
) -> None:
    """Write arena_crop_preview.png: full first frame with crop rectangle (debug only)."""
    with VideoReader(video_path) as reader:
        for idx, frame in reader.iter_frames():
            if idx > 0:
                break
            out = frame.copy()
            x1, y1, x2, y2 = crop_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                "Arena crop (green)",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if dar:
                out = resize_to_display_aspect(out, dar)
            preview_path = debug_frames_dir / "arena_crop_preview.png"
            cv2.imwrite(str(preview_path), out)
            break


def _write_per_frame(
    path: Path,
    points: list[TrackPoint],
    *,
    det_infos: list[FrameDetectorInfo] | None = None,
) -> None:
    has_det = det_infos is not None and len(det_infos) == len(points)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["frame", "x", "y", "area", "confidence", "flags"]
        if has_det:
            header += ["roi_x1", "roi_y1", "roi_x2", "roi_y2", "detector_score", "detector_used"]
        writer.writerow(header)
        for i, point in enumerate(points):
            flags = ";".join(point.flags) if point.flags else ""
            row: list[str | int | float] = [
                point.frame_idx,
                "" if point.x is None else f"{point.x:.3f}",
                "" if point.y is None else f"{point.y:.3f}",
                "" if point.area is None else f"{point.area:.1f}",
                f"{point.confidence:.3f}",
                flags,
            ]
            if has_det and det_infos is not None:
                di = det_infos[i]
                if di.roi_xyxy is not None:
                    row += [di.roi_xyxy[0], di.roi_xyxy[1], di.roi_xyxy[2], di.roi_xyxy[3]]
                else:
                    row += ["", "", "", ""]
                row.append("" if di.detector_score is None else f"{di.detector_score:.4f}")
                row.append("1" if di.detector_used else "0")
            writer.writerow(row)


def _summarize(
    points: list[TrackPoint],
    config: TrackCentroidConfig,
    *,
    det_infos: list[FrameDetectorInfo] | None = None,
) -> dict[str, float]:
    total = len(points)
    tracked = [p for p in points if p.x is not None and p.y is not None]
    coverage = (len(tracked) / total) if total > 0 else 0.0
    mean_conf = (
        float(np.mean([p.confidence for p in tracked])) if tracked else 0.0
    )
    jump_rate = _jump_rate(tracked, config)
    summary: dict[str, float] = {
        "coverage": float(coverage),
        "mean_conf": float(mean_conf),
        "jump_rate": float(jump_rate),
    }

    # Detector metrics
    if det_infos is not None and len(det_infos) > 0:
        det_used = [d for d in det_infos if d.detector_used]
        det_scores = [d.detector_score for d in det_used if d.detector_score is not None]
        det_with_roi = [d for d in det_infos if d.roi_xyxy is not None]
        summary["detector_coverage"] = float(len(det_with_roi) / len(det_infos)) if det_infos else 0.0
        summary["detector_mean_score"] = float(np.mean(det_scores)) if det_scores else 0.0

    return summary


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


def _write_detector_debug_frames(
    video_path: Path,
    det_infos: list[FrameDetectorInfo],
    points: list[TrackPoint],
    out_dir: Path,
    max_frames: int = 10,
    min_score: float = 0.35,
) -> None:
    """Write sampled frames with detector bbox overlay and score to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    dar = get_video_display_aspect_ratio(video_path)

    # Pick evenly-spaced frames that had a detector ROI
    candidates = [
        i for i, di in enumerate(det_infos) if di.roi_xyxy is not None
    ]
    if not candidates:
        return

    step = max(1, len(candidates) // max_frames)
    selected = candidates[::step][:max_frames]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    with VideoReader(video_path) as reader:
        for idx, frame in reader.iter_frames():
            if idx not in selected:
                continue
            di = det_infos[idx]
            if di.roi_xyxy is not None:
                x1, y1, x2, y2 = di.roi_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pt = points[idx] if idx < len(points) else None
            if pt is not None and pt.x is not None and pt.y is not None:
                cv2.circle(frame, (int(pt.x), int(pt.y)), 5, (0, 0, 255), -1)

            # Score and threshold overlay (always show something)
            y_text = 36
            if di.detector_score is not None:
                score_str = f"score: {di.detector_score:.3f}"
                meets = di.detector_score >= min_score
                color = (0, 255, 0) if meets else (0, 0, 255)  # BGR green / red
                cv2.putText(frame, score_str, (10, y_text), font, font_scale, color, thickness, cv2.LINE_AA)
                y_text += 28
                thresh_str = f"min: {min_score:.2f}  " + ("OK" if meets else "below")
                cv2.putText(frame, thresh_str, (10, y_text), font, font_scale * 0.9, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(frame, "score: --", (10, y_text), font, font_scale, (128, 128, 128), thickness, cv2.LINE_AA)
                y_text += 28
                cv2.putText(
                    frame,
                    "ROI interpolated (detector not run this frame)" if not di.detector_used else "no detection",
                    (10, y_text), font, font_scale * 0.8, (128, 128, 128), thickness, cv2.LINE_AA,
                )

            if dar:
                frame = resize_to_display_aspect(frame, dar)
            cv2.imwrite(str(out_dir / f"det_frame_{idx:06d}.png"), frame)


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

