from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import typer

from scindra_engine import __version__
from pydantic import ValidationError

from scindra_engine.hash_utils import file_bytes, sha256_file
from scindra_engine.schemas import AnalysisConfig, TrackCentroidConfig
from scindra_engine.video_io import FrameSampler, VideoIOError, VideoReader
from scindra_engine.runners.track_centroid import run_track_centroid

app = typer.Typer(no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"scindra-engine {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    if ctx.invoked_subcommand is None and not version:
        pass  # No subcommand and no --version: could show help or do nothing


@app.command("engine-info")
def engine_info(json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON")) -> None:
    """Show engine version and system information."""
    try:
        info: dict[str, object] = {
            "engine_version": __version__,
            "git_commit": os.environ.get("GIT_COMMIT"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "opencv_version": None,
        }

        try:
            import cv2

            info["opencv_version"] = cv2.__version__
        except ImportError:
            pass

        if json_output:
            typer.echo(json.dumps(info))
        else:
            for key, value in info.items():
                typer.echo(f"{key}: {value}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("probe")
def probe(
    video: Path = typer.Option(..., "--video", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input video path"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON probe result"),
) -> None:
    """Probe video file and output metadata."""
    try:
        video_path = video.resolve()
        sha256_hash = sha256_file(video_path)
        num_bytes = file_bytes(video_path)

        with VideoReader(video_path) as reader:
            result: dict[str, object] = {
                "path": str(video_path),
                "sha256": sha256_hash,
                "bytes": num_bytes,
                "fps": reader.fps,
                "frame_count": reader.frame_count,
                "width": reader.width,
                "height": reader.height,
            }

        if json_output:
            typer.echo(json.dumps(result))
        else:
            typer.echo(f"Video: {result['path']}")
            typer.echo(f"  SHA256: {result['sha256']}")
            typer.echo(f"  Size: {result['bytes']} bytes")
            typer.echo(f"  FPS: {result['fps']}")
            typer.echo(f"  Frames: {result['frame_count']}")
            typer.echo(f"  Resolution: {result['width']}x{result['height']}")

    except (FileNotFoundError, VideoIOError, OSError) as e:
        typer.echo(f"Error: could not open video '{video}': {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("extract-frames")
def extract_frames(
    video: Path = typer.Option(..., "--video", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input video path"),
    out_dir: Path = typer.Option(..., "--out", file_okay=False, help="Directory to write PNG frames"),
    count: int = typer.Option(10, "--count", min=1, help="Number of frames to sample"),
) -> None:
    """Extract evenly sampled frames from a video as PNG files."""
    try:
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)

        with VideoReader(video) as reader:
            sampler = FrameSampler(reader)
            frames = sampler.sample(count)

            if not frames:
                typer.echo(f"Error: no frames could be sampled from video '{video}'", err=True)
                raise typer.Exit(code=1)

            written = 0
            for frame_index, frame_bgr in frames:
                filename = out_dir / f"frame_{frame_index:06d}.png"
                if not cv2.imwrite(str(filename), frame_bgr):
                    typer.echo(f"Error: failed to write frame {frame_index} to {filename}", err=True)
                    raise typer.Exit(code=1)
                written += 1

            typer.echo(f"Wrote {written} frames to {out_dir}")

    except (FileNotFoundError, VideoIOError, OSError) as e:
        typer.echo(f"Error: could not process video '{video}': {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted string like "1m 23s" or "45s" or "0.5s".
    """
    if seconds < 1.0:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def _load_config(config_path: Path) -> dict[str, object]:
    """Load a JSON or YAML config file and return a dict."""
    suffix = config_path.suffix.lower()
    data: dict[str, object]

    if suffix == ".json":
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("JSON file must contain an object at the top level")
            data = loaded
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            typer.echo("Error: pyyaml is required for YAML config files. Install with: pip install pyyaml", err=True)
            raise typer.Exit(code=1)

        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if loaded is None:
                raise ValueError("YAML file is empty or contains only null")
            if not isinstance(loaded, dict):
                raise ValueError("YAML file must contain a mapping at the top level")
            data = loaded
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Only .json, .yaml, and .yml are supported.")

    return data


@app.command("validate-config")
def validate_config(
    config_path: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False, readable=True, help="JSON or YAML analysis config file"),
) -> None:
    """Validate an analysis configuration file."""
    try:
        data = _load_config(config_path)
        AnalysisConfig.model_validate(data)
        typer.echo(f"Config is valid: {config_path}")

    except json.JSONDecodeError as e:
        typer.echo(f"Config invalid: JSON decode error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Config invalid: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("track-centroid")
def track_centroid(
    video: Path = typer.Option(..., "--video", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input video path"),
    out_dir: Path = typer.Option(..., "--out", file_okay=False, help="Output directory"),
    config_path: Path | None = typer.Option(None, "--config", exists=True, file_okay=True, dir_okay=False, readable=True, help="Optional JSON or YAML config"),
    overlay: bool = typer.Option(
        True,
        "--overlay/--no-overlay",
        help="Enable or disable overlay video output (default: on)",
    ),
    heatmap: bool = typer.Option(
        True,
        "--heatmap/--no-heatmap",
        help="Enable or disable heatmap PNG output (default: on)",
    ),
    trail_length: int = typer.Option(
        30,
        "--trail-length",
        min=1,
        help="Number of recent centroids to include in the overlay trail",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Number of parallel workers for frame processing (default: auto, based on CPU count)",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Number of frames per chunk for parallel processing (default: 200)",
    ),
    downsample_factor: float | None = typer.Option(
        None,
        "--downsample-factor",
        min=1.0,
        help="Downsample frames by this factor before processing (e.g., 2.0 = half resolution). Coordinates are scaled back to original resolution.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Enable debug mode: write frames showing centroid blobs (detected/excluded). Uses sequential processing.",
    ),
    debug_interval: int | None = typer.Option(
        None,
        "--debug-interval",
        min=1,
        help="When --debug is set, write a debug frame every N frames (default from config).",
    ),
    debug_max_frames: int | None = typer.Option(
        None,
        "--debug-max-frames",
        min=1,
        help="When --debug is set, cap the number of debug frames. Omit for config default; use 0 for no cap (requires config support).",
    ),
    detector: bool = typer.Option(
        False,
        "--detector/--no-detector",
        help="Enable detector-assisted ROI tracking (requires scindra-engine[detector]).",
    ),
    detector_model: Path | None = typer.Option(
        None,
        "--detector-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the YOLOX ONNX model file.",
    ),
    detector_every_n: int | None = typer.Option(
        None,
        "--detector-every-n",
        min=1,
        help="Run detector every N frames (default: 15).",
    ),
    detector_min_score: float | None = typer.Option(
        None,
        "--detector-min-score",
        min=0.0,
        max=1.0,
        help="Minimum detector score to accept a detection (default: 0.35).",
    ),
    roi_padding_px: int | None = typer.Option(
        None,
        "--roi-padding-px",
        min=0,
        help="Pixels to pad around the detector bounding box for the ROI (default: 60).",
    ),
) -> None:
    """Track a centroid using the classical backend."""
    try:
        if config_path:
            data = _load_config(config_path)
            config = TrackCentroidConfig.model_validate(data)
        else:
            config = TrackCentroidConfig.model_validate({})

        overrides: dict[str, object] = {}
        if downsample_factor is not None:
            overrides["downsample_factor"] = downsample_factor
        if debug:
            overrides["debug_mode"] = True
        if debug_interval is not None:
            overrides["debug_frame_interval"] = debug_interval
        if debug_max_frames is not None:
            overrides["debug_max_frames"] = debug_max_frames
        if overrides:
            config = config.model_copy(update=overrides)

        # --- Detector setup ---
        det_instance = None
        if detector or config.detector.enabled:
            det_instance = _try_create_detector(
                model_path=str(detector_model) if detector_model else config.detector.model_path,
                fallback=config.detector.fallback_to_classical_full_frame,
            )
            # Apply detector CLI overrides
            det_overrides: dict[str, object] = {"enabled": True}
            if detector_model is not None:
                det_overrides["model_path"] = str(detector_model)
            if detector_every_n is not None:
                det_overrides["every_n_frames"] = detector_every_n
            if detector_min_score is not None:
                det_overrides["min_score"] = detector_min_score
            if roi_padding_px is not None:
                det_overrides["roi_padding_px"] = roi_padding_px
            new_det_cfg = config.detector.model_copy(update=det_overrides)
            config = config.model_copy(update={"detector": new_det_cfg})

        # Track start time for progress reporting
        start_time = time.time()

        def progress_callback(done: int, total: int) -> None:
            if total > 0:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Calculate percentage
                percentage = (done / total) * 100
                
                # Calculate rate (frames per second)
                if done > 0 and elapsed > 0:
                    rate = done / elapsed
                else:
                    rate = 0.0
                
                # Estimate time remaining
                if done > 0 and done < total:
                    remaining_frames = total - done
                    if rate > 0:
                        eta_seconds = remaining_frames / rate
                        eta_str = _format_duration(eta_seconds)
                    else:
                        eta_str = "?"
                elif done >= total:
                    eta_str = "0s"
                else:
                    eta_str = "?"
                
                elapsed_str = _format_duration(elapsed)
                
                # Format rate
                if rate > 0:
                    rate_str = f"{rate:.1f} fps"
                else:
                    rate_str = "0.0 fps"
                
                typer.echo(
                    f"PROGRESS {done}/{total} ({percentage:.1f}%) | "
                    f"Elapsed: {elapsed_str} | "
                    f"Rate: {rate_str} | "
                    f"ETA: {eta_str}"
                )

        run_track_centroid(
            video_path=video,
            out_dir=out_dir,
            config=config,
            progress_callback=progress_callback,
            write_overlay_video=overlay,
            write_heatmap=heatmap,
            trail_length=trail_length,
            parallel_workers=workers,
            chunk_size=chunk_size,
            detector=det_instance,
        )
    except (FileNotFoundError, VideoIOError, OSError) as e:
        typer.echo(f"Error: could not open video '{video}': {e}", err=True)
        raise typer.Exit(code=1)
    except ValidationError as e:
        typer.echo(f"Config invalid: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def _try_create_detector(
    model_path: str | None,
    fallback: bool,
) -> object | None:
    """Attempt to create a YOLOX ONNX detector instance.

    Returns the detector or None (with warnings).
    """
    from scindra_engine.detectors.base import ModelResolver

    try:
        # Check onnxruntime availability first
        import onnxruntime as _ort  # noqa: F401
    except ImportError:
        msg = (
            "Detector enabled but optional dependency missing. "
            "Install: pip install scindra-engine[detector]  (or: uv pip install scindra-engine[detector])"
        )
        typer.echo(f"WARNING: {msg}", err=True)
        if not fallback:
            raise typer.Exit(code=1)
        typer.echo("WARNING: DETECTOR_DEP_MISSING — continuing with classical-only tracking.", err=True)
        return None

    resolver = ModelResolver(model_path)
    resolved = resolver.resolve()
    if resolved is None:
        msg = "Detector enabled but no model found (checked --detector-model, SCINDRA_YOLOX_ONNX_PATH, packaged asset)."
        typer.echo(f"WARNING: DETECTOR_UNAVAILABLE — {msg}", err=True)
        if not fallback:
            raise typer.Exit(code=1)
        return None

    from scindra_engine.detectors.yolox_onnx import YOLOXOnnxDetector

    path, meta = resolved
    return YOLOXOnnxDetector(str(path), meta)


@app.command("detect-mouse")
def detect_mouse(
    video: Path = typer.Option(
        ..., "--video", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input video path"
    ),
    out_dir: Path = typer.Option(
        ..., "--out", file_okay=False, help="Output directory"
    ),
    model: Path | None = typer.Option(
        None, "--model", exists=True, file_okay=True, dir_okay=False, readable=True, help="YOLOX ONNX model path"
    ),
    every_n: int = typer.Option(
        15, "--every-n", min=1, help="Run detector every N frames"
    ),
) -> None:
    """Run the YOLOX detector on a video and produce detections.csv + debug frames."""
    import csv as _csv

    from scindra_engine.detectors.base import ModelResolver

    try:
        import onnxruntime as _ort  # noqa: F401
    except ImportError:
        typer.echo(
            "Error: onnxruntime is required. Install: pip install scindra-engine[detector]  (or: uv pip install scindra-engine[detector])",
            err=True,
        )
        raise typer.Exit(code=1)

    resolver = ModelResolver(str(model) if model else None)
    resolved = resolver.resolve()
    if resolved is None:
        typer.echo(
            "Error: no YOLOX model found (provide --model, set SCINDRA_YOLOX_ONNX_PATH, or bundle asset).",
            err=True,
        )
        raise typer.Exit(code=1)

    from scindra_engine.detectors.yolox_onnx import YOLOXOnnxDetector

    model_path, meta = resolved
    det = YOLOXOnnxDetector(str(model_path), meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "detections.csv"
    debug_dir = out_dir / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)

    import cv2 as _cv2

    total_frames = 0
    detection_count = 0
    scores: list[float] = []
    debug_written = 0
    max_debug = 10

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["frame", "score", "x1", "y1", "x2", "y2"])

        with VideoReader(video) as reader:
            total_frames = reader.frame_count
            debug_step = max(1, total_frames // (every_n * max_debug)) * every_n

            for idx, frame in reader.iter_frames():
                if idx % every_n != 0:
                    continue
                result = det.detect(frame)
                if result.best is not None:
                    b = result.best
                    writer.writerow([idx, f"{b.score:.4f}", *b.bbox_xyxy])
                    detection_count += 1
                    scores.append(b.score)
                else:
                    writer.writerow([idx, "", "", "", "", ""])

                # Debug frames (sampled)
                if debug_written < max_debug and (idx % debug_step == 0 or idx == 0):
                    if result.best is not None:
                        x1, y1, x2, y2 = result.best.bbox_xyxy
                        _cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    _cv2.imwrite(str(debug_dir / f"det_{idx:06d}.png"), frame)
                    debug_written += 1

    frames_checked = (total_frames + every_n - 1) // every_n if total_frames > 0 else 0
    coverage_pct = (detection_count / frames_checked * 100.0) if frames_checked > 0 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    typer.echo(f"Frames checked: {frames_checked}")
    typer.echo(f"Detections: {detection_count} ({coverage_pct:.1f}%)")
    typer.echo(f"Mean score: {mean_score:.4f}")
    typer.echo(f"Output: {csv_path}")
