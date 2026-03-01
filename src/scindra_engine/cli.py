from __future__ import annotations

import json
import os
import platform
import shlex
import sys
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


@app.command("arena-crop-test")
def arena_crop_test(
    out_dir: Path = typer.Option(..., "--out", file_okay=False, help="Directory to write static image, edges, and result JSON"),
    video: Path | None = typer.Option(None, "--video", exists=True, file_okay=True, dir_okay=False, readable=True, help="Build static image from this video (sample N frames)"),
    image: Path | None = typer.Option(None, "--image", exists=True, file_okay=True, dir_okay=False, readable=True, help="Use this single image as the static image (no averaging)"),
    frames: int = typer.Option(50, "--frames", min=5, help="When using --video, number of frames to sample for the static image"),
    config_path: Path | None = typer.Option(None, "--config", exists=True, file_okay=True, dir_okay=False, readable=True, help="Optional JSON/YAML config; arena_crop params will be used"),
    min_area_ratio: float | None = typer.Option(None, "--min-area-ratio", min=0.0, max=1.0, help="Override min contour area as fraction of frame (e.g. 0.02)"),
    canny_low: int | None = typer.Option(None, "--canny-low", min=0, max=255, help="Override Canny low threshold"),
    canny_high: int | None = typer.Option(None, "--canny-high", min=0, max=255, help="Override Canny high threshold"),
    blur_ksize: int | None = typer.Option(None, "--blur-ksize", min=1, max=31, help="Override Gaussian blur kernel size (odd)"),
    margin_px: int | None = typer.Option(None, "--margin-px", help="Override margin pixels (negative = shrink box)"),
    morph_close_ksize: int | None = typer.Option(None, "--morph-close-ksize", min=0, max=31, help="Close edges to connect gaps (0=off, try 9-15 for broken circles)"),
    crop_expand_ratio: float | None = typer.Option(None, "--crop-expand-ratio", min=0.0, max=0.5, help="Expand detected box by this fraction so outer arena edge is included (e.g. 0.05)"),
    use_hough_circle: bool = typer.Option(True, "--hough/--no-hough", help="Try Hough circle detection first for circular arenas"),
    hough_min_radius_ratio: float | None = typer.Option(None, "--hough-min-radius", min=0.01, max=0.5, help="Hough min radius as fraction of min(w,h)"),
    hough_max_radius_ratio: float | None = typer.Option(None, "--hough-max-radius", min=0.1, max=0.99, help="Hough max radius as fraction of min(w,h)"),
    hough_acc_threshold: int | None = typer.Option(None, "--hough-acc-threshold", min=5, max=100, help="Hough accumulator threshold; lower = more sensitive"),
    circle_only: bool = typer.Option(False, "--circle-only/--contour-fallback", help="Only use circle detection; no contour fallback"),
    debug_out: Path | None = typer.Option(None, "--debug-out", file_okay=False, help="Write pipeline step images and manifest to this directory (what the pipeline sees at each step)"),
) -> None:
    """Test arena detection on a video or single image without running full tracking.

    Builds (or loads) a static image, runs the contour-based arena detector, and writes
    arena_crop_static.png, arena_crop_edges.png, and arena_crop.json to --out so you can
    iterate on parameters quickly. Use --video + --frames to average frames, or --image
    to test on a single frame.
    """
    import cv2

    from scindra_engine.arena_crop import (
        build_static_image,
        detect_arena_crop_xyxy,
        expand_arena_box_xyxy,
        get_arena_detection_edges,
        get_arena_detection_edges_with_close,
    )
    from scindra_engine.video_io import crop_to_display_aspect, draw_crop_box_for_display, ensure_square_image, get_video_display_aspect_ratio, resize_to_display_aspect

    if video is None and image is None:
        typer.echo("Error: provide either --video or --image.", err=True)
        raise typer.Exit(code=1)
    if video is not None and image is not None:
        typer.echo("Error: provide only one of --video or --image.", err=True)
        raise typer.Exit(code=1)

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve params from config or defaults
    canny_low_val = 50
    canny_high_val = 150
    blur_ksize_val = 5
    min_area_ratio_val = 0.05
    margin_px_val = 0
    crop_expand_ratio_val = 0.0
    circle_padding_ratio_val = 0.03
    force_square_crop_val = True
    morph_close_ksize_val = 0
    use_hough_circle_val = True
    hough_min_radius_val = 0.08
    hough_max_radius_val = 0.48
    hough_center_margin_val = 0.15
    hough_acc_threshold_val = 25
    circle_only_val = False
    if config_path is not None:
        data = _load_config(config_path)
        try:
            cfg = TrackCentroidConfig.model_validate(data)
            ac = cfg.arena_crop
            canny_low_val = ac.canny_low
            canny_high_val = ac.canny_high
            blur_ksize_val = ac.blur_ksize
            min_area_ratio_val = ac.min_area_ratio
            margin_px_val = ac.margin_px
            crop_expand_ratio_val = getattr(ac, "crop_expand_ratio", 0.0)
            circle_padding_ratio_val = getattr(ac, "circle_padding_ratio", 0.03)
            force_square_crop_val = getattr(ac, "force_square_crop", True)
            morph_close_ksize_val = getattr(ac, "morph_close_ksize", 0)
            use_hough_circle_val = getattr(ac, "use_hough_circle", True)
            hough_min_radius_val = getattr(ac, "hough_min_radius_ratio", 0.08)
            hough_max_radius_val = getattr(ac, "hough_max_radius_ratio", 0.48)
            hough_center_margin_val = getattr(ac, "hough_center_margin_ratio", 0.15)
            hough_acc_threshold_val = getattr(ac, "hough_acc_threshold", 25)
            circle_only_val = getattr(ac, "circle_only", False)
        except Exception as e:
            typer.echo(f"Warning: could not load arena_crop from config: {e}", err=True)
    if canny_low is not None:
        canny_low_val = canny_low
    if canny_high is not None:
        canny_high_val = canny_high
    if blur_ksize is not None:
        blur_ksize_val = blur_ksize
    if min_area_ratio is not None:
        min_area_ratio_val = min_area_ratio
    if margin_px is not None:
        margin_px_val = margin_px
    if crop_expand_ratio is not None:
        crop_expand_ratio_val = crop_expand_ratio
    if morph_close_ksize is not None:
        morph_close_ksize_val = morph_close_ksize
    if not use_hough_circle:
        use_hough_circle_val = False
    if hough_min_radius_ratio is not None:
        hough_min_radius_val = hough_min_radius_ratio
    if hough_max_radius_ratio is not None:
        hough_max_radius_val = hough_max_radius_ratio
    if hough_acc_threshold is not None:
        hough_acc_threshold_val = hough_acc_threshold
    if circle_only:
        circle_only_val = True

    dar: str | None = None
    if video is not None:
        with VideoReader(video) as reader:
            static_img = build_static_image(reader, min(frames, reader.frame_count), method="median")
        dar = get_video_display_aspect_ratio(video)
    else:
        static_img = cv2.imread(str(image))
        if static_img is None:
            typer.echo(f"Error: could not load image '{image}'", err=True)
            raise typer.Exit(code=1)

    static_for_png = resize_to_display_aspect(static_img, dar) if dar else static_img
    cv2.imwrite(str(out_dir / "arena_crop_static.png"), static_for_png)
    edges = get_arena_detection_edges(
        static_img, canny_low=canny_low_val, canny_high=canny_high_val, blur_ksize=blur_ksize_val
    )
    edges_for_png = resize_to_display_aspect(edges, dar) if dar else edges
    cv2.imwrite(str(out_dir / "arena_crop_edges.png"), edges_for_png)
    if morph_close_ksize_val > 0:
        edges_closed = get_arena_detection_edges_with_close(
            static_img,
            canny_low=canny_low_val,
            canny_high=canny_high_val,
            blur_ksize=blur_ksize_val,
            morph_close_ksize=morph_close_ksize_val,
        )
        edges_closed_png = resize_to_display_aspect(edges_closed, dar) if dar else edges_closed
        cv2.imwrite(str(out_dir / "arena_crop_edges_closed.png"), edges_closed_png)

    debug_manifest: list[dict] = []

    if debug_out is not None:
        debug_out.mkdir(parents=True, exist_ok=True)

    def _arena_debug_callback(step: str, data: dict) -> None:
        img = data.get("image")
        if img is not None and debug_out is not None:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out_img = resize_to_display_aspect(img, dar) if dar else img
            path = debug_out / f"arena_debug_{step}.png"
            cv2.imwrite(str(path), out_img)
        manifest_entry = {k: v for k, v in data.items() if k != "image"}
        manifest_entry["step"] = step
        debug_manifest.append(manifest_entry)

    box, _chosen_circle = detect_arena_crop_xyxy(
        static_img,
        margin_px=margin_px_val,
        min_area_ratio=min_area_ratio_val,
        canny_low=canny_low_val,
        canny_high=canny_high_val,
        blur_ksize=blur_ksize_val,
        morph_close_ksize=morph_close_ksize_val,
        use_hough_circle=use_hough_circle_val,
        hough_min_radius_ratio=hough_min_radius_val,
        hough_max_radius_ratio=hough_max_radius_val,
        hough_center_margin_ratio=hough_center_margin_val,
        hough_acc_threshold=hough_acc_threshold_val,
        circle_only=circle_only_val,
        circle_padding_ratio=circle_padding_ratio_val,
        force_square_crop=force_square_crop_val,
        debug_callback=_arena_debug_callback if debug_out is not None else None,
        dar=dar,
    )
    if box is not None and crop_expand_ratio_val > 0:
        h, w = static_img.shape[:2]
        box = expand_arena_box_xyxy(box, w, h, crop_expand_ratio_val)

    if debug_out is not None:
        (debug_out / "arena_debug_manifest.json").write_text(
            json.dumps(debug_manifest, indent=2), encoding="utf-8"
        )
        typer.echo(f"Debug pipeline output written to {debug_out}")

    h, w = static_img.shape[:2]
    area_total = w * h
    min_area_px = max(100, int(area_total * min_area_ratio_val))
    result = {
        "applied": box is not None,
        "params": {
            "canny_low": canny_low_val,
            "canny_high": canny_high_val,
            "blur_ksize": blur_ksize_val,
            "morph_close_ksize": morph_close_ksize_val,
            "use_hough_circle": use_hough_circle_val,
            "hough_min_radius_ratio": hough_min_radius_val,
            "hough_max_radius_ratio": hough_max_radius_val,
            "hough_center_margin_ratio": hough_center_margin_val,
            "min_area_ratio": min_area_ratio_val,
            "min_area_px": min_area_px,
            "margin_px": margin_px_val,
        },
        "static_image": "arena_crop_static.png",
        "edges_image": "arena_crop_edges.png",
    }
    if morph_close_ksize_val > 0:
        result["edges_closed_image"] = "arena_crop_edges_closed.png"
    if box is not None:
        result["crop_xyxy"] = list(box)
        result["width"] = box[2] - box[0]
        result["height"] = box[3] - box[1]
        box_for_png = draw_crop_box_for_display(static_img, box, dar)
        cv2.imwrite(str(out_dir / "arena_crop_box.png"), box_for_png)
        cropped = crop_to_display_aspect(static_img, box, dar)
        if _chosen_circle is not None and force_square_crop_val:
            cropped = ensure_square_image(cropped)
        cv2.imwrite(str(out_dir / "arena_crop_cropped.png"), cropped)
        result["box_image"] = "arena_crop_box.png"
        result["cropped_image"] = "arena_crop_cropped.png"
        typer.echo(f"Detected arena crop: {box[2] - box[0]}x{box[3] - box[1]} at {box}")
    else:
        result["reason"] = "no_contour"
        typer.echo("No arena contour met the criteria. Try lowering --min-area-ratio or adjusting --canny-low/--canny-high.")

    (out_dir / "arena_crop.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    files_written = ["arena_crop_static.png", "arena_crop_edges.png", "arena_crop.json"]
    if morph_close_ksize_val > 0:
        files_written.append("arena_crop_edges_closed.png")
    if box is not None:
        files_written.extend(["arena_crop_box.png", "arena_crop_cropped.png"])
    typer.echo("Wrote " + ", ".join(str(out_dir / f) for f in files_written))


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
    roi_padding_ratio: float | None = typer.Option(
        None,
        "--roi-padding-ratio",
        min=0.0,
        help="Padding as fraction of bbox size (e.g. 0.5 = 50%% of smaller dimension). Overrides roi_padding_px when set.",
    ),
    max_roi_jump_ratio: float | None = typer.Option(
        None,
        "--max-roi-jump-ratio",
        min=0.0,
        help="Max ROI center jump as fraction of bbox size. Overrides max_roi_jump_px when set.",
    ),
    overlay_scale: float = typer.Option(
        0.25,
        "--overlay-scale",
        min=0.05,
        max=1.0,
        help="Scale factor for overlay video resolution (1.0 = full-res, 0.25 = quarter-res).",
    ),
    heatmap_blur_ksize: int | None = typer.Option(
        None,
        "--heatmap-blur-ksize",
        min=1,
        help="Gaussian blur kernel size for heatmap (must be odd). When omitted, use config default.",
    ),
    arena_crop: str = typer.Option(
        "off",
        "--arena-crop",
        help="Crop video to arena: auto (detect from static frame), manual (use --crop-xyxy), or off.",
    ),
    crop_x1: int | None = typer.Option(None, "--crop-x1", min=0, help="Arena crop left (manual mode)."),
    crop_y1: int | None = typer.Option(None, "--crop-y1", min=0, help="Arena crop top (manual mode)."),
    crop_x2: int | None = typer.Option(None, "--crop-x2", min=0, help="Arena crop right (manual mode)."),
    crop_y2: int | None = typer.Option(None, "--crop-y2", min=0, help="Arena crop bottom (manual mode)."),
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
        # Arena crop: auto | manual | off
        ac_val = (arena_crop or "off").strip().lower()
        if ac_val in ("auto", "manual"):
            ac_overrides: dict[str, object] = {"enabled": True, "mode": ac_val.upper()}
            if ac_val == "manual":
                if crop_x1 is None or crop_y1 is None or crop_x2 is None or crop_y2 is None:
                    typer.echo("Error: --arena-crop manual requires --crop-x1, --crop-y1, --crop-x2, --crop-y2.", err=True)
                    raise typer.Exit(1)
                ac_overrides["manual_crop_xyxy"] = (crop_x1, crop_y1, crop_x2, crop_y2)
            overrides["arena_crop"] = config.arena_crop.model_copy(update=ac_overrides)
        elif ac_val != "off":
            typer.echo(f"Error: --arena-crop must be auto, manual, or off (got {arena_crop!r}).", err=True)
            raise typer.Exit(1)
        if debug:
            overrides["debug_mode"] = True
        if debug_interval is not None:
            overrides["debug_frame_interval"] = debug_interval
        if debug_max_frames is not None:
            overrides["debug_max_frames"] = debug_max_frames
        if overlay_scale is not None:
            overrides["overlay_scale"] = overlay_scale
        if heatmap_blur_ksize is not None:
            overrides["heatmap_blur_ksize"] = heatmap_blur_ksize
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
            if roi_padding_ratio is not None:
                det_overrides["roi_padding_ratio"] = roi_padding_ratio
            if max_roi_jump_ratio is not None:
                det_overrides["max_roi_jump_ratio"] = max_roi_jump_ratio
            new_det_cfg = config.detector.model_copy(update=det_overrides)
            config = config.model_copy(update={"detector": new_det_cfg})

        # Track start time for progress reporting
        start_time = time.time()
        last_line_length = 0

        def progress_callback(done: int, total: int) -> None:
            nonlocal last_line_length

            if total <= 0:
                return

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
            if done > 0 and done < total and rate > 0:
                remaining_frames = total - done
                eta_seconds = remaining_frames / rate
                eta_str = _format_duration(eta_seconds)
            elif done >= total:
                eta_str = "0s"
            else:
                eta_str = "?"

            elapsed_str = _format_duration(elapsed)

            # Format rate
            rate_str = f"{rate:.1f} fps" if rate > 0 else "0.0 fps"

            base_msg = (
                f"PROGRESS {done}/{total} ({percentage:.1f}%) | "
                f"Elapsed: {elapsed_str} | "
                f"Rate: {rate_str} | "
                f"ETA: {eta_str}"
            )

            # Pad with spaces to fully overwrite previous content
            pad = max(0, last_line_length - len(base_msg))
            msg = "\r" + base_msg + (" " * pad)
            last_line_length = len(base_msg)

            # For intermediate updates, stay on the same line; on completion, end the line
            if done < total:
                typer.echo(msg, nl=False)
            else:
                typer.echo(msg)

        command = shlex.join(sys.argv)
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
            command=command,
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
