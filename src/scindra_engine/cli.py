from __future__ import annotations

import json
import os
import platform
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
) -> None:
    """Track a centroid using the classical backend."""
    try:
        if config_path:
            data = _load_config(config_path)
            config = TrackCentroidConfig.model_validate(data)
        else:
            config = TrackCentroidConfig.model_validate({})

        def progress_callback(done: int, total: int) -> None:
            if total > 0:
                typer.echo(f"PROGRESS {done}/{total}")

        run_track_centroid(
            video_path=video,
            out_dir=out_dir,
            config=config,
            progress_callback=progress_callback,
            write_overlay_video=overlay,
            write_heatmap=heatmap,
            trail_length=trail_length,
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
