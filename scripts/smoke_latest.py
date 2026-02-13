#!/usr/bin/env python3
"""Adaptive smoke test for scindra-engine CLI.

This script detects available CLI capabilities and exercises the most end-to-end
path available. It uses CLI commands exclusively (no internal function calls) and
produces artifacts in out/smoke_latest/<timestamp>/.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]


def get_repo_root() -> Path:
    """Get the repository root directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def run_cmd(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a command and raise on nonzero exit code."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        args,
        cwd=cwd or get_repo_root(),
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"Command failed: {' '.join(args)}", file=sys.stderr)
        print(f"stdout: {result.stdout}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    if result.stdout:
        print(result.stdout, end="")


def cmd_help(cmd: str) -> str:
    """Get help output for a command."""
    try:
        result = subprocess.run(
            ["uv", "run", "scindra-engine", cmd, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def detect_capabilities() -> dict[str, Any]:
    """Detect available CLI capabilities."""
    caps: dict[str, Any] = {
        "commands": {},
        "flags": {},
    }

    # Detect commands
    for cmd in ["engine-info", "probe", "extract-frames", "validate-config", "track-centroid", "init-config", "auto-setup", "batch"]:
        help_output = cmd_help(cmd)
        if help_output:  # Command exists if help output is non-empty
            caps["commands"][cmd] = True
            # Get help for specific flags
            if cmd == "track-centroid":
                caps["flags"]["overlay"] = "--overlay" in help_output or "--no-overlay" in help_output
                caps["flags"]["heatmap"] = "--heatmap" in help_output or "--no-heatmap" in help_output
                caps["flags"]["config"] = "--config" in help_output

    return caps


def write_good_open_field_video(path: Path, num_frames: int = 30, size: tuple[int, int] = (160, 120), fps: float = 15.0) -> None:
    """Write a synthetic video with a single dark blob moving across frame."""
    import cv2  # type: ignore[import-untyped]

    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    background_color = 200
    blob_color = 40

    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), background_color, dtype=np.uint8)
            # Single blob moving from left to right
            cx = int(20 + (width - 40) * (i / max(1, num_frames - 1)))
            cy = int(height / 2 + 5 * np.sin(i / 5.0))

            cv2.ellipse(
                frame,
                (cx, cy),
                (8, 5),
                0,
                0,
                360,
                (blob_color, blob_color, blob_color),
                -1,
            )

            writer.write(frame)
    finally:
        writer.release()


def write_ambiguous_shadow_video(path: Path, num_frames: int = 30, size: tuple[int, int] = (160, 120), fps: float = 15.0) -> None:
    """Write a synthetic video with mouse-like and shadow-like blobs (ambiguous)."""
    # Reuse existing fixture function
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root))
    from tests.fixtures.synth_mouse_shadow import write_synth_mouse_shadow_video

    write_synth_mouse_shadow_video(path, num_frames=num_frames, size=size, fps=fps)


def find_newest_run_dir(out_dir: Path) -> Path:
    """Find the newest run directory."""
    run_dirs = sorted(out_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {out_dir}")
    return run_dirs[-1]


def patch_config_file(config_path: Path, video_path: Path, outputs_dir: Path) -> None:
    """Patch a config file to set video.path and outputs.out_dir."""
    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        # Set video.path
        if "video" not in data:
            data["video"] = {}
        data["video"]["path"] = str(video_path.resolve())
        # Set outputs.out_dir
        if "outputs" not in data:
            data["outputs"] = {}
        data["outputs"]["out_dir"] = str(outputs_dir.resolve())
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "video" not in data:
            data["video"] = {}
        data["video"]["path"] = str(video_path.resolve())
        if "outputs" not in data:
            data["outputs"] = {}
        data["outputs"]["out_dir"] = str(outputs_dir.resolve())
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def main() -> int:
    """Main smoke test orchestrator."""
    repo_root = get_repo_root()
    os.chdir(repo_root)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "out" / "smoke_latest" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Detect capabilities
    print("==> Detecting CLI capabilities")
    caps = detect_capabilities()
    print(f"Commands detected: {list(caps['commands'].keys())}")
    print(f"Flags detected: {caps['flags']}")

    # Generate synthetic videos
    print("==> Generating synthetic fixture videos")
    good_video = run_dir / "good_open_field.mp4"
    ambiguous_video = run_dir / "ambiguous_shadow.mp4"
    write_good_open_field_video(good_video)
    write_ambiguous_shadow_video(ambiguous_video)
    print(f"Generated: {good_video}")
    print(f"Generated: {ambiguous_video}")

    # Baseline UX (always run)
    print("==> Running baseline UX tests")
    # Capture output to file
    result = subprocess.run(
        ["uv", "run", "scindra-engine", "engine-info", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    (run_dir / "engine_info.json").write_text(result.stdout, encoding="utf-8")

    result = subprocess.run(
        ["uv", "run", "scindra-engine", "probe", "--video", str(good_video), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    (run_dir / "probe_good.json").write_text(result.stdout, encoding="utf-8")

    frames_dir = run_dir / "frames_good"
    run_cmd(
        ["uv", "run", "scindra-engine", "extract-frames", "--video", str(good_video), "--out", str(frames_dir), "--count", "8"],
    )
    png_files = list(frames_dir.glob("*.png"))
    assert len(png_files) == 8, f"Expected 8 PNG files, found {len(png_files)}"
    print(f"✓ Extracted {len(png_files)} frames")

    # E3 Config UX (if available)
    if caps["commands"].get("init-config"):
        print("==> Running E3 config UX tests")
        config_path = run_dir / "config_open_field.yaml"
        run_cmd(
            ["uv", "run", "scindra-engine", "init-config", "--assay", "OPEN_FIELD", "--out", str(config_path)],
        )
        patch_config_file(config_path, good_video, run_dir / "results_good")
        if caps["commands"].get("validate-config"):
            run_cmd(
                ["uv", "run", "scindra-engine", "validate-config", "--config", str(config_path)],
            )
            print("✓ Config validated")

    # Track-Centroid (if available)
    if caps["commands"].get("track-centroid"):
        print("==> Running track-centroid tests")
        # Build command with optional flags
        cmd_good = [
            "uv",
            "run",
            "scindra-engine",
            "track-centroid",
            "--video",
            str(good_video),
            "--out",
            str(run_dir / "results_good"),
        ]
        if caps["flags"].get("overlay"):
            cmd_good.append("--overlay")
        if caps["flags"].get("heatmap"):
            cmd_good.append("--heatmap")
        run_cmd(cmd_good)

        # Validate good run
        results_good_dir = run_dir / "results_good"
        run_dir_good = find_newest_run_dir(results_good_dir)
        per_frame_csv = run_dir_good / "per_frame.csv"
        assert per_frame_csv.exists(), f"Missing {per_frame_csv}"
        assert per_frame_csv.stat().st_size > 0, f"Empty {per_frame_csv}"
        print(f"✓ per_frame.csv exists: {per_frame_csv}")

        if caps["flags"].get("overlay"):
            overlay_mp4 = run_dir_good / "overlay.mp4"
            assert overlay_mp4.exists(), f"Missing {overlay_mp4}"
            assert overlay_mp4.stat().st_size > 0, f"Empty {overlay_mp4}"
            print(f"✓ overlay.mp4 exists: {overlay_mp4}")

        if caps["flags"].get("heatmap"):
            heatmap_png = run_dir_good / "heatmap.png"
            assert heatmap_png.exists(), f"Missing {heatmap_png}"
            assert heatmap_png.stat().st_size > 0, f"Empty {heatmap_png}"
            print(f"✓ heatmap.png exists: {heatmap_png}")

        # Run on ambiguous fixture
        cmd_ambiguous = [
            "uv",
            "run",
            "scindra-engine",
            "track-centroid",
            "--video",
            str(ambiguous_video),
            "--out",
            str(run_dir / "results_ambiguous"),
        ]
        if caps["flags"].get("overlay"):
            cmd_ambiguous.append("--overlay")
        if caps["flags"].get("heatmap"):
            cmd_ambiguous.append("--heatmap")
        run_cmd(cmd_ambiguous)

        # Validate ambiguous run
        results_ambiguous_dir = run_dir / "results_ambiguous"
        run_dir_ambiguous = find_newest_run_dir(results_ambiguous_dir)
        per_frame_csv_ambiguous = run_dir_ambiguous / "per_frame.csv"
        assert per_frame_csv_ambiguous.exists(), f"Missing {per_frame_csv_ambiguous}"
        assert per_frame_csv_ambiguous.stat().st_size > 0, f"Empty {per_frame_csv_ambiguous}"
        print(f"✓ ambiguous per_frame.csv exists: {per_frame_csv_ambiguous}")

        # Check for manifest.json (E6.2+)
        manifest_path = run_dir_ambiguous / "manifest.json"
        if manifest_path.exists():
            print("==> Validating manifest.json")
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            required_keys = [
                "engine_version",
                "run_id",
                "input_files",
                "outputs",
                "warnings",
                "needs_review",
                "review_reasons",
                "confidence",
            ]
            for key in required_keys:
                assert key in manifest_data, f"Missing required key in manifest: {key}"
            print("✓ manifest.json has all required keys")

            # Check needs_review for ambiguous run
            needs_review = manifest_data.get("needs_review", False)
            if needs_review:
                print("✓ ambiguous run correctly flagged as needs_review")
            else:
                print("Note: ambiguous run not flagged as needs_review (may not be implemented yet)")

            # Check for support_bundle.zip
            support_bundle_path = run_dir_ambiguous / "support_bundle.zip"
            if support_bundle_path.exists() and needs_review:
                print("==> Validating support_bundle.zip")
                assert support_bundle_path.stat().st_size < 5 * 1024 * 1024, "Support bundle too large (>5MB)"
                # Check that video is NOT included
                with zipfile.ZipFile(support_bundle_path, "r") as zf:
                    video_names = [name for name in zf.namelist() if ambiguous_video.name in name]
                    assert not video_names, f"Raw video found in support bundle: {video_names}"
                print("✓ support_bundle.zip validated (no raw video)")

    # Auto-Setup (if available)
    if caps["commands"].get("auto-setup"):
        print("==> Running auto-setup tests")
        autosetup_dir = run_dir / "autosetup_good"
        run_cmd(
            [
                "uv",
                "run",
                "scindra-engine",
                "auto-setup",
                "--video",
                str(good_video),
                "--out",
                str(autosetup_dir),
                "--json",
            ],
        )
        arena_json = autosetup_dir / "arena.json"
        assay_guess_json = autosetup_dir / "assay_guess.json"
        assert arena_json.exists(), f"Missing {arena_json}"
        assert assay_guess_json.exists(), f"Missing {assay_guess_json}"
        # Check for confidence fields
        arena_data = json.loads(arena_json.read_text(encoding="utf-8"))
        assay_data = json.loads(assay_guess_json.read_text(encoding="utf-8"))
        assert "confidence" in arena_data or "confidence" in assay_data, "Missing confidence field"
        print("✓ auto-setup outputs validated")

    # Zones (if integrated)
    if caps["commands"].get("track-centroid"):
        # Check if zone column exists in per_frame.csv
        results_good_dir = run_dir / "results_good"
        run_dir_good = find_newest_run_dir(results_good_dir)
        per_frame_csv = run_dir_good / "per_frame.csv"
        if per_frame_csv.exists():
            with per_frame_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                if "zone" in header:
                    print("✓ Zones detected in per_frame.csv")

    # Batch (if available)
    if caps["commands"].get("batch"):
        print("==> Running batch tests")
        batch_in = run_dir / "batch_in"
        batch_in.mkdir(parents=True, exist_ok=True)
        # Copy both fixtures to batch input
        shutil.copy2(good_video, batch_in / good_video.name)
        shutil.copy2(ambiguous_video, batch_in / ambiguous_video.name)
        batch_out = run_dir / "batch_out"
        run_cmd(
            ["uv", "run", "scindra-engine", "batch", "--in", str(batch_in), "--out", str(batch_out)],
        )
        batch_summary = batch_out / "batch_summary.csv"
        needs_review_csv = batch_out / "needs_review.csv"
        assert batch_summary.exists(), f"Missing {batch_summary}"
        assert batch_summary.stat().st_size > 0, f"Empty {batch_summary}"
        assert needs_review_csv.exists(), f"Missing {needs_review_csv}"
        # Check for at least one run folder
        run_folders = list(batch_out.glob("run_*"))
        assert run_folders, "No run folders found in batch output"
        print("✓ Batch outputs validated")

    # Optional real video
    video_path_env = os.environ.get("VIDEO_PATH")
    strict_real_video = os.environ.get("STRICT_REAL_VIDEO") == "1"
    if video_path_env:
        print("==> Testing with real video (optional)")
        real_video_path = Path(video_path_env)
        if not real_video_path.exists():
            msg = f"VIDEO_PATH does not exist: {real_video_path}"
            if strict_real_video:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")
        else:
            try:
                result = subprocess.run(
                    ["uv", "run", "scindra-engine", "probe", "--video", str(real_video_path), "--json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                (run_dir / "probe_real.json").write_text(result.stdout, encoding="utf-8")
                if caps["commands"].get("track-centroid"):
                    cmd_real = [
                        "uv",
                        "run",
                        "scindra-engine",
                        "track-centroid",
                        "--video",
                        str(real_video_path),
                        "--out",
                        str(run_dir / "results_real"),
                    ]
                    if caps["flags"].get("overlay"):
                        cmd_real.append("--overlay")
                    if caps["flags"].get("heatmap"):
                        cmd_real.append("--heatmap")
                    run_cmd(cmd_real)
                print("✓ Real video test completed")
            except Exception as e:
                msg = f"Real video test failed: {e}"
                if strict_real_video:
                    raise RuntimeError(msg) from e
                print(f"WARNING: {msg}")

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE_LATEST OK")
    print(f"Capabilities detected: {list(caps['commands'].keys())}")
    print(f"Run directory: {run_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
