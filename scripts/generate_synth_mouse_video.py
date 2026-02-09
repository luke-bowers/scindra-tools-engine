"""Generate a synthetic video with a moving mouse and shadow. Run from repo root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing tests.fixtures when run as scripts/generate_synth_mouse_video.py
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tests.fixtures.synth_mouse_shadow import write_synth_mouse_shadow_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic MP4 with a moving mouse and shadow blob."
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("out/cli_smoke/synth_mouse_shadow.mp4"),
        help="Output file or directory (default: out/cli_smoke/synth_mouse_shadow.mp4)",
    )
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=30,
        metavar="N",
        help="Number of frames (default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Frame width (default: 160)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=120,
        help="Frame height (default: 120)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Frames per second (default: 15.0)",
    )
    args = parser.parse_args()

    out = args.out
    if out.suffix.lower() != ".mp4":
        out = out / "synth_mouse_shadow.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    write_synth_mouse_shadow_video(
        out,
        num_frames=args.num_frames,
        size=(args.width, args.height),
        fps=args.fps,
    )
    print(out)


if __name__ == "__main__":
    main()
