"""Extract sample frames from input and overlay videos and compare dimensions/circle placement."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent / "out" / "run_20260228_035824_e7b51cbc"
    overlay_path = run_dir / "overlay.mp4"
    per_frame_path = run_dir / "per_frame.csv"

    # Get input video path from tracking_summary.json (or hardcode for this run)
    import json
    summary_path = run_dir / "tracking_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    input_path = Path(summary["input_video"])

    if not input_path.exists():
        print(f"Input video not found: {input_path}")
        sys.exit(1)
    if not overlay_path.exists():
        print(f"Overlay not found: {overlay_path}")
        sys.exit(1)

    # Load one frame from input and overlay at same index
    frame_idx = 2  # frame with a centroid
    cap_in = cv2.VideoCapture(str(input_path))
    cap_out = cv2.VideoCapture(str(overlay_path))

    meta_w_in = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    meta_h_in = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap_in.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok_in, frame_in = cap_in.read()
    cap_in.release()

    meta_w_out = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    meta_h_out = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok_out, frame_out = cap_out.read()
    cap_out.release()

    if not ok_in or frame_in is None:
        print("Failed to read input frame")
        sys.exit(1)
    if not ok_out or frame_out is None:
        print("Failed to read overlay frame")
        sys.exit(1)

    h_in, w_in = frame_in.shape[:2]
    h_out, w_out = frame_out.shape[:2]

    print("=== Input video ===")
    print(f"  Metadata: {meta_w_in} x {meta_h_in}")
    print(f"  Actual frame shape: {frame_in.shape} -> width={w_in}, height={h_in}")
    print("=== Overlay video ===")
    print(f"  Metadata: {meta_w_out} x {meta_h_out}")
    print(f"  Actual frame shape: {frame_out.shape} -> width={w_out}, height={h_out}")
    print("=== Match? ===")
    print(f"  Same dimensions: {(w_in, h_in) == (w_out, h_out)}")
    print(f"  Metadata vs actual (input): match={ (meta_w_in, meta_h_in) == (w_in, h_in)}")
    print(f"  Metadata vs actual (overlay): match={ (meta_w_out, meta_h_out) == (w_out, h_out)}")

    # Read centroid for this frame from per_frame.csv
    with open(per_frame_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["frame"]) == frame_idx:
                x_str, y_str = row.get("x", ""), row.get("y", "")
                if x_str and y_str:
                    x_pt, y_pt = float(x_str), float(y_str)
                    break
        else:
            x_pt = y_pt = None
    print(f"\n=== Centroid for frame {frame_idx} (from per_frame.csv) ===")
    print(f"  x={x_pt}, y={y_pt}")
    print(f"  In bounds input (0..{w_in}, 0..{h_in})? {0 <= x_pt < w_in and 0 <= y_pt < h_in if x_pt is not None else 'N/A'}")
    print(f"  In bounds overlay (0..{w_out}, 0..{h_out})? {0 <= x_pt < w_out and 0 <= y_pt < h_out if x_pt is not None else 'N/A'}")

    # Draw circle on input at (x_pt, y_pt) and save for visual comparison
    out_dir = run_dir / "compare_frames"
    out_dir.mkdir(exist_ok=True)
    if x_pt is not None and y_pt is not None:
        cx, cy = int(round(x_pt)), int(round(y_pt))
        frame_in_draw = frame_in.copy()
        cv2.circle(frame_in_draw, (cx, cy), 6, (0, 0, 255), -1)
        cv2.imwrite(str(out_dir / "input_with_circle.png"), frame_in_draw)
        print(f"\nSaved {out_dir / 'input_with_circle.png'} with circle at ({cx},{cy})")
    cv2.imwrite(str(out_dir / "input_frame.png"), frame_in)
    cv2.imwrite(str(out_dir / "overlay_frame.png"), frame_out)
    print(f"Saved input_frame.png and overlay_frame.png to {out_dir}")

    # Check if overlay circle position matches (x_pt, y_pt) by sampling overlay at that pixel
    if x_pt is not None and y_pt is not None:
        cx, cy = int(round(x_pt)), int(round(y_pt))
        if 0 <= cx < w_out and 0 <= cy < h_out:
            b, g, r = frame_out[cy, cx]
            print(f"\nOverlay pixel at ({cx},{cy}): BGR=({b},{g},{r}) (red circle = 0,0,255)")
        # If the circle appears squashed, the overlay might have been encoded with wrong SAR
        # or the frame we're reading from overlay might be scaled. Check frame sizes again.
        if (w_in, h_in) != (w_out, h_out):
            print("\n>>> Dimension mismatch: overlay frame size differs from input!")
        elif (meta_w_in, meta_h_in) != (w_in, h_in):
            print("\n>>> Input has rotation/metadata: metadata dimensions != actual frame shape.")


if __name__ == "__main__":
    main()
