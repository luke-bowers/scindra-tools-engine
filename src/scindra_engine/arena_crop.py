"""Arena/maze detection and crop box computation from static video frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np

from scindra_engine.video_io import FrameSampler, get_display_dimensions, resize_to_display_aspect

if TYPE_CHECKING:
    from scindra_engine.video_io import VideoReader


def build_static_image(
    reader: VideoReader,
    n_frames: int,
    method: str = "median",
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Build a static structure image by averaging over N frames.

    The mouse and temporary lighting are averaged out so static structures
    (maze walls, floor boundary) dominate. Used for arena detection.

    Args:
        reader: Open VideoReader.
        n_frames: Number of frames to sample (evenly spaced).
        method: "median" or "mean".
        progress_callback: Optional (current, total) progress callback.

    Returns:
        BGR image at original video resolution.
    """
    sampler = FrameSampler(reader)
    sampled = sampler.sample(n_frames, progress_callback=progress_callback)
    frames = [f for _, f in sampled]
    if not frames:
        raise ValueError("No frames could be sampled for static image")

    stack = np.stack(frames, axis=0).astype(np.float32)
    if method == "mean":
        combined = np.mean(stack, axis=0).astype(np.uint8)
    else:
        combined = np.median(stack, axis=0).astype(np.uint8)
    return combined


def _edges_for_detection(
    static_image_bgr: np.ndarray,
    canny_low: int,
    canny_high: int,
    blur_ksize: int,
    morph_close_ksize: int,
) -> np.ndarray:
    """Build edge map with optional morphological closing to connect gaps."""
    gray = (
        cv2.cvtColor(static_image_bgr, cv2.COLOR_BGR2GRAY)
        if static_image_bgr.ndim == 3
        else static_image_bgr
    )
    k = max(1, blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    if morph_close_ksize > 0:
        close_k = morph_close_ksize if morph_close_ksize % 2 == 1 else morph_close_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def _refine_circle_center_from_edges(
    edges: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    inner_ratio: float = 0.6,
    outer_ratio: float = 1.2,
) -> tuple[float, float]:
    """Refine circle center using centroid of edge pixels in an annulus around (cx, cy).

    Hough can bias the center when one side of the rim has stronger edges. This pulls
    the center toward where the edges actually are. Returns (cx, cy) as floats.
    """
    h, w = edges.shape[:2]
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    inner = int(r * inner_ratio)
    outer = int(r * outer_ratio)
    if outer <= inner or outer < 5:
        return (float(cx), float(cy))
    mask = (dist >= inner) & (dist <= outer) & (edges > 0)
    if not np.any(mask):
        return (float(cx), float(cy))
    ys, xs = np.where(mask)
    new_cx = float(np.mean(xs))
    new_cy = float(np.mean(ys))
    return (new_cx, new_cy)


def _detect_arena_hough(
    edges: np.ndarray,
    w: int,
    h: int,
    margin_px: int,
    min_radius_ratio: float,
    max_radius_ratio: float,
    center_margin_ratio: float,
    acc_threshold: int = 25,
    circle_padding_ratio: float = 0.0,
    force_square_crop: bool = True,
) -> tuple[
    tuple[int, int, int, int] | None,
    list[tuple[int, int, int]],
    tuple[int, int, int] | None,
]:
    """Detect a circular arena via Hough circles; return (bbox, all_passing_circles, chosen_circle)."""
    min_dim = min(w, h)
    min_r = int(min_dim * min_radius_ratio)
    max_r = int(min_dim * max_radius_ratio)
    empty: list[tuple[int, int, int]] = []
    if min_r < 10 or max_r <= min_r:
        return (None, empty, None)
    margin = center_margin_ratio
    x_lo, x_hi = int(w * margin), int(w * (1 - margin))
    y_lo, y_hi = int(h * margin), int(h * (1 - margin))
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dim * 0.3,
        param1=50,
        param2=acc_threshold,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None or circles.size == 0:
        return (None, empty, None)
    circles = np.uint16(np.around(circles))
    # Choose the largest circle by radius (outer rim of arena)
    best = None
    best_radius = -1
    passing: list[tuple[int, int, int]] = []
    for c in circles[0, :]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
            continue
        if r < min_r or r > max_r:
            continue
        passing.append((cx, cy, r))
        if r > best_radius:
            best_radius = r
            best = (cx, cy, r)
    if best is None:
        return (None, passing, None)
    cx, cy, r = best
    # Refine center using edge pixels in an annulus: Hough can be biased (e.g. stronger top arc).
    # Use centroid of edge pixels in the ring to pull center toward the actual rim.
    cx, cy = _refine_circle_center_from_edges(edges, cx, cy, r)
    cx_i, cy_i = int(round(cx)), int(round(cy))
    pad = int(r * circle_padding_ratio) if circle_padding_ratio > 0 else 0
    half = r + margin_px + pad
    if force_square_crop:
        # Force square crop so output is not a non-square rectangle when near frame edges.
        half = min(half, cx_i, w - cx_i, cy_i, h - cy_i)
        if half < 1:
            return (None, passing, best)
        x1 = cx_i - half
        y1 = cy_i - half
        x2 = cx_i + half
        y2 = cy_i + half
    else:
        x1 = max(0, cx_i - half)
        y1 = max(0, cy_i - half)
        x2 = min(w, cx_i + half)
        y2 = min(h, cy_i + half)
    if x2 <= x1 or y2 <= y1:
        return (None, passing, best)
    return ((x1, y1, x2, y2), passing, (cx_i, cy_i, r))


def _detect_arena_contour(
    edges: np.ndarray,
    w: int,
    h: int,
    margin_px: int,
    min_area_ratio: float,
) -> tuple[
    tuple[int, int, int, int] | None,
    list[float],
    tuple[int, int, int, int] | None,
]:
    """Detect arena from contours: choose the one with largest bounding box (span) so the outer rim wins over inner features. Return (bbox, candidate_areas, chosen_bbox)."""
    area_total = w * h
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = max(100, int(area_total * min_area_ratio))
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    areas = [cv2.contourArea(c) for c in candidates]
    if not candidates:
        return (None, [], None)
    # Use union of all candidate bounding boxes so outer rim arcs are included (they're separate contours with smaller area than inner features)
    rx1, ry1, rx2, ry2 = w, h, 0, 0
    for c in candidates:
        cx1, cy1, cw, ch = cv2.boundingRect(c)
        cx2, cy2 = cx1 + cw, cy1 + ch
        rx1 = min(rx1, cx1)
        ry1 = min(ry1, cy1)
        rx2 = max(rx2, cx2)
        ry2 = max(ry2, cy2)
    x1 = max(0, rx1 - margin_px)
    y1 = max(0, ry1 - margin_px)
    x2 = min(w, rx2 + margin_px)
    y2 = min(h, ry2 + margin_px)
    if x2 <= x1 or y2 <= y1:
        return (None, areas, None)
    chosen_bbox = (int(x1), int(y1), int(x2), int(y2))
    return ((int(x1), int(y1), int(x2), int(y2)), areas, chosen_bbox)


def _detect_arena_open_field(
    static_image_bgr: np.ndarray,
    w: int,
    h: int,
    margin_px: int,
    blur_ksize: int,
    open_field_white_threshold: int,
    open_field_min_area_ratio: float,
    open_field_rectangularity_min: float,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray, tuple[int, int, int, int] | None]:
    """Detect open-field arena as largest bright (white) region. Returns (box, mask, chosen_bbox)."""
    gray = (
        cv2.cvtColor(static_image_bgr, cv2.COLOR_BGR2GRAY)
        if static_image_bgr.ndim == 3
        else static_image_bgr
    )
    k = max(1, blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    if open_field_white_threshold > 0:
        _, binary = cv2.threshold(
            blurred, open_field_white_threshold, 255, cv2.THRESH_BINARY
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    # Light morphology: close with small kernel to fill tiny holes inside the box but not bridge a thin gap (e.g. between maze and tiles to the right)
    close_k = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_total = w * h
    min_area = max(100, int(area_total * open_field_min_area_ratio))
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x1, y1, cw, ch = cv2.boundingRect(c)
        bbox_area = cw * ch
        if bbox_area <= 0:
            continue
        rectangularity = area / bbox_area
        if rectangularity >= open_field_rectangularity_min:
            candidates.append((c, area, rectangularity))
    if not candidates:
        return (None, binary, None)
    # Prefer large and box-like: score = area * rectangularity so maze wins over merged maze+tiles
    best_contour = max(candidates, key=lambda x: x[1] * x[2])[0]
    x1, y1, cw, ch = cv2.boundingRect(best_contour)
    x2 = x1 + cw
    y2 = y1 + ch
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(w, x2 + margin_px)
    y2 = min(h, y2 + margin_px)
    if x2 <= x1 or y2 <= y1:
        return (None, binary, None)
    chosen_bbox = (int(x1), int(y1), int(x2), int(y2))
    return ((int(x1), int(y1), int(x2), int(y2)), binary, chosen_bbox)


def _detect_arena_elevated_plus(
    static_image_bgr: np.ndarray,
    w: int,
    h: int,
    margin_px: int,
    plus_maze_arm_length_ratio: float,
    plus_maze_arm_width_ratio: float,
    plus_maze_center_size_ratio: float,
    plus_maze_aspect_tolerance: float,
    plus_maze_min_area_ratio: float,
    canny_low: int,
    canny_high: int,
    blur_ksize: int,
    morph_close_ksize: int,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray, tuple[int, int, int, int] | None]:
    """Detect elevated plus maze as cross-shaped bright structure. Returns (box, edges, chosen_bbox)."""
    # Convert to grayscale
    gray = (
        cv2.cvtColor(static_image_bgr, cv2.COLOR_BGR2GRAY)
        if static_image_bgr.ndim == 3
        else static_image_bgr
    )
    
    # Apply blur
    k = max(1, blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Morphological operations to enhance cross structure
    if morph_close_ksize > 0:
        kernel_size = max(3, morph_close_ksize if morph_close_ksize % 2 == 1 else morph_close_ksize + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate expected dimensions
    min_dim = min(w, h)
    expected_arm_length = int(min_dim * plus_maze_arm_length_ratio)
    expected_arm_width = int(expected_arm_length * plus_maze_arm_width_ratio)
    expected_center_size = int(expected_arm_length * plus_maze_center_size_ratio)
    
    area_total = w * h
    min_area = max(500, int(area_total * plus_maze_min_area_ratio))
    
    # Look for plus-shaped contours
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Get bounding rectangle
        x1, y1, cw, ch = cv2.boundingRect(contour)
        
        # Basic size validation
        if cw < expected_arm_length * 0.5 or ch < expected_arm_length * 0.5:
            continue
            
        # Check if the contour could form a plus shape
        # A plus should have roughly equal width and height
        aspect_ratio = max(cw, ch) / min(cw, ch)
        if aspect_ratio > (1.0 + plus_maze_aspect_tolerance):
            continue
        
        # Calculate center point
        cx = x1 + cw // 2
        cy = y1 + ch // 2
        
        # Create a mask for this contour to analyze its shape
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Check for plus-like properties by analyzing structure
        # Sample points along horizontal and vertical lines through center
        h_line_coverage = np.sum(mask[cy, max(0, cx - cw//2):min(w, cx + cw//2)] > 0) / min(cw, w - max(0, cx - cw//2))
        v_line_coverage = np.sum(mask[max(0, cy - ch//2):min(h, cy + ch//2), cx] > 0) / min(ch, h - max(0, cy - ch//2))
        
        # A plus should have good coverage along both axes
        cross_score = min(h_line_coverage, v_line_coverage)
        
        # Prefer contours with good cross coverage and appropriate size
        score = area * cross_score * (1.0 / aspect_ratio)  # Penalty for non-square shapes
        candidates.append((contour, area, score, x1, y1, cw, ch))
    
    if not candidates:
        return (None, edges, None)
    
    # Choose the best candidate (highest score)
    best_contour, best_area, best_score, x1, y1, cw, ch = max(candidates, key=lambda x: x[2])
    
    # Calculate final bounding box with margins
    x2 = x1 + cw
    y2 = y1 + ch
    x1 = max(0, x1 - margin_px)
    y1 = max(0, y1 - margin_px)
    x2 = min(w, x2 + margin_px)
    y2 = min(h, y2 + margin_px)
    
    if x2 <= x1 or y2 <= y1:
        return (None, edges, None)
    
    chosen_bbox = (int(x1), int(y1), int(x2), int(y2))
    return ((int(x1), int(y1), int(x2), int(y2)), edges, chosen_bbox)


def detect_arena_crop_xyxy(
    static_image_bgr: np.ndarray,
    margin_px: int = 0,
    min_area_ratio: float = 0.05,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_ksize: int = 5,
    morph_close_ksize: int = 0,
    use_hough_circle: bool = True,
    hough_min_radius_ratio: float = 0.08,
    hough_max_radius_ratio: float = 0.48,
    hough_center_margin_ratio: float = 0.15,
    hough_acc_threshold: int = 25,
    circle_only: bool = False,
    circle_padding_ratio: float = 0.0,
    force_square_crop: bool = True,
    debug_callback: Callable[[str, dict[str, Any]], None] | None = None,
    dar: str | None = None,
    arena_type: str = "elevated_zero",
    open_field_white_threshold: int = 200,
    open_field_min_area_ratio: float = 0.02,
    open_field_rectangularity_min: float = 0.6,
    plus_maze_arm_length_ratio: float = 0.3,
    plus_maze_arm_width_ratio: float = 0.2,
    plus_maze_center_size_ratio: float = 0.15,
    plus_maze_aspect_tolerance: float = 0.3,
    plus_maze_min_area_ratio: float = 0.08,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int] | None]:
    """Detect the most likely arena bounding box from a static BGR image.

    Returns (box, chosen_circle). chosen_circle is (cx, cy, r) when detection used
    a Hough circle (so the caller can apply a circular arena mask); None otherwise.

    Step-by-step process:
    --------
    1. Static image
       Input is typically a median/mean over N frames so the arena rim is
       stable and the mouse/noise are averaged out.

    2. Edges
       Grayscale → Gaussian blur (blur_ksize) → Canny (canny_low, canny_high).
       Optionally morphological close (morph_close_ksize) to connect gaps in
       the arena boundary.

    3a. Circles path (use_hough_circle=True)
        Run Hough circle detection on the edge image. Filter candidates by:
        - Radius in [min_radius_ratio, max_radius_ratio] × min(width, height)
        - Center inside frame (center_margin_ratio from edges).
        Choose the circle with the LARGEST radius among those that pass
        (the outer rim of the arena). Crop box = square (center ± radius)
        plus margin_px. This is "find circles, crop to the largest one."

    3b. Contour fallback (when circle_only=False and no circle found)
        Find contours on the edge map; keep those with area ≥ min_area_ratio
        of frame. Take the LARGEST by area and use its axis-aligned bounding
        rect plus margin_px. This can fit irregular or non-circular shapes
        but may be tighter than the outer rim.

    4. Optional expand
        Caller may apply expand_arena_box_xyxy so the box includes the
        outer edge of the arena (see crop_expand_ratio in config).

    Args:
        static_image_bgr: Static scene image (BGR) at original resolution.
        margin_px: Pixels to add to each side (negative = shrink).
        min_area_ratio: Minimum contour area as fraction of frame area (contour path).
        canny_low, canny_high, blur_ksize: Edge detection parameters.
        morph_close_ksize: If > 0, close edges with an ellipse kernel to connect gaps.
        use_hough_circle: If True, try Hough circle detection first (find circles, crop to largest).
        hough_*: Hough circle filtering (radius and center margins).
        hough_acc_threshold: Hough accumulator threshold; lower = more circles (sensitivity).
        circle_only: If True, never use contour fallback; return None when no circle found.
        debug_callback: If set, called at each step with (step_id, data) for debug output.

    Returns:
        (box, chosen_circle): box is (x1, y1, x2, y2) or None; chosen_circle is (cx, cy, r) when from Hough, else None.
    """
    h, w = static_image_bgr.shape[:2]

    # Open-field path: segment white region, largest contour → bbox
    if arena_type == "open_field":
        box, mask, chosen_bbox = _detect_arena_open_field(
            static_image_bgr,
            w,
            h,
            margin_px,
            blur_ksize,
            open_field_white_threshold,
            open_field_min_area_ratio,
            open_field_rectangularity_min,
        )
        if debug_callback is not None:
            overlay = static_image_bgr.copy()
            if chosen_bbox is not None:
                x1, y1, x2, y2 = chosen_bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    "Arena crop (detected)",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            debug_callback("open_field", {
                "image": overlay,
                "mask": mask,
                "chosen_bbox": list(chosen_bbox) if chosen_bbox else None,
                "description": "Open-field: white region mask; green box = chosen crop",
            })
        return (box, None)
    
    # Elevated-plus path: edge detection + cross pattern analysis
    if arena_type == "elevated_plus":
        box, edges, chosen_bbox = _detect_arena_elevated_plus(
            static_image_bgr,
            w,
            h,
            margin_px,
            plus_maze_arm_length_ratio,
            plus_maze_arm_width_ratio,
            plus_maze_center_size_ratio,
            plus_maze_aspect_tolerance,
            plus_maze_min_area_ratio,
            canny_low,
            canny_high,
            blur_ksize,
            morph_close_ksize,
        )
        if debug_callback is not None:
            overlay = static_image_bgr.copy()
            if chosen_bbox is not None:
                x1, y1, x2, y2 = chosen_bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    "Plus maze crop (detected)",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            debug_callback("elevated_plus", {
                "image": overlay,
                "mask": edges,
                "chosen_bbox": list(chosen_bbox) if chosen_bbox else None,
                "description": "Elevated-plus: edge map; green box = chosen crop",
            })
        return (box, None)

    # Step: edges (optionally show raw edges before morph)
    if morph_close_ksize > 0 and debug_callback is not None:
        edges_raw = _edges_for_detection(
            static_image_bgr, canny_low, canny_high, blur_ksize, 0
        )
        debug_callback("edges_raw", {"image": edges_raw, "description": "Canny edges before morphological close"})
    edges = _edges_for_detection(
        static_image_bgr, canny_low, canny_high, blur_ksize, morph_close_ksize
    )
    if debug_callback is not None:
        debug_callback("edges", {
            "image": edges,
            "morph_close_ksize": morph_close_ksize,
            "description": "Edge map used for detection" + (" (after morph close)" if morph_close_ksize > 0 else ""),
        })

    box: tuple[int, int, int, int] | None = None

    if use_hough_circle:
        box, passing_circles, chosen_circle = _detect_arena_hough(
            edges,
            w,
            h,
            margin_px,
            hough_min_radius_ratio,
            hough_max_radius_ratio,
            hough_center_margin_ratio,
            hough_acc_threshold,
            circle_padding_ratio,
            force_square_crop,
        )
        if debug_callback is not None:
            # When dar is set, draw on DAR-resized image with scaled coords so circles render round (not oval).
            if dar:
                overlay = resize_to_display_aspect(static_image_bgr.copy(), dar)
                rw, rh = overlay.shape[1], overlay.shape[0]
                sx, sy = rw / w, rh / h
                rs = min(sx, sy)  # radius scale so circle stays round in display space
            else:
                overlay = static_image_bgr.copy()
                rw, rh = w, h
                sx, sy, rs = 1.0, 1.0, 1.0
            for (cx, cy, r) in passing_circles:
                color = (0, 255, 0) if chosen_circle and (cx, cy, r) == chosen_circle else (128, 128, 128)
                cx_d, cy_d = int(round(cx * sx)), int(round(cy * sy))
                r_d = max(2, int(round(r * rs)))
                cv2.circle(overlay, (cx_d, cy_d), r_d, color, 2)
            if chosen_circle is not None and box is not None:
                cx, cy, r = chosen_circle
                x1, y1, x2, y2 = box
                x1_d = int(round(x1 * sx))
                y1_d = int(round(y1 * sy))
                x2_d = int(round(x2 * sx))
                y2_d = int(round(y2 * sy))
                cv2.rectangle(overlay, (x1_d, y1_d), (x2_d, y2_d), (0, 255, 0), 2)
            debug_callback("hough", {
                "image": overlay,
                "circles_found": len(passing_circles),
                "circles": [{"cx": cx, "cy": cy, "r": r} for (cx, cy, r) in passing_circles],
                "chosen": {"cx": chosen_circle[0], "cy": chosen_circle[1], "r": chosen_circle[2]} if chosen_circle else None,
                "box": list(box) if box else None,
                "description": "Hough circles (green = chosen, gray = others); green box = crop" if chosen_circle else "Hough found no circle passing filters",
            })
        if box is not None:
            return (box, chosen_circle)
        if circle_only:
            return (None, None)

    box, areas, chosen_bbox = _detect_arena_contour(edges, w, h, margin_px, min_area_ratio)
    if debug_callback is not None:
        overlay = static_image_bgr.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_total = w * h
        min_area = max(100, int(area_total * min_area_ratio))
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(overlay, [c], -1, (128, 128, 255), 2)
        if chosen_bbox is not None:
            x1, y1, x2, y2 = chosen_bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        debug_callback("contour", {
            "image": overlay,
            "num_candidates": len(areas),
            "areas": areas,
            "chosen_bbox": list(chosen_bbox) if chosen_bbox else None,
            "description": "Contour fallback: candidates in purple, chosen bbox in green",
        })
    return (box, None)


def expand_arena_box_xyxy(
    box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand the arena box so the outer edge of the arena is included.

    Grows the box by expand_ratio of its width (left/right) and height (top/bottom),
    then clamps to frame bounds.

    Args:
        box: (x1, y1, x2, y2) in pixel coordinates.
        frame_w: Frame width.
        frame_h: Frame height.
        expand_ratio: Fraction of box dimensions to add on each side (e.g. 0.05 = 5%).

    Returns:
        (x1, y1, x2, y2) expanded and clamped to [0, frame_w] x [0, frame_h].
    """
    if expand_ratio <= 0:
        return box
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = (expand_ratio * bw) / 2
    dy = (expand_ratio * bh) / 2
    x1_new = max(0, x1 - dx)
    y1_new = max(0, y1 - dy)
    x2_new = min(frame_w, x2 + dx)
    y2_new = min(frame_h, y2 + dy)
    return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))


def get_arena_detection_edges(
    static_image_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Return the raw edge map (Canny only, no morph) for debugging."""
    gray = (
        cv2.cvtColor(static_image_bgr, cv2.COLOR_BGR2GRAY)
        if static_image_bgr.ndim == 3
        else static_image_bgr
    )
    k = max(1, blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.Canny(blurred, canny_low, canny_high)


def get_arena_detection_edges_with_close(
    static_image_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_ksize: int = 5,
    morph_close_ksize: int = 0,
) -> np.ndarray:
    """Return the edge map as used for detection (with optional morphological close)."""
    return _edges_for_detection(
        static_image_bgr, canny_low, canny_high, blur_ksize, morph_close_ksize
    )


def crop_frame(
    frame: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop a frame to the given box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = crop_xyxy
    return frame[y1:y2, x1:x2].copy()
