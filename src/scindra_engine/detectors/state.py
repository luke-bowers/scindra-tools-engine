"""Per-run detector state for ROI tracking with hysteresis."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import numpy as np

from scindra_engine.detectors.base import Detector, DetectorResult
from scindra_engine.schemas import DetectorConfig


@dataclass
class FrameDetectorInfo:
    """Per-frame detector metadata (stored alongside each TrackPoint)."""

    roi_xyxy: tuple[int, int, int, int] | None = None
    detector_score: float | None = None
    detector_used: bool = False


class DetectorState:
    """Manages when to invoke the detector and maintains ROI state.

    This class encapsulates:

    * **scheduling** – decides *when* to run the detector
    * **ROI tracking** – maintains the last-known bounding box and produces a
      padded ROI for classical segmentation
    * **hysteresis** – a detection that jumps too far from the previous bbox is
      not trusted until it persists for 2 consecutive frames

    The caller feeds frames in via :meth:`step` and receives a
    :class:`FrameDetectorInfo` describing the ROI to use (or ``None`` for
    full-frame).
    """

    def __init__(
        self,
        detector: Detector,
        config: DetectorConfig,
    ) -> None:
        self._detector = detector
        self._cfg = config

        # Persistent state across frames
        self.last_bbox: tuple[int, int, int, int] | None = None
        self.last_detection_score: float | None = None
        self.frames_since_detect: int = 0
        self.consecutive_missing: int = 0

        # Hysteresis state for large jumps
        self._pending_jump_bbox: tuple[int, int, int, int] | None = None
        self._pending_jump_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_run(
        self,
        frame_idx: int,
        tracking_conf: float,
        has_centroid: bool,
    ) -> bool:
        """Decide whether to invoke the detector on *frame_idx*."""
        # Always run on the very first frame
        if frame_idx == 0:
            return True

        # Periodic schedule
        if self.frames_since_detect >= self._cfg.every_n_frames:
            return True

        # Lost tracking for several frames
        if not has_centroid:
            self.consecutive_missing += 1
        else:
            self.consecutive_missing = 0
        if self.consecutive_missing >= 3:
            return True

        # Tracking confidence is low
        if (
            self._cfg.reacquire_on_low_tracking_conf
            and tracking_conf < self._cfg.min_score
        ):
            return True

        return False

    def step(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        tracking_conf: float,
        has_centroid: bool,
    ) -> FrameDetectorInfo:
        """Process a single frame: optionally run detector and return ROI info.

        Args:
            frame_bgr: Full-resolution BGR frame.
            frame_idx: 0-based frame index.
            tracking_conf: Confidence of the *previous* frame's tracking result.
            has_centroid: Whether the previous frame had a valid centroid.

        Returns:
            :class:`FrameDetectorInfo` with the ROI (if any) and score.
        """
        run = self.should_run(frame_idx, tracking_conf, has_centroid)
        info = FrameDetectorInfo()

        if run:
            result = self._detector.detect(frame_bgr)
            info.detector_used = True
            info.detector_score = result.confidence
            self._apply_result(result, frame_bgr.shape[:2])
        else:
            self.frames_since_detect += 1

        # Compute padded ROI from current bbox
        if self.last_bbox is not None:
            info.roi_xyxy = self._padded_roi(
                self.last_bbox, frame_bgr.shape[:2]
            )

        return info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_result(
        self,
        result: DetectorResult,
        frame_hw: tuple[int, int],
    ) -> None:
        """Update state from a DetectorResult, with jump hysteresis."""
        if result.best is not None and result.best.score >= self._cfg.min_score:
            new_bbox = result.best.bbox_xyxy
            self.last_detection_score = result.best.score

            if self.last_bbox is not None:
                jump = self._bbox_center_dist(self.last_bbox, new_bbox)
                if jump > self._cfg.max_roi_jump_px:
                    # Hysteresis: require 2 consecutive frames
                    if (
                        self._pending_jump_bbox is not None
                        and self._bbox_center_dist(
                            self._pending_jump_bbox, new_bbox
                        )
                        < self._cfg.max_roi_jump_px
                    ):
                        self._pending_jump_count += 1
                    else:
                        self._pending_jump_bbox = new_bbox
                        self._pending_jump_count = 1

                    if self._pending_jump_count >= 2:
                        # Accept the jump
                        self.last_bbox = new_bbox
                        self._pending_jump_bbox = None
                        self._pending_jump_count = 0
                    # else: keep old bbox, do not update
                else:
                    # Normal update, no large jump
                    self.last_bbox = new_bbox
                    self._pending_jump_bbox = None
                    self._pending_jump_count = 0
            else:
                # First detection – accept unconditionally
                self.last_bbox = new_bbox
                self._pending_jump_bbox = None
                self._pending_jump_count = 0

            self.frames_since_detect = 0
        else:
            # No good detection – do NOT clear last_bbox (keep stale ROI)
            self.frames_since_detect += 1

    def _padded_roi(
        self,
        bbox: tuple[int, int, int, int],
        frame_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Expand *bbox* by configured padding and clamp to frame bounds."""
        x1, y1, x2, y2 = bbox
        fh, fw = frame_hw

        if self._cfg.roi_padding_scale is not None:
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            new_bw = bw * self._cfg.roi_padding_scale
            new_bh = bh * self._cfg.roi_padding_scale
            x1 = int(cx - new_bw / 2.0)
            y1 = int(cy - new_bh / 2.0)
            x2 = int(cx + new_bw / 2.0)
            y2 = int(cy + new_bh / 2.0)
        else:
            pad = self._cfg.roi_padding_px
            x1 -= pad
            y1 -= pad
            x2 += pad
            y2 += pad

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw, x2)
        y2 = min(fh, y2)

        return (x1, y1, x2, y2)

    @staticmethod
    def _bbox_center_dist(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        acx = (a[0] + a[2]) / 2.0
        acy = (a[1] + a[3]) / 2.0
        bcx = (b[0] + b[2]) / 2.0
        bcy = (b[1] + b[3]) / 2.0
        return hypot(acx - bcx, acy - bcy)
