"""Frame-difference motion accumulator for suppressing static foreground."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class MotionAccumulator:
    """Accumulates frame-to-frame motion over a sliding window.

    Maintains a ring buffer of the last *history_len* grayscale frames.
    For each new frame the maximum absolute pixel difference against all
    buffered frames is computed and thresholded to produce a binary
    "recently moved" mask.  A generous dilation ensures that the full body
    of a moving animal is captured even when only part of it moved between
    consecutive frames.
    """

    def __init__(
        self,
        history_len: int = 5,
        threshold: int = 15,
        dilate_ksize: int = 7,
        dilate_iters: int = 3,
    ) -> None:
        self._history_len = max(1, history_len)
        self._threshold = threshold
        if dilate_ksize >= 1 and dilate_iters >= 1:
            self._dilate_kernel: np.ndarray | None = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize)
            )
        else:
            self._dilate_kernel = None
        self._dilate_iters = dilate_iters
        self._buffer: deque[np.ndarray] = deque(maxlen=self._history_len)

    # ------------------------------------------------------------------
    def update(self, gray: np.ndarray) -> np.ndarray:
        """Feed a new grayscale frame and return the motion mask.

        The mask is ``uint8`` where **255** marks pixels that changed
        significantly in at least one of the buffered frames.

        On the very first call (empty buffer) a fully-white mask is
        returned so that the first frame is not discarded.
        """
        if len(self._buffer) == 0:
            self._buffer.append(gray.copy())
            return np.full(gray.shape[:2], 255, dtype=np.uint8)

        # Max absolute diff across the whole ring buffer
        max_diff = np.zeros(gray.shape[:2], dtype=np.uint8)
        for prev in self._buffer:
            diff = cv2.absdiff(gray, prev)
            np.maximum(max_diff, diff, out=max_diff)

        # Push current frame *after* comparison
        self._buffer.append(gray.copy())

        # Threshold → binary motion mask
        _, motion = cv2.threshold(
            max_diff, self._threshold, 255, cv2.THRESH_BINARY
        )

        # Dilate so that the full animal body is included
        if self._dilate_kernel is not None and self._dilate_iters > 0:
            motion = cv2.dilate(
                motion, self._dilate_kernel, iterations=self._dilate_iters
            )

        return motion
