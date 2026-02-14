"""Constant-velocity 2-D Kalman filter for single-point tracking."""

from __future__ import annotations

import cv2
import numpy as np


class KalmanPointTracker:
    """Wraps ``cv2.KalmanFilter`` with a constant-velocity motion model.

    State vector : ``[x, y, vx, vy]``
    Measurement  : ``[x, y]``

    Typical usage inside a tracking loop::

        tracker = KalmanPointTracker()
        for frame in frames:
            if tracker.initialized:
                tracker.predict()            # step 1
            ...
            if got_detection:
                if not tracker.initialized:
                    tracker.initialize(x, y)  # first-ever detection
                else:
                    tracker.update(x, y)      # step 2
            else:
                tracker.mark_no_measurement()
    """

    def __init__(
        self,
        process_noise: float = 4.0,
        measurement_noise: float = 4.0,
        gate_sigma: float = 4.0,
    ) -> None:
        self._kf = cv2.KalmanFilter(4, 2)

        # --- Transition (constant velocity) ---
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # --- Measurement ---
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # --- Noise covariances ---
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # --- Initial error covariance ---
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self._gate_sigma = gate_sigma
        self._initialized = False
        self._predicted = False  # True after predict(), before update()
        self._frames_without_measurement = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, x: float, y: float) -> None:
        """Set the initial state from the very first measurement."""
        self._kf.statePost = np.array(
            [[x], [y], [0.0], [0.0]], dtype=np.float32
        )
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self._initialized = True
        self._predicted = False
        self._frames_without_measurement = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Predict / Update
    # ------------------------------------------------------------------
    def predict(self) -> tuple[float, float]:
        """Run the prediction step and return ``(pred_x, pred_y)``."""
        if not self._initialized:
            raise RuntimeError("KalmanPointTracker has not been initialized")
        pred = self._kf.predict()
        self._predicted = True
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Incorporate a measurement and return the corrected ``(x, y)``."""
        meas = np.array([[x], [y]], dtype=np.float32)
        corrected = self._kf.correct(meas)
        self._frames_without_measurement = 0
        self._predicted = False
        return float(corrected[0, 0]), float(corrected[1, 0])

    def mark_no_measurement(self) -> None:
        """Call when no detection is available this frame."""
        self._frames_without_measurement += 1
        self._predicted = False

    @property
    def frames_without_measurement(self) -> int:
        return self._frames_without_measurement

    # ------------------------------------------------------------------
    # Gating helpers
    # ------------------------------------------------------------------
    @property
    def predicted_position(self) -> tuple[float, float] | None:
        """Return the predicted position after :meth:`predict`, or *None*."""
        if not self._initialized:
            return None
        # After predict(): statePre holds the prediction.
        # After update():  statePost holds the corrected estimate.
        state = self._kf.statePre if self._predicted else self._kf.statePost
        return (float(state[0, 0]), float(state[1, 0]))

    def gate_distance(self, x: float, y: float) -> float:
        """Mahalanobis-like distance from the predicted state to ``(x, y)``.

        Uses the innovation covariance ``S = H P H^T + R`` to scale the
        distance.  Returns the number of *sigma* (standard deviations).
        """
        if not self._initialized:
            return 0.0

        state = self._kf.statePre if self._predicted else self._kf.statePost
        pred = state[:2].flatten()

        H = self._kf.measurementMatrix
        P = (
            self._kf.errorCovPre
            if self._predicted
            else self._kf.errorCovPost
        )
        R = self._kf.measurementNoiseCov

        # Innovation covariance
        S = H @ P @ H.T + R
        innovation = np.array(
            [x - pred[0], y - pred[1]], dtype=np.float32
        )

        try:
            S_inv = np.linalg.inv(S)
            mahal_sq = float(innovation.T @ S_inv @ innovation)
            return float(np.sqrt(max(0.0, mahal_sq)))
        except np.linalg.LinAlgError:
            # Fallback to Euclidean
            return float(np.sqrt(float(innovation[0] ** 2 + innovation[1] ** 2)))

    def is_within_gate(self, x: float, y: float) -> bool:
        """Check whether ``(x, y)`` is within the gating threshold."""
        return self.gate_distance(x, y) <= self._gate_sigma

    def search_radius(
        self,
        sigma_multiplier: float = 3.0,
        min_radius: float = 20.0,
        max_radius: float = 80.0,
    ) -> float:
        """Return a search radius (pixels) derived from the state covariance.

        The radius is ``sigma_multiplier`` times the larger of the position
        standard deviations in x and y, clamped to ``[min_radius, max_radius]``.
        When the filter has not been initialised, ``min_radius`` is returned.
        """
        if not self._initialized:
            return min_radius
        P = self._kf.errorCovPre if self._predicted else self._kf.errorCovPost
        std_x = float(np.sqrt(max(0.0, float(P[0, 0]))))
        std_y = float(np.sqrt(max(0.0, float(P[1, 1]))))
        r = sigma_multiplier * max(std_x, std_y)
        return float(max(min_radius, min(max_radius, r)))
