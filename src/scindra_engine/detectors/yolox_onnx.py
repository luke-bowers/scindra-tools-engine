"""YOLOX ONNX detector backend (requires ``onnxruntime``)."""
from __future__ import annotations

import numpy as np

from scindra_engine.detectors.base import (
    Detection,
    DetectorResult,
    ModelMeta,
)
from scindra_engine.detectors.nms import nms


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize *image* with letterboxing to *target_size* ``(H, W)``.

    Returns:
        ``(padded_image, scale, (pad_top, pad_left))``
    """
    ih, iw = image.shape[:2]
    th, tw = target_size
    scale = min(tw / iw, th / ih)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))

    # Use cv2 if available (already a dependency), else fall back to numpy
    import cv2  # type: ignore[import-untyped]

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (th - new_h) // 2
    pad_left = (tw - new_w) // 2

    if image.ndim == 3:
        padded = np.full((th, tw, image.shape[2]), pad_value, dtype=np.uint8)
    else:
        padded = np.full((th, tw), pad_value, dtype=np.uint8)

    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return padded, scale, (pad_top, pad_left)


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Full preprocessing pipeline: letterbox, BGR->RGB, HWC->CHW, float32.

    YOLOX expects pixel values in the **0-255** float32 range (no /255
    normalisation).  This matches the standard YOLOX training pipeline.

    Args:
        frame_bgr: Input frame in BGR ``uint8``.
        input_size: Target ``(H, W)`` for the network.

    Returns:
        ``(tensor, scale, (pad_top, pad_left))`` where *tensor* has shape
        ``(1, 3, H, W)`` in ``float32``.
    """
    padded, scale, pad = letterbox(frame_bgr, input_size)

    # BGR -> RGB
    rgb = padded[:, :, ::-1].copy()

    # HWC -> CHW
    chw = rgb.transpose(2, 0, 1)

    # float32 -- keep 0..255 range (YOLOX convention, no /255 normalisation)
    tensor = chw.astype(np.float32)

    # Add batch dimension
    return np.expand_dims(tensor, axis=0), scale, pad


def preprocess_frames(
    frames_bgr: list[np.ndarray],
    input_size: tuple[int, int],
) -> tuple[np.ndarray, list[float], list[tuple[int, int]], list[tuple[int, int]]]:
    """Preprocess multiple frames for batch inference.

    Returns:
        batched_tensor: shape (B, 3, H, W) float32.
        scales: length-B list of scale factors.
        pads: length-B list of (pad_top, pad_left).
        orig_hw_list: length-B list of (height, width) for each frame.
    """
    if not frames_bgr:
        return (
            np.zeros((0, 3, input_size[0], input_size[1]), dtype=np.float32),
            [],
            [],
            [],
        )
    tensors: list[np.ndarray] = []
    scales: list[float] = []
    pads: list[tuple[int, int]] = []
    orig_hw_list: list[tuple[int, int]] = []
    for frame in frames_bgr:
        tensor, scale, pad = preprocess_frame(frame, input_size)
        tensors.append(tensor)
        scales.append(scale)
        pads.append(pad)
        orig_hw_list.append(frame.shape[:2])
    batched = np.concatenate(tensors, axis=0)
    return batched, scales, pads, orig_hw_list


# ---------------------------------------------------------------------------
# YOLOX ONNX Detector
# ---------------------------------------------------------------------------

class YOLOXOnnxDetector:
    """YOLOX object detector backed by an ONNX Runtime session.

    The ``onnxruntime`` import happens lazily in ``__init__`` so that the
    module can be imported without the optional dependency installed.
    """

    def __init__(
        self,
        model_path: str,
        meta: ModelMeta,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        if providers is None:
            # Prefer GPU (CUDA) when available, otherwise fall back to CPU.
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                if "CPUExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CUDAExecutionProvider"]
            elif "CPUExecutionProvider" in available:
                providers = ["CPUExecutionProvider"]
            else:
                # As a last resort, let ORT decide based on whatever providers it has.
                providers = available or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._meta = meta
        self._input_name: str = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return "YOLOX_ONNX"

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> DetectorResult:
        """Run detection on a single BGR frame."""
        results = self.detect_batch([frame_bgr])
        return results[0]

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[DetectorResult]:
        """Run detection on a batch of BGR frames in one session.run.

        More efficient than calling detect() repeatedly when the ONNX model
        supports a batch dimension (e.g. input shape (batch, 3, H, W)).
        """
        if not frames_bgr:
            return []
        if len(frames_bgr) == 1:
            tensor, scale, pad = preprocess_frame(
                frames_bgr[0], self._meta.input_size
            )
            outputs = self._session.run(None, {self._input_name: tensor})
            raw = outputs[0]
            return [
                self._postprocess(raw, scale, pad, frames_bgr[0].shape[:2])
            ]
        batched, scales, pads, orig_hw_list = preprocess_frames(
            frames_bgr, self._meta.input_size
        )
        try:
            outputs = self._session.run(None, {self._input_name: batched})
        except Exception:
            # Model may have fixed batch size 1; fall back to per-frame
            return [self.detect(f) for f in frames_bgr]
        raw = outputs[0]  # (B, N, 5 + num_classes)
        return [
            self._postprocess(
                raw[i : i + 1], scales[i], pads[i], orig_hw_list[i]
            )
            for i in range(len(frames_bgr))
        ]

    # -----------------------------------------------------------------------
    # Postprocessing
    # -----------------------------------------------------------------------

    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_hw: tuple[int, int],
    ) -> DetectorResult:
        """Decode YOLOX raw output into ``DetectorResult``."""
        preds = raw[0]  # (N, 5 + num_classes)

        if preds.ndim != 2 or preds.shape[1] < 6:
            return DetectorResult(
                detections=[],
                best=None,
                confidence=0.0,
                reasons=["INVALID_OUTPUT_SHAPE"],
            )

        # Columns: cx, cy, w, h, obj_conf, class_conf_0, ...
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        obj_conf = preds[:, 4]
        class_confs = preds[:, 5:]

        # Best class per row
        class_ids = class_confs.argmax(axis=1)
        class_scores = class_confs[np.arange(len(class_confs)), class_ids]

        # Final score = obj_conf * class_conf
        scores = obj_conf * class_scores

        # Filter by score threshold
        mask = scores >= self._meta.score_thresh
        if not np.any(mask):
            return DetectorResult(
                detections=[],
                best=None,
                confidence=0.0,
                reasons=["NO_DETECTIONS"],
            )

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # cxcywh -> xyxy (still in letterboxed space)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Undo letterbox: subtract pad, then divide by scale
        pad_top, pad_left = pad
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # Clamp to original image bounds
        orig_h, orig_w = orig_hw
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Stack for NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, scores, self._meta.nms_iou)

        if len(keep) == 0:
            return DetectorResult(
                detections=[],
                best=None,
                confidence=0.0,
                reasons=["NO_DETECTIONS"],
            )

        # Build Detection objects
        detections: list[Detection] = []
        for idx in keep:
            detections.append(
                Detection(
                    bbox_xyxy=(
                        int(round(boxes[idx, 0])),
                        int(round(boxes[idx, 1])),
                        int(round(boxes[idx, 2])),
                        int(round(boxes[idx, 3])),
                    ),
                    score=float(scores[idx]),
                    class_id=int(class_ids[idx]),
                )
            )

        best = detections[0]  # highest score (NMS preserves score order)
        reasons: list[str] = []
        if best.score < self._meta.score_thresh:
            reasons.append("LOW_SCORE")

        return DetectorResult(
            detections=detections,
            best=best,
            confidence=best.score,
            reasons=reasons,
        )
