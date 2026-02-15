"""Base types and model resolver for the optional detector subsystem."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A single object detection."""

    bbox_xyxy: tuple[int, int, int, int]
    score: float
    class_id: int


@dataclass(frozen=True)
class DetectorResult:
    """Result of running a detector on a single frame."""

    detections: list[Detection]
    best: Detection | None
    confidence: float
    reasons: list[str]


@dataclass(frozen=True)
class ModelMeta:
    """Metadata describing the ONNX model expectations."""

    input_size: tuple[int, int] = (640, 640)
    num_classes: int = 1
    class_names: list[str] = field(default_factory=lambda: ["mouse"])
    score_thresh: float = 0.25
    nms_iou: float = 0.45


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Detector(Protocol):
    """Protocol that any detector backend must satisfy."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    def detect(self, frame_bgr: np.ndarray) -> DetectorResult: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

_PACKAGED_ASSET = "scindra_engine/assets/models/yolox_mouse_640.onnx"


def _load_meta_sidecar(model_path: Path) -> ModelMeta:
    """Load a ``.json`` sidecar next to the ONNX file, or return defaults."""
    meta_path = model_path.with_suffix(".json")
    if not meta_path.is_file():
        return ModelMeta()
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ModelMeta()

    input_size = tuple(raw.get("input_size", [640, 640]))
    if len(input_size) != 2:
        input_size = (640, 640)
    return ModelMeta(
        input_size=(int(input_size[0]), int(input_size[1])),
        num_classes=int(raw.get("num_classes", 1)),
        class_names=list(raw.get("class_names", ["mouse"])),
        score_thresh=float(raw.get("score_thresh", 0.25)),
        nms_iou=float(raw.get("nms_iou", 0.45)),
    )


class ModelResolver:
    """Resolve the ONNX model path using a priority chain.

    Resolution order:
    1. Explicit *model_path* argument (e.g. ``--detector-model`` CLI flag)
    2. ``SCINDRA_YOLOX_ONNX_PATH`` environment variable
    3. Packaged asset shipped inside the wheel
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._explicit = model_path

    def resolve(self) -> tuple[Path, ModelMeta] | None:
        """Return ``(model_path, meta)`` or ``None`` when unavailable."""
        # 1. Explicit path
        if self._explicit is not None:
            p = Path(self._explicit)
            if p.is_file():
                return p, _load_meta_sidecar(p)
            return None  # explicit path given but missing

        # 2. Environment variable
        env_path = os.environ.get("SCINDRA_YOLOX_ONNX_PATH")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                return p, _load_meta_sidecar(p)

        # 3. Packaged asset
        try:
            ref = resources.files("scindra_engine").joinpath(
                "assets/models/yolox_mouse_640.onnx"
            )
            # resources.as_file gives a context-manager; but we only need the path
            # check.  Traversable.is_file() suffices.
            if hasattr(ref, "is_file") and ref.is_file():  # type: ignore[union-attr]
                return Path(str(ref)), _load_meta_sidecar(Path(str(ref)))
        except (TypeError, FileNotFoundError, ModuleNotFoundError):
            pass

        return None
