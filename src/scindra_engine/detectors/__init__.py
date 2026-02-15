"""Optional detector subsystem for scindra-engine.

All public types are importable from this package without requiring
``onnxruntime``.  The concrete ``YOLOXOnnxDetector`` lazily imports
``onnxruntime`` only when instantiated.
"""
from __future__ import annotations

from scindra_engine.detectors.base import (
    Detection,
    Detector,
    DetectorResult,
    ModelMeta,
    ModelResolver,
)

__all__ = [
    "Detection",
    "Detector",
    "DetectorResult",
    "ModelMeta",
    "ModelResolver",
]
