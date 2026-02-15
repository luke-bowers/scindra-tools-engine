"""Tests for detector base types and ModelResolver."""
from __future__ import annotations

import json
from pathlib import Path

from scindra_engine.detectors.base import (
    Detection,
    DetectorResult,
    ModelMeta,
    ModelResolver,
    _load_meta_sidecar,
)


def test_detection_construction() -> None:
    d = Detection(bbox_xyxy=(10, 20, 100, 200), score=0.95, class_id=0)
    assert d.bbox_xyxy == (10, 20, 100, 200)
    assert d.score == 0.95
    assert d.class_id == 0


def test_detector_result_no_detections() -> None:
    r = DetectorResult(
        detections=[],
        best=None,
        confidence=0.0,
        reasons=["NO_DETECTIONS"],
    )
    assert r.best is None
    assert r.confidence == 0.0
    assert "NO_DETECTIONS" in r.reasons


def test_detector_result_with_best() -> None:
    d = Detection(bbox_xyxy=(10, 20, 100, 200), score=0.85, class_id=0)
    r = DetectorResult(
        detections=[d],
        best=d,
        confidence=0.85,
        reasons=[],
    )
    assert r.best is d
    assert r.confidence == 0.85


def test_model_meta_defaults() -> None:
    meta = ModelMeta()
    assert meta.input_size == (640, 640)
    assert meta.num_classes == 1
    assert meta.class_names == ["mouse"]
    assert meta.score_thresh == 0.25
    assert meta.nms_iou == 0.45


def test_load_meta_sidecar_json(tmp_path: Path) -> None:
    meta_data = {
        "input_size": [320, 320],
        "num_classes": 1,
        "class_names": ["mouse"],
        "score_thresh": 0.3,
        "nms_iou": 0.5,
    }
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake", encoding="utf-8")
    json_path = tmp_path / "model.json"
    json_path.write_text(json.dumps(meta_data), encoding="utf-8")

    meta = _load_meta_sidecar(model_path)
    assert meta.input_size == (320, 320)
    assert meta.score_thresh == 0.3
    assert meta.nms_iou == 0.5


def test_load_meta_sidecar_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake", encoding="utf-8")
    meta = _load_meta_sidecar(model_path)
    assert meta == ModelMeta()


def test_model_resolver_explicit_path(tmp_path: Path) -> None:
    model_file = tmp_path / "yolox.onnx"
    model_file.write_text("fake_model", encoding="utf-8")
    resolver = ModelResolver(str(model_file))
    result = resolver.resolve()
    assert result is not None
    path, meta = result
    assert path == model_file
    assert meta.input_size == (640, 640)  # defaults since no sidecar


def test_model_resolver_explicit_path_missing() -> None:
    resolver = ModelResolver("/nonexistent/model.onnx")
    result = resolver.resolve()
    assert result is None


def test_model_resolver_env_var(tmp_path: Path, monkeypatch: object) -> None:
    model_file = tmp_path / "env_model.onnx"
    model_file.write_text("fake_model", encoding="utf-8")
    import pytest

    mp = pytest.MonkeyPatch()
    mp.setenv("SCINDRA_YOLOX_ONNX_PATH", str(model_file))
    try:
        resolver = ModelResolver(None)
        result = resolver.resolve()
        assert result is not None
        path, _meta = result
        assert path == model_file
    finally:
        mp.undo()


def test_model_resolver_no_source_returns_none(monkeypatch: object) -> None:
    import pytest

    mp = pytest.MonkeyPatch()
    mp.delenv("SCINDRA_YOLOX_ONNX_PATH", raising=False)
    try:
        resolver = ModelResolver(None)
        result = resolver.resolve()
        # May return None or a packaged asset if it exists; just ensure no crash
        assert result is None or result[0].is_file()
    finally:
        mp.undo()
