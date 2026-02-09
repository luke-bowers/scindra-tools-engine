from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from scindra_engine.schemas import (  # type: ignore[import-untyped]
    AnalysisConfig,
    ArtifactManifest,
    InputFile,
    OutputFile,
)


def load_json_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema from shared/schemas/."""
    root = Path(__file__).parent.parent
    schema_path = root / "shared" / "schemas" / f"{name}.schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_minimal_valid_config(self) -> None:
        """Test creating a minimal valid config."""
        config = AnalysisConfig(
            assay={"selection_mode": "AUTO"},
            video={"path": "test.mp4"},
            outputs={"out_dir": "out"},
        )
        assert config.assay.selection_mode == "AUTO"
        assert config.video.path == "test.mp4"
        assert config.outputs.out_dir == "out"

    def test_pydantic_roundtrip(self) -> None:
        """Test serialization and deserialization roundtrip."""
        config_dict = {
            "assay": {"selection_mode": "AUTO"},
            "video": {"path": "test.mp4", "fps_override": 30.0},
            "outputs": {"out_dir": "out", "debug_frame_count": 5},
        }
        config = AnalysisConfig(**config_dict)
        serialized = config.model_dump(mode="json")
        deserialized = AnalysisConfig(**serialized)
        assert deserialized.model_dump() == config.model_dump()

    def test_manual_assay_type_required(self) -> None:
        """Test that assay_type is required when selection_mode is MANUAL."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "MANUAL"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out"},
            )
        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("assay", "assay_type")
            and "required" in e["msg"].lower()
            for e in errors
        ) or any(
            "assay_type is required when selection_mode is 'MANUAL'" in str(e)
            for e in errors
        )

    def test_manual_assay_type_provided(self) -> None:
        """Test that MANUAL mode works when assay_type is provided."""
        config = AnalysisConfig(
            assay={"selection_mode": "MANUAL", "assay_type": "OPEN_FIELD"},
            video={"path": "test.mp4"},
            outputs={"out_dir": "out"},
        )
        assert config.assay.selection_mode == "MANUAL"
        assert config.assay.assay_type == "OPEN_FIELD"

    def test_manual_value_required_for_manual_threshold(self) -> None:
        """Test that manual_value is required when threshold is manual."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out"},
                segmentation={"threshold": "manual"},
            )
        errors = exc_info.value.errors()
        assert any(
            "manual_value is required when threshold is 'manual'" in str(e)
            for e in errors
        )

    def test_manual_value_range(self) -> None:
        """Test that manual_value must be 0-255."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out"},
                segmentation={"threshold": "manual", "manual_value": 300},
            )
        errors = exc_info.value.errors()
        assert any(
            "manual_value must be between 0 and 255" in str(e) for e in errors
        )

    def test_adaptive_block_size_odd(self) -> None:
        """Test that adaptive_block_size must be odd."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out"},
                segmentation={"adaptive_block_size": 34},
            )
        errors = exc_info.value.errors()
        assert any(
            "adaptive_block_size must be odd" in str(e) for e in errors
        )

    def test_adaptive_block_size_minimum(self) -> None:
        """Test that adaptive_block_size must be >= 3."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out"},
                segmentation={"adaptive_block_size": 1},
            )
        errors = exc_info.value.errors()
        assert any(
            "adaptive_block_size must be >= 3" in str(e) for e in errors
        )

    def test_confidence_thresholds_range(self) -> None:
        """Test that confidence thresholds must be 0..1."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO", "min_confidence_to_auto_accept": 1.5},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out"},
            )
        errors = exc_info.value.errors()
        assert any(
            "less than or equal to 1" in str(e).lower() for e in errors
        )

    def test_debug_frame_count_range(self) -> None:
        """Test that debug_frame_count must be 1..50."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out", "debug_frame_count": 0},
            )
        errors = exc_info.value.errors()
        assert any("greater than or equal to 1" in str(e).lower() for e in errors)

        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4"},
                outputs={"out_dir": "out", "debug_frame_count": 51},
            )
        errors = exc_info.value.errors()
        assert any("less than or equal to 50" in str(e).lower() for e in errors)

    def test_time_range_validation(self) -> None:
        """Test that start_time_s must be less than end_time_s."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisConfig(
                assay={"selection_mode": "AUTO"},
                video={"path": "test.mp4", "start_time_s": 10.0, "end_time_s": 5.0},
                outputs={"out_dir": "out"},
            )
        errors = exc_info.value.errors()
        assert any(
            "start_time_s must be less than end_time_s" in str(e) for e in errors
        )

    def test_json_schema_validation(self) -> None:
        """Test that a valid config validates against the JSON schema."""
        schema = load_json_schema("analysis_config")
        config_dict = {
            "assay": {"selection_mode": "AUTO"},
            "video": {"path": "test.mp4"},
            "outputs": {"out_dir": "out"},
        }
        jsonschema.validate(instance=config_dict, schema=schema)

    def test_json_schema_validation_invalid(self) -> None:
        """Test that an invalid config fails JSON schema validation."""
        schema = load_json_schema("analysis_config")
        config_dict = {
            "assay": {"selection_mode": "INVALID"},
            "video": {"path": "test.mp4"},
            "outputs": {"out_dir": "out"},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config_dict, schema=schema)


class TestArtifactManifest:
    """Tests for ArtifactManifest model."""

    def test_minimal_valid_manifest(self) -> None:
        """Test creating a minimal valid manifest."""
        manifest = ArtifactManifest(
            engine_version="0.1.0",
            run_id="test-run-123",
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:01:00Z",
            input_files=[
                InputFile(path="input.mp4", sha256="abc123", bytes=1000)
            ],
            config={"path": "config.yaml", "sha256": "def456"},
            outputs=[
                OutputFile(
                    kind="video", path="out.mp4", sha256="ghi789", bytes=2000
                )
            ],
            qc_metrics={"track_coverage": 0.95},
            summary_metrics={"total_distance": 100.5},
            warnings=[],
            needs_review=False,
            review_reasons=[],
            confidence={"arena": 0.92, "tracking": 0.88},
        )
        assert manifest.engine_version == "0.1.0"
        assert manifest.run_id == "test-run-123"
        assert manifest.needs_review is False

    def test_pydantic_roundtrip(self) -> None:
        """Test serialization and deserialization roundtrip."""
        manifest_dict = {
            "engine_version": "0.1.0",
            "run_id": "test-run-123",
            "started_at": "2024-01-01T00:00:00Z",
            "finished_at": "2024-01-01T00:01:00Z",
            "input_files": [
                {"path": "input.mp4", "sha256": "abc123", "bytes": 1000}
            ],
            "config": {"path": "config.yaml", "sha256": "def456"},
            "outputs": [
                {
                    "kind": "video",
                    "path": "out.mp4",
                    "sha256": "ghi789",
                    "bytes": 2000,
                }
            ],
            "qc_metrics": {"track_coverage": 0.95},
            "summary_metrics": {"total_distance": 100.5},
            "warnings": [],
            "needs_review": False,
            "review_reasons": [],
            "confidence": {"arena": 0.92, "tracking": 0.88},
        }
        manifest = ArtifactManifest(**manifest_dict)
        serialized = manifest.model_dump(mode="json")
        deserialized = ArtifactManifest(**serialized)
        assert deserialized.model_dump() == manifest.model_dump()

    def test_json_schema_validation(self) -> None:
        """Test that a valid manifest validates against the JSON schema."""
        schema = load_json_schema("artifact_manifest")
        manifest_dict = {
            "engine_version": "0.1.0",
            "run_id": "test-run-123",
            "started_at": "2024-01-01T00:00:00Z",
            "finished_at": "2024-01-01T00:01:00Z",
            "input_files": [
                {"path": "input.mp4", "sha256": "abc123", "bytes": 1000}
            ],
            "config": {"path": "config.yaml", "sha256": "def456"},
            "outputs": [
                {
                    "kind": "video",
                    "path": "out.mp4",
                    "sha256": "ghi789",
                    "bytes": 2000,
                }
            ],
            "qc_metrics": {"track_coverage": 0.95},
            "summary_metrics": {"total_distance": 100.5},
            "warnings": [],
            "needs_review": False,
            "review_reasons": [],
            "confidence": {"arena": 0.92, "tracking": 0.88},
        }
        jsonschema.validate(instance=manifest_dict, schema=schema)
