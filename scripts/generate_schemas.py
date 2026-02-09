#!/usr/bin/env python3
"""Generate JSON schemas from Pydantic models."""
import json
from pathlib import Path

from scindra_engine.schemas import AnalysisConfig, ArtifactManifest


def main() -> None:
    """Generate JSON schemas and write them to shared/schemas/."""
    root = Path(__file__).parent.parent
    schemas_dir = root / "shared" / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)

    # Generate AnalysisConfig schema
    analysis_config_schema = AnalysisConfig.model_json_schema(mode="serialization")
    analysis_config_path = schemas_dir / "analysis_config.schema.json"
    with open(analysis_config_path, "w", encoding="utf-8") as f:
        json.dump(
            analysis_config_schema,
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Generated: {analysis_config_path}")

    # Generate ArtifactManifest schema
    manifest_schema = ArtifactManifest.model_json_schema(mode="serialization")
    manifest_path = schemas_dir / "artifact_manifest.schema.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest_schema,
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Generated: {manifest_path}")


if __name__ == "__main__":
    main()
