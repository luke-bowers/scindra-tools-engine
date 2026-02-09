#!/usr/bin/env python3
"""Check that committed JSON schemas match regenerated schemas."""
import json
import sys
import tempfile
from pathlib import Path

from scindra_engine.schemas import AnalysisConfig, ArtifactManifest


def main() -> int:
    """Regenerate schemas and compare with committed versions."""
    root = Path(__file__).parent.parent
    schemas_dir = root / "shared" / "schemas"

    # Regenerate schemas to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Generate AnalysisConfig schema
        analysis_config_schema = AnalysisConfig.model_json_schema(mode="serialization")
        tmp_analysis_config = tmp_path / "analysis_config.schema.json"
        with open(tmp_analysis_config, "w", encoding="utf-8") as f:
            json.dump(
                analysis_config_schema,
                f,
                indent=2,
                sort_keys=True,
            )

        # Generate ArtifactManifest schema
        manifest_schema = ArtifactManifest.model_json_schema(mode="serialization")
        tmp_manifest = tmp_path / "artifact_manifest.schema.json"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(
                manifest_schema,
                f,
                indent=2,
                sort_keys=True,
            )

        # Compare with committed schemas
        committed_analysis_config = schemas_dir / "analysis_config.schema.json"
        committed_manifest = schemas_dir / "artifact_manifest.schema.json"

        errors = []

        if not committed_analysis_config.exists():
            errors.append(
                f"Missing committed schema: {committed_analysis_config}"
            )
        else:
            with open(committed_analysis_config, "rb") as f:
                committed_content = f.read()
            with open(tmp_analysis_config, "rb") as f:
                regenerated_content = f.read()
            if committed_content != regenerated_content:
                errors.append(
                    f"Schema drift detected in {committed_analysis_config.name}"
                )
                print(
                    f"Error: {committed_analysis_config.name} differs from regenerated version",
                    file=sys.stderr,
                )

        if not committed_manifest.exists():
            errors.append(f"Missing committed schema: {committed_manifest}")
        else:
            with open(committed_manifest, "rb") as f:
                committed_content = f.read()
            with open(tmp_manifest, "rb") as f:
                regenerated_content = f.read()
            if committed_content != regenerated_content:
                errors.append(
                    f"Schema drift detected in {committed_manifest.name}"
                )
                print(
                    f"Error: {committed_manifest.name} differs from regenerated version",
                    file=sys.stderr,
                )

        if errors:
            print(
                "\nSchema drift detected! Run 'python scripts/generate_schemas.py' to update.",
                file=sys.stderr,
            )
            return 1

    print("Schema check passed: committed schemas match regenerated schemas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
