import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from typer.testing import CliRunner

from scindra_engine.__main__ import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "scindra-engine" in result.stdout
    assert "0.1.0" in result.stdout


def test_engine_info_json() -> None:
    """Test engine-info --json outputs valid JSON with required keys."""
    result = runner.invoke(app, ["engine-info", "--json"])
    assert result.exit_code == 0

    data = json.loads(result.stdout)
    assert "engine_version" in data
    assert "git_commit" in data
    assert "python_version" in data
    assert "platform" in data
    assert "opencv_version" in data
    assert isinstance(data["engine_version"], str)
    assert data["git_commit"] is None or isinstance(data["git_commit"], str)


def test_validate_config_valid() -> None:
    """Test validate-config with a valid minimal config."""
    config_data = {
        "assay": {"selection_mode": "AUTO"},
        "video": {"path": "test.mp4"},
        "outputs": {"out_dir": "out"},
    }

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = Path(f.name)

    try:
        result = runner.invoke(app, ["validate-config", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()
    finally:
        config_path.unlink()


def test_validate_config_invalid() -> None:
    """Test validate-config with an invalid config."""
    config_data = {"invalid": "config"}

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = Path(f.name)

    try:
        result = runner.invoke(app, ["validate-config", "--config", str(config_path)])
        assert result.exit_code == 1
        # Error messages go to stderr
        assert "invalid" in (result.stdout + result.stderr).lower()
    finally:
        config_path.unlink()
