from typer.testing import CliRunner

from scindra_engine.__main__ import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "scindra-engine" in result.stdout
    assert "0.1.0" in result.stdout
