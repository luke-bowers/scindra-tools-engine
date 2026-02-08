def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("scindra-engine")
    except Exception:
        pass
    # Fallback for editable installs or when metadata is missing
    import tomllib
    from pathlib import Path
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    return "0.0.0+unknown"


__version__ = _get_version()
