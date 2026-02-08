#!/usr/bin/env python3
"""Check that the git tag version matches the package version in pyproject.toml.

Usage:
    python scripts/check_release_version.py v0.1.0

Exits 0 if the tag (after stripping a leading 'v') matches project.version in pyproject.toml.
Exits 1 and prints an error otherwise.
"""
import sys
import tomllib
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: check_release_version.py <tag>", file=sys.stderr)
        print("Example: check_release_version.py v0.1.0", file=sys.stderr)
        return 1

    tag = sys.argv[1].strip()
    if not tag:
        print("Error: tag must be non-empty", file=sys.stderr)
        return 1

    # Normalize tag: strip leading 'v' if present
    normalized_tag = tag[1:] if tag.startswith("v") else tag

    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        return 1

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    try:
        package_version = data["project"]["version"]
    except KeyError:
        print("Error: [project].version not found in pyproject.toml", file=sys.stderr)
        return 1

    if normalized_tag != package_version:
        print(
            f"Error: tag version does not match pyproject.toml\n"
            f"  tag (normalized): {normalized_tag!r}\n"
            f"  project.version:  {package_version!r}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
