#!/usr/bin/env python3
"""Generate constraints-desktop.txt from a built wheel for Desktop Pro wheelhouse builds.

Usage:
    python scripts/generate_constraints_desktop.py --wheel dist/<wheel>.whl --out constraints-desktop.txt

Creates a temporary venv, installs the wheel (non-editable), runs pip freeze,
filters out pip/setuptools/wheel and the project itself (so pip download -c
constraints + local wheel does not conflict), and writes the result to --out.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


EXCLUDE_PACKAGES = frozenset({"pip", "setuptools", "wheel"})


def project_name_from_wheel(wheel_path: Path) -> str:
    """Derive PyPI project name from wheel filename (e.g. scindra_engine-0.1.0-... -> scindra-engine)."""
    stem = wheel_path.stem  # e.g. scindra_engine-0.1.0-py3-none-any
    match = re.match(r"^(.+?)-\d", stem)
    if match:
        dist = match.group(1)  # e.g. scindra_engine
        return dist.replace("_", "-").lower()
    return stem.split("-")[0].replace("_", "-").lower()


def get_venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate constraints-desktop.txt from a built wheel."
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help="Path to the built .whl file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for constraints file",
    )
    args = parser.parse_args()

    wheel_path = args.wheel.resolve()
    if not wheel_path.exists():
        print(f"Error: wheel not found: {wheel_path}", file=sys.stderr)
        return 1
    if wheel_path.suffix != ".whl":
        print(f"Error: not a wheel file: {wheel_path}", file=sys.stderr)
        return 1

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="constraints_desktop_")
    try:
        venv_dir = Path(tmpdir) / "venv"
        venv.create(venv_dir, with_pip=True)
        py = get_venv_python(venv_dir)
        if not py.exists():
            print(f"Error: venv python not found: {py}", file=sys.stderr)
            return 1

        subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", str(wheel_path)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [str(py), "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )

        exclude = EXCLUDE_PACKAGES | {project_name_from_wheel(wheel_path)}
        lines = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line:
                name = line.split("==")[0].strip().lower()
                if name in exclude:
                    continue
            lines.append(line)

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} constraints to {out_path}")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
