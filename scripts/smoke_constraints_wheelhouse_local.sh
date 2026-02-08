#!/usr/bin/env bash
# E2E smoke: constraints-desktop.txt + offline wheelhouse. Run from repo root.
# Requires Python 3.11+ and pip (no uv). Cleans dist/ and .smoke_wheelhouse, then:
# pip dev install, ruff/mypy/pytest, build, twine, generate constraints,
# pip download wheelhouse (wheels only), offline venv install, scindra-engine --version.
set -eu

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SMOKE_DIR="$ROOT/.smoke_wheelhouse"
WHEELHOUSE="$SMOKE_DIR/wheelhouse"
VENV2="$SMOKE_DIR/venv2"
CONSTRAINTS="$SMOKE_DIR/constraints-desktop.txt"

echo "==> Clean dist/ and smoke temp dir"
rm -rf "$SMOKE_DIR"
if [ -d dist ]; then
  rm -rf dist/*
fi

echo "==> Dev install"
python -m pip install -q -e ".[dev]"

echo "==> Lint"
ruff check .

echo "==> Type check"
mypy src

echo "==> Test"
PYTHONPATH=src pytest

echo "==> Build"
python -m build

echo "==> Twine check"
python -m twine check dist/*

echo "==> Resolve wheel and generate constraints"
WHEEL="$(echo dist/scindra_engine-*.whl)"
if [ ! -f "$WHEEL" ]; then
  echo "Error: no wheel found in dist/" >&2
  exit 1
fi
mkdir -p "$SMOKE_DIR"
python scripts/generate_constraints_desktop.py --wheel "$WHEEL" --out "$CONSTRAINTS"

echo "==> Build wheelhouse (wheels only)"
mkdir -p "$WHEELHOUSE"
python -m pip download --only-binary=:all: -d "$WHEELHOUSE" -c "$CONSTRAINTS" "$WHEEL" || { echo "Error: pip download failed" >&2; exit 1; }
sdist_count=$(find "$WHEELHOUSE" -maxdepth 1 \( -name '*.tar.gz' -o -name '*.zip' \) 2>/dev/null | wc -l)
if [ "$sdist_count" -gt 0 ]; then
  echo "Error: sdist found in wheelhouse (expected wheels only)" >&2
  exit 1
fi

echo "==> Create fresh venv and install offline"
python -m venv "$VENV2"
# shellcheck source=/dev/null
"$VENV2/bin/pip" install --no-index --find-links "$WHEELHOUSE" "$WHEEL" || { echo "Error: pip install --no-index failed (wheelhouse may be incomplete)" >&2; exit 1; }

echo "==> Run scindra-engine --version"
if ! "$VENV2/bin/scindra-engine" --version; then
  echo "Error: scindra-engine --version failed" >&2
  exit 1
fi

echo "Smoke constraints wheelhouse local: OK"
