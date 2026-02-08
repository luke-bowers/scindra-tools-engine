#!/usr/bin/env bash
# Smoke test for release: lint, typecheck, test, build, twine check, install wheel, run --version.
# Run from repo root. Requires uv and Python 3.11.
set -eu

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "==> Lint"
uv run ruff check .

echo "==> Type check"
uv run mypy src

echo "==> Test"
uv run pytest

echo "==> Build"
uv run python -m build

echo "==> Twine check"
uv run twine check dist/*

echo "==> Create temp venv and install wheel"
SMOKE_VENV="$ROOT/.smoke_venv"
rm -rf "$SMOKE_VENV"
uv venv "$SMOKE_VENV" --python 3.11
# shellcheck source=/dev/null
source "$SMOKE_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "$ROOT"/dist/scindra_engine-*.whl

echo "==> Run scindra-engine --version"
if ! scindra-engine --version; then
  echo "Error: scindra-engine --version failed" >&2
  exit 1
fi

echo "Smoke release local: OK"
