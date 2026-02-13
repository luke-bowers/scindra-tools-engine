#!/usr/bin/env bash
# Adaptive smoke test for scindra-engine CLI.
# Run from repo root. Requires uv and Python 3.11+.
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "==> Install dev dependencies (uv sync)"
uv sync --extra dev

echo "==> Lint"
uv run ruff check .

echo "==> Type check"
uv run mypy src

echo "==> Test"
uv run pytest

echo "==> Run smoke_latest"
uv run python scripts/smoke_latest.py
