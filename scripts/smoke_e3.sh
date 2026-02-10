#!/usr/bin/env bash
# E2E smoke test for E3: schemas, validation, and schema generation.
# Run from repo root. Requires uv and Python 3.11+.
set -eu

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

echo "==> Generate schemas"
uv run python scripts/generate_schemas.py

echo "==> Check schema drift"
uv run python scripts/check_schemas_up_to_date.py

echo "==> Quick import test"
uv run python -c "from scindra_engine.schemas import AnalysisConfig; AnalysisConfig(assay={'selection_mode':'AUTO'}, video={'path':'x.mp4'}, outputs={'out_dir':'out'})"

echo "Smoke E3: OK"
