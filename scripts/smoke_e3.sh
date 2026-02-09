#!/usr/bin/env bash
# E2E smoke test for E3: schemas, validation, and schema generation.
# Run from repo root. Requires Python 3.11+.
set -eu

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "==> Install dev dependencies"
python -m pip install -e ".[dev]"

echo "==> Lint"
python -m ruff check .

echo "==> Type check"
python -m mypy src

echo "==> Test"
python -m pytest

echo "==> Generate schemas"
python scripts/generate_schemas.py

echo "==> Check schema drift"
python scripts/check_schemas_up_to_date.py

echo "==> Quick import test"
python -c "from scindra_engine.schemas import AnalysisConfig; AnalysisConfig(assay={'selection_mode':'AUTO'}, video={'path':'x.mp4'}, outputs={'out_dir':'out'})"

echo "Smoke E3: OK"
