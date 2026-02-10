#!/usr/bin/env bash
# E2E smoke test for E4: assay registry.
# Run from repo root. Requires uv and Python 3.11+.
set -eu

SCRIPT_DIR="$(dirname "$0")"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

"$SCRIPT_DIR/smoke_e3.sh"

echo "==> Assay registry checks"
uv run python -c "from scindra_engine.assays.registry import list_assays; a=list_assays(); assert 'OPEN_FIELD' in a and len(a)==12"
uv run python -c "from scindra_engine.assays.registry import get_assay; d=get_assay('EPM'); assert d.zone_template_id=='EPM_ARMS'"

echo "Smoke E4: OK"

