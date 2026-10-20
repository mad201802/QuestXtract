#!/usr/bin/env bash
# Runs same checks as CI. Auto-fixes where possible (ruff).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run ruff format .
uv run mypy
# Offline, the QTI XSD tests are skipped here; CI always runs them.
if uv run python scripts/fetch_qti_xsd.py; then
    QUESTXTRACT_REQUIRE_QTI_XSD=1 uv run pytest
else
    echo "ci-check: QTI XSDs not available, the XSD validation tests are skipped (CI runs them)" >&2
    uv run pytest
fi
