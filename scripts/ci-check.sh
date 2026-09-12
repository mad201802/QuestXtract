#!/usr/bin/env bash
# Runs same checks as CI. Auto-fixes where possible (ruff).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run ruff format .
uv run mypy
uv run pytest
