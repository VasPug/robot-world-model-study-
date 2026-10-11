#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
uv run --locked python navigation.py "$@"
