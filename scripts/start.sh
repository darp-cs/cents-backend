#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Virtual environment not found. Please run ./scripts/setup.sh first."
  exit 1
fi

"$VENV_PY" -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000
