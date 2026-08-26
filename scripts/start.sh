#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [ ! -d .venv ]; then
  echo "Virtual environment not found. Please run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
