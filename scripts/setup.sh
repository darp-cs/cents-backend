#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required but was not found on PATH."
  echo "Install it with: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install -y python3.12 python3.12-venv python3.12-dev"
  exit 1
fi

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  rm -rf .venv
  python3.12 -m venv .venv
  if [ ! -x "$VENV_PY" ]; then
    echo "Failed to create virtual environment. Ensure python3.12-venv is installed:"
    echo "  sudo apt install -y python3.12-venv"
    exit 1
  fi
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt
"$VENV_PY" -c "import asyncio; from app.db.base import init_db; from app.vector_store import ensure_vector_store_ready; asyncio.run(init_db()); ensure_vector_store_ready()"

echo ""
echo "Setup complete."
echo "To start the app, run:"
echo "  ./scripts/start.sh"
