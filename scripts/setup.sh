#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.12 is required but was not found on PATH."
  echo "Install Python 3.12 and rerun this script."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH."
  echo "Install Docker Desktop or Docker Engine and rerun this script."
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

docker compose up -d
python -m alembic upgrade head

echo ""
echo "Setup complete."
echo "To start the app, run:"
echo "  ./scripts/start.sh"
