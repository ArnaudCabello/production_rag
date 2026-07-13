#!/usr/bin/env bash
# One-command launcher: builds the frontend if needed, then serves the app and
# the API from a single process at http://127.0.0.1:8642
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

if [ ! -d frontend/dist ]; then
  echo "Building frontend (first run only)..."
  (cd frontend && npm install --no-audit --no-fund && npm run build)
fi

( sleep 2
  xdg-open http://127.0.0.1:8642 2>/dev/null || open http://127.0.0.1:8642 2>/dev/null || true
) &

exec python -m uvicorn backend.main:app --port 8642
