#!/usr/bin/env bash
# Start the Pocket Coach dashboard locally.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Creating virtualenv…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

# The cold path pulls candles once and freezes them into data/snapshot/ so the
# dashboard still has real data when the network is unavailable.
if [ "${1:-}" = "--snapshot" ]; then
  echo "Refreshing candle snapshot…"
  $PY -m scripts.build_snapshot
  exit 0
fi

PORT="${PORT:-8848}"
echo "Pocket Coach -> http://127.0.0.1:${PORT}"
exec $PY -m uvicorn po_coach.server:app --host 127.0.0.1 --port "$PORT" "$@"
