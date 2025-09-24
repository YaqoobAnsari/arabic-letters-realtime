#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
source "$ROOT/.venv/bin/activate"
python "$ROOT/Code/serve_realtime_fastapi.py" --host 0.0.0.0 --port 7860 "$@"
