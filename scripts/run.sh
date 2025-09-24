#!/usr/bin/env bash
set -Eeuo pipefail

# 1) Find repo root (works even if you run from anywhere)
if ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  # fallback: scripts/.. relative
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

VENV="$ROOT/.venv"
APP="$ROOT/Code/serve_realtime_fastapi.py"

# 2) Check venv & app exist
if [[ ! -f "$APP" ]]; then
  echo "❌ Cannot find app at: $APP"
  exit 1
fi

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "❌ No virtualenv found at: $VENV"
  echo "   Run: bash \"$ROOT/scripts/install.sh\""
  exit 1
fi

# 3) Activate venv (must use bash, not sh)
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# 4) Run the server; pass through any extra args you supply
exec python "$APP" --host 0.0.0.0 --port 7860 "$@"
