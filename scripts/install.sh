#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT"

# -------- 0) Pick a python executable (linux/mac: python3|python, windows git-bash: py -3 ) --------
pick_python() {
  if command -v python3 >/dev/null 2>&1; then echo python3; return 0; fi
  if command -v python  >/dev/null 2>&1; then echo python;  return 0; fi
  # Windows Git Bash often has the launcher `py`
  if command -v py >/dev/null 2>&1; then echo "py -3"; return 0; fi
  echo "❌ Could not find python3, python, or py on PATH." >&2
  echo "   Install Python 3.x and ensure it's on PATH, then re-run." >&2
  exit 1
}
PYEXE="$(pick_python)"
echo "[python] using: $PYEXE"

# -------- 1) Create venv --------
echo "[1/4] Creating venv at .venv"
# shellcheck disable=SC2086
$PYEXE -m venv .venv

# Activate (linux/mac -> bin/activate, windows git-bash -> Scripts/activate)
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  echo "❌ Could not find venv activation script (.venv/bin/activate or .venv/Scripts/activate)" >&2
  exit 1
fi

python -m pip install --upgrade pip

# -------- 2) Install PyTorch (GPU if available, else CPU) --------
echo "[2/4] Installing PyTorch (GPU if available, else CPU)…"
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio \
  || pip install torch torchvision torchaudio
else
  pip install torch torchvision torchaudio
fi

# -------- 3) Other deps (and ensure HF CLI exists) --------
echo "[3/4] Installing other dependencies"
pip install -r requirements.txt
# some windows envs don’t get the cli entrypoint without this
python -c "import huggingface_hub" 2>/dev/null || pip install -U huggingface_hub

# -------- 4) Download model from HF --------
echo "[4/4] Downloading model from Hugging Face"
MODEL_DEST="$ROOT/Models/facebook__wav2vec2-base"
mkdir -p "$MODEL_DEST"

# Try CLI; if unavailable, fall back to Python API (works without git-lfs)
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download yansari/arabic-letters-wav2vec2-base --local-dir "$MODEL_DEST" --quiet || NEED_PY=1
else
  NEED_PY=1
fi

if [[ "${NEED_PY:-0}" == "1" ]]; then
  python - <<'PY'
from huggingface_hub import HfApi
import os
dest = os.path.join(os.getcwd(), "Models", "facebook__wav2vec2-base")
HfApi().download_repo(
    repo_id="yansari/arabic-letters-wav2vec2-base",
    repo_type="model",
    local_dir=dest,
)
print("Downloaded via Python API to:", dest)
PY
fi

echo
echo "✅ Install complete."
echo "Run the server with:"
echo "  # Linux/macOS/WSL:"
echo "  source .venv/bin/activate && python Code/serve_realtime_fastapi.py --host 0.0.0.0 --port 7860"
echo "  # Windows (Git Bash):"
echo "  source .venv/Scripts/activate && python Code/serve_realtime_fastapi.py --host 0.0.0.0 --port 7860"
