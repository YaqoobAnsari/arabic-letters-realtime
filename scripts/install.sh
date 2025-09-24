#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT"

echo "[1/4] Creating venv at .venv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[2/4] Installing PyTorch (GPU if available, else CPU)…"
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio \
  || pip install torch torchvision torchaudio
else
  pip install torch torchvision torchaudio
fi

echo "[3/4] Installing other dependencies"
pip install -r requirements.txt

echo "[4/4] Downloading model from Hugging Face"
MODEL_DEST="$ROOT/Models/facebook__wav2vec2-base"
mkdir -p "$MODEL_DEST"
# Uses public HF repo (no login required for downloading public models)
huggingface-cli download yansari/arabic-letters-wav2vec2-base --local-dir "$MODEL_DEST" --quiet

echo
echo "✅ Install complete."
echo "To run:"
echo "  source .venv/bin/activate"
echo "  python Code/serve_realtime_fastapi.py --host 0.0.0.0 --port 7860"
