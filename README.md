# Arabic Alphabet Realtime — wav2vec2-base

FastAPI demo for a fine-tuned **wav2vec2-base** model that classifies **Arabic letters** from short audio snippets (mic or file).  
No Gradio, no ffmpeg—just vanilla HTML/JS in the browser and a FastAPI backend. Uses GPU if available.

**Model weights** are hosted on Hugging Face: [`yansari/arabic-letters-wav2vec2-base`](https://huggingface.co/yansari/arabic-letters-wav2vec2-base).  
The installer downloads them automatically into `Models/facebook__wav2vec2-base/`.

---

## Features

- 🎙️ In-browser mic recording (WAV encoded client-side)
- ⚡ FastAPI backend with torch/transformers (CUDA if present)
- 📈 Live probabilities, waveform plot, top-K table
- 🧪 Optional training diagnostics (confusion matrix, per-class report)
- 🌐 One-flag public sharing via ngrok (`--share`)

---

## Quickstart

```bash
# 1) clone
git clone https://github.com/YaqoobAnsari/arabic-letters-realtime.git
cd arabic-letters-realtime

# 2) install (creates .venv, installs deps, downloads model from HF)
bash scripts/install.sh

# 3) run
bash scripts/run.sh
# open http://localhost:7860
