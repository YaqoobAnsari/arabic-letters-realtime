# Arabic Alphabet Realtime — wav2vec2-base

FastAPI demo for a fine-tuned wav2vec2-base Arabic letters classifier.  
- Pure FastAPI + vanilla HTML/JS (in-browser mic recording)
- No ffmpeg required
- GPU if available
- Optional public link via ngrok

Model weights are hosted on Hugging Face: **yansari/arabic-letters-wav2vec2-base**.  
The installer downloads them automatically into `Models/facebook__wav2vec2-base/`.

---

## Quickstart

\`\`\`bash
# 1) clone
git clone https://github.com/YaqoobAnsari/arabic-letters-realtime.git
cd arabic-letters-realtime

# 2) install (creates .venv, installs deps, downloads model)
bash scripts/install.sh

# 3) run (activates .venv and serves on port 7860)
bash scripts/run.sh
# then open http://localhost:7860
\`\`\`

