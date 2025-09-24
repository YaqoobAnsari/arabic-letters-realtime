#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_realtime_fastapi.py
Headless realtime demo for your fine-tuned wav2vec2-base Arabic letters model.

- Pure FastAPI + vanilla HTML/JS (records mic, encodes WAV in-browser)
- No Gradio. No server-side ffmpeg.
- GPU if available. Serves a premium dashboard with metrics.
- Optional --share uses ngrok if NGROK_AUTHTOKEN is set.

Run:
  python serve_realtime_fastapi.py --host 0.0.0.0 --port 7860
  # or with a public link (requires: pip install pyngrok && export NGROK_AUTHTOKEN=...):
  python serve_realtime_fastapi.py --host 0.0.0.0 --port 7860 --share
"""

from __future__ import annotations
import os, io, time, json, argparse, base64, threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
import uvicorn

try:
    import resampy
    HAVE_RESAMPY = True
except Exception:
    HAVE_RESAMPY = False

from transformers import AutoProcessor, AutoModelForAudioClassification


# --------------------------- Repo paths & model discovery ---------------------------

def resolve_repo_paths(script_file: Path) -> Dict[str, Path]:
    code_dir = script_file.resolve().parent
    root = code_dir.parent
    return {
        "root": root,
        "code": code_dir,
        "dataset": root / "Dataset",
        "models": root / "Models",
        "results": root / "Results",
    }

def preferred_model_dir(paths: Dict[str, Path]) -> Path:
    res = paths["results"] / "facebook__wav2vec2-base" / "model"
    mod = paths["models"]  / "facebook__wav2vec2-base"
    if (res / "config.json").exists():
        return res
    if (mod / "config.json").exists():
        return mod
    raise FileNotFoundError(
        f"Could not find the fine-tuned wav2vec2-base model.\n"
        f"Looked in:\n  {res}\n  {mod}\n"
        "Make sure finetune_eval.py finished for facebook/wav2vec2-base."
    )

def optional_confusion_paths(paths: Dict[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    png = paths["results"] / "facebook__wav2vec2-base" / "confusion_matrix.png"
    csv = paths["results"] / "facebook__wav2vec2-base" / "per_class_report.csv"
    return (png if png.exists() else None, csv if csv.exists() else None)


# --------------------------- Audio helpers ---------------------------

def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)

def resample_if_needed(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    if HAVE_RESAMPY:
        return resampy.resample(y, sr_in, sr_out).astype(np.float32, copy=False)
    # Linear resample fallback
    duration = len(y) / float(sr_in)
    new_len = max(1, int(round(duration * sr_out)))
    x_old = np.linspace(0, len(y), num=len(y), endpoint=False)
    x_new = np.linspace(0, len(y), num=new_len, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32, copy=False)

def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")


# --------------------------- Plot helpers (return base64 PNG) ---------------------------

def fig_to_base64_png(fig, dpi: int = 160) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{b}"

def plot_waveform_b64(y: np.ndarray, sr: int) -> str:
    t = np.arange(len(y)) / float(sr)
    fig = plt.figure(figsize=(8, 1.8))
    ax = fig.add_subplot(111)
    ax.plot(t, y, linewidth=1.0)
    ax.set_xlabel("Seconds"); ax.set_ylabel("Amplitude"); ax.set_title("Recorded waveform")
    plt.tight_layout()
    return fig_to_base64_png(fig)

def plot_prob_bars_b64(labels: List[str], probs: np.ndarray, top: int = 10) -> str:
    idxs = np.argsort(-probs)[:top]
    top_labels = [labels[i] for i in idxs]
    top_probs = probs[idxs]
    fig = plt.figure(figsize=(8, 3.2))
    ax = fig.add_subplot(111)
    ax.bar(top_labels, top_probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(f"Top-{top} class probabilities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig_to_base64_png(fig)


# --------------------------- Model wrapper ---------------------------

class Wav2Vec2Classifier:
    def __init__(self, model_dir: Path, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForAudioClassification.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()

        cfg = self.model.config
        if getattr(cfg, "id2label", None):
            self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        else:
            summary = model_dir.parent / "summary.json"
            if summary.exists():
                with open(summary, "r", encoding="utf-8") as f:
                    labels = json.load(f).get("labels", [])
                self.id2label = {i: lab for i, lab in enumerate(labels)}
            else:
                self.id2label = {i: f"CLASS_{i}" for i in range(cfg.num_labels)}

        sr = getattr(self.processor, "sampling_rate", None)
        if sr is None and hasattr(self.processor, "feature_extractor"):
            sr = getattr(self.processor.feature_extractor, "sampling_rate", None)
        self.sampling_rate = int(sr) if sr else 16000

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.size_mb = round(sum(p.element_size() * p.nelement() for p in self.model.parameters()) / (1024**2), 2)

    @torch.no_grad()
    def infer_numpy(
        self,
        y: np.ndarray,
        sr_in: int,
        top_k: int = 5,
        fixed_seconds: float = 1.0
    ) -> Dict:
        y = to_mono(y)
        y = resample_if_needed(y, sr_in, self.sampling_rate)
        max_len = int(round(fixed_seconds * self.sampling_rate))
        y = pad_or_trim(y, max_len)

        t0 = time.perf_counter()
        inputs = self.processor(
            [y],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        t1 = time.perf_counter()

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        top_k = max(1, min(top_k, probs.shape[-1]))
        idxs = np.argsort(-probs)[:top_k]
        labels = [self.id2label[int(i)] for i in idxs]
        scores = [float(probs[i]) for i in idxs]

        return {
            "top1_label": labels[0],
            "top1_prob": scores[0],
            "topk": list(zip(labels, scores)),
            "probs": probs,
            "latency_ms": (t1 - t0) * 1000.0,
            "sr_used": self.sampling_rate,
            "processed_audio": y,  # return for plotting
        }


# --------------------------- App state ---------------------------

app = FastAPI(title="Arabic Alphabet Realtime (wav2vec2-base)", version="1.0.0")
_MODEL: Optional[Wav2Vec2Classifier] = None
_CONF_PNG: Optional[Path] = None
_CONF_CSV: Optional[Path] = None
_PATHS: Optional[Dict[str, Path]] = None

def _load_once():
    global _MODEL, _CONF_PNG, _CONF_CSV, _PATHS
    if _MODEL is None:
        _PATHS = resolve_repo_paths(Path(__file__))
        mdir = preferred_model_dir(_PATHS)
        _MODEL = Wav2Vec2Classifier(mdir, device="auto")
        _CONF_PNG, _CONF_CSV = optional_confusion_paths(_PATHS)
    return _MODEL


# --------------------------- Routes: static training artifacts ---------------------------

@app.get("/confusion", response_class=Response)
def get_confusion():
    _load_once()
    if _CONF_PNG is None:
        return Response(status_code=204)
    return FileResponse(str(_CONF_PNG))

@app.get("/per_class_report", response_class=Response)
def get_per_class_report():
    _load_once()
    if _CONF_CSV is None:
        return Response(status_code=204)
    return FileResponse(str(_CONF_CSV), media_type="text/csv", filename="per_class_report.csv")


# --------------------------- Main page (HTML + JS) ---------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Arabic Alphabet Realtime — wav2vec2-base</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0b0f19; --panel:#121828; --border:#1d2742; --muted:#9aa4b2; --text:#e6e9ef;
    --primary:#3b82f6; --primary-600:#2563eb; --danger:#ef4444; --accent:#22c55e; --card:#0f1525;
  }
  body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, Cantarell;
         background: var(--bg); color: var(--text); }
  header { padding: 18px 22px; border-bottom: 1px solid var(--border); background: var(--panel); }
  header h2 { margin:0; font-weight:700; letter-spacing:.2px; }
  main { padding: 22px; max-width: 1200px; margin: 0 auto; }
  .row { display:flex; gap:20px; flex-wrap:wrap; }
  .col { flex:1 1 420px; }
  .card { background: var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
  .card h3 { margin: 0 0 10px 0; font-size: 18px; }
  .btn { padding:10px 14px; border-radius:10px; border:1px solid var(--border); cursor:pointer; margin-right:8px; background:#0f1525; color:var(--text);}
  .btn:hover { border-color:#2a355a; }
  .btn-primary { background: var(--primary); border-color: var(--primary-600); color: #fff; }
  .btn-danger  { background: var(--danger); border-color: #b91c1c; color: #fff; }
  .btn-ghost   { background: #0f1525; color: var(--muted); }
  .muted { color: var(--muted); }
  .kv { display:grid; grid-template-columns: 140px auto; gap:4px 10px; font-size:14px; }
  .kpi strong { font-size: 17px; }
  .meter { position:relative; width:100%; height:10px; background:#0b1223; border-radius:6px; overflow:hidden; border:1px solid var(--border);}
  .meter > span { position:absolute; left:0; top:0; height:100%; width:0%; background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444); transition: width .08s linear; }
  audio { width:100%; margin-top:10px; }
  input[type="range"] { width:100%; }
  table { width:100%; border-collapse: collapse; font-size:14px; }
  th, td { padding: 8px 8px; border-bottom:1px solid var(--border); }
  th { text-align:left; color: var(--muted); font-weight:600; }
  .imgbox { display:flex; align-items:center; justify-content:center; min-height:140px; border:1px dashed var(--border);
            border-radius:10px; background: var(--card); }
  a { color: var(--primary); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<header>
  <h2>🔤 Arabic Alphabet Realtime — <code>wav2vec2-base</code></h2>
</header>
<main>
  <div class="row">
    <div class="col">
      <div class="card">
        <h3>1) Record or Upload</h3>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <button id="btnStart" class="btn btn-primary">🎙️ Start</button>
          <button id="btnStop"  class="btn btn-danger" disabled>⏹ Stop</button>
          <button id="btnReset" class="btn btn-ghost" disabled>↺ Reset</button>
          <span class="muted">Status:</span><span id="recState">Idle</span>
          <span class="muted" style="margin-left:14px;">Duration:</span><span id="timer">0.0s</span>
        </div>
        <div style="margin-top:10px">
          <div class="meter"><span id="level" style="width:0%"></span></div>
        </div>
        <div style="margin-top:12px">
          <input type="file" id="upload" accept="audio/wav,audio/*" />
        </div>
        <audio id="preview" controls></audio>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>2) Settings</h3>
        <div>
          <label for="topk" class="muted">Top-K</label>
          <input id="topk" type="range" min="1" max="28" step="1" value="5" />
          <div><span class="muted">Value: </span><span id="topkVal">5</span></div>
        </div>
        <div style="margin-top:14px">
          <label for="fixed" class="muted">Auto-stop window (sec)</label>
          <input id="fixed" type="range" min="0.6" max="2.5" step="0.1" value="1.0" />
          <div><span class="muted">Value: </span><span id="fixedVal">1.0</span></div>
        </div>
        <div style="margin-top:16px">
          <button id="btnClassify" class="btn btn-primary" disabled>🚀 Classify</button>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>Top-1</h3>
        <div class="kv">
          <div class="muted">Label</div><div><strong id="top1Label">—</strong></div>
          <div class="muted">Probability</div><div><strong id="top1Prob">—</strong></div>
          <div class="muted">Latency</div><div><strong id="latency">—</strong></div>
        </div>
        <div class="muted" style="margin-top:10px" id="modelInfo">—</div>
      </div>
    </div>
  </div>

  <div class="row" style="margin-top:18px">
    <div class="col">
      <div class="card">
        <h3>Probabilities</h3>
        <div class="imgbox"><img id="probImg" style="max-width:100%"/></div>
      </div>
    </div>
    <div class="col">
      <div class="card">
        <h3>Waveform</h3>
        <div class="imgbox"><img id="waveImg" style="max-width:100%"/></div>
      </div>
    </div>
  </div>

  <div class="row" style="margin-top:18px">
    <div class="col">
      <div class="card">
        <h3>Top-K Table</h3>
        <table id="topkTable">
          <thead><tr><th style="text-align:right">rank</th><th>label</th><th>probability</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>Training Diagnostics</h3>
        <div><img id="confImg" style="max-width:100%; display:none"/></div>
        <div style="margin-top:8px">
          <a id="reportLink" class="muted" href="#" style="display:none" download="per_class_report.csv">Download per-class report</a>
        </div>
        <div id="diagNote" class="muted" style="margin-top:8px"></div>
      </div>
    </div>
  </div>
</main>

<script>
// --- Simple WAV encoder using WebAudio (16-bit PCM) ---
class WavRecorder {
  constructor() {
    this.chunks = [];
    this.sampleRate = 44100;
    this.recording = false;
    this._audioCtx = null;
    this._processor = null;
    this._source = null;
    this._stream = null;
    this.onlevel = null; // callback(rms: 0..1)
  }

  async start() {
    if (this.recording) return;
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this._audioCtx.sampleRate;
    const source = this._audioCtx.createMediaStreamSource(this._stream);
    const processor = this._audioCtx.createScriptProcessor(4096, 1, 1);
    this._source = source;
    this._processor = processor;
    this.chunks = [];
    processor.onaudioprocess = (e) => {
      if (!this.recording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
      if (this.onlevel) {
        // quick RMS for VU meter
        let sum = 0;
        for (let i=0;i<input.length;i++) { const v=input[i]; sum += v*v; }
        const rms = Math.sqrt(sum / input.length); // ~0..1
        this.onlevel(Math.min(1, rms*1.6));
      }
    };
    source.connect(processor);
    processor.connect(this._audioCtx.destination);
    this.recording = true;
  }

  async stop() {
    if (!this.recording) return null;
    this.recording = false;
    try { this._processor.disconnect(); } catch(_){}
    try { this._source.disconnect(); } catch(_){}
    try { this._stream.getTracks().forEach(t => t.stop()); } catch(_){}
    try { await this._audioCtx.close(); } catch(_){}

    // Merge float32 chunks
    const length = this.chunks.reduce((sum, arr) => sum + arr.length, 0);
    const pcm = new Float32Array(length);
    let offset = 0;
    for (const arr of this.chunks) { pcm.set(arr, offset); offset += arr.length; }
    return this._encodeWav(pcm, this.sampleRate);
  }

  _encodeWav(samples, sampleRate) {
    // Convert Float32 [-1,1] to 16-bit PCM
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    // RIFF header
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');

    // fmt subchunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);  // PCM
    view.setUint16(22, 1, true);  // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);  // block align
    view.setUint16(34, 16, true); // bits per sample

    // data subchunk
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM data
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], {type: 'audio/wav'});
  }
}

// --- UI Logic ---
const btnStart   = document.getElementById('btnStart');
const btnStop    = document.getElementById('btnStop');
const btnReset   = document.getElementById('btnReset');
const btnClassify= document.getElementById('btnClassify');
const timer      = document.getElementById('timer');
const upload     = document.getElementById('upload');
const preview    = document.getElementById('preview');
const topk       = document.getElementById('topk');
const topkVal    = document.getElementById('topkVal');
const fixed      = document.getElementById('fixed');
const fixedVal   = document.getElementById('fixedVal');

const top1Label  = document.getElementById('top1Label');
const top1Prob   = document.getElementById('top1Prob');
const latency    = document.getElementById('latency');
const modelInfo  = document.getElementById('modelInfo');
const probImg    = document.getElementById('probImg');
const waveImg    = document.getElementById('waveImg');
const topkTable  = document.getElementById('topkTable').querySelector('tbody');
const confImg    = document.getElementById('confImg');
const reportLink = document.getElementById('reportLink');
const diagNote   = document.getElementById('diagNote');
const level      = document.getElementById('level');
const recState   = document.getElementById('recState');

let recorder = null;
let currentBlob = null;
let timerId = null;
let autoStopId = null;
let t0 = 0;

function stopTimers(){
  if (timerId) clearInterval(timerId);
  if (autoStopId) clearTimeout(autoStopId);
  timerId = null; autoStopId = null;
}

topk.addEventListener('input', () => topkVal.textContent = topk.value);
fixed.addEventListener('input', () => fixedVal.textContent = fixed.value);

btnStart.addEventListener('click', async () => {
  btnStart.disabled = true; btnStop.disabled = false; btnClassify.disabled = true; btnReset.disabled = false;
  upload.value = '';
  currentBlob = null;
  recState.textContent = 'Recording';
  level.style.width = '0%';

  recorder = new WavRecorder();
  recorder.onlevel = (rms) => { level.style.width = Math.round(Math.min(100, rms*100)) + '%'; };
  await recorder.start();

  t0 = performance.now();
  timer.textContent = '0.0s';
  timerId = setInterval(() => {
    const dt = (performance.now() - t0) / 1000.0;
    timer.textContent = dt.toFixed(1) + 's';
  }, 100);

  // Auto-stop after chosen window
  const windowSec = parseFloat(fixed.value || '1.0');
  autoStopId = setTimeout(async ()=>{
    if (recorder && recorder.recording) {
      const blob = await recorder.stop();
      window.afterAutoStop(blob);
    }
  }, Math.max(100, windowSec * 1000));
});

btnStop.addEventListener('click', async () => {
  btnStop.disabled = true;
  if (recorder && recorder.recording) {
    const blob = await recorder.stop();
    currentBlob = blob;
    preview.src = URL.createObjectURL(blob);
    preview.play();
  }
  stopTimers();
  level.style.width = '0%';                // clear VU meter
  recState.textContent = 'Captured';
  btnStart.disabled = false;               // allow new recordings
  btnClassify.disabled = !currentBlob ? true : false;
  recorder = null;                          // fresh recorder next time
});

window.afterAutoStop = async (blob)=>{
  currentBlob = blob;
  preview.src = URL.createObjectURL(blob);
  preview.play();

  // fully leave "recording" state (this fixed the bug)
  stopTimers();
  level.style.width = '0%';
  btnStart.disabled = false;

  btnStop.disabled = true;
  btnClassify.disabled = false;
  recState.textContent = 'Captured';
  recorder = null;                          // allow a brand-new recorder next time
};

btnReset.addEventListener('click', async ()=>{
  // stop if still recording
  try { if (recorder && recorder.recording) { await recorder.stop(); } } catch(_){}
  stopTimers();
  level.style.width = '0%';

  // wipe UI
  currentBlob = null;
  preview.src = '';
  probImg.src = '';
  waveImg.src = '';
  topkTable.innerHTML = '';
  top1Label.textContent = '—';
  top1Prob.textContent  = '—';
  latency.textContent   = '—';
  recState.textContent  = 'Idle';
  timer.textContent     = '0.0s';

  // restore controls
  btnStart.disabled    = false;
  btnStop.disabled     = true;
  btnClassify.disabled = true;
  btnReset.disabled    = true;
  recorder = null;
});

upload.addEventListener('change', () => {
  if (upload.files && upload.files[0]) {
    currentBlob = upload.files[0];
    preview.src = URL.createObjectURL(currentBlob);
    preview.play();
    btnClassify.disabled = false;
    btnReset.disabled = false;
    // Ensure we're not in a recording state
    stopTimers();
    level.style.width = '0%';
    btnStart.disabled = false;
    btnStop.disabled = true;
    recState.textContent = 'Loaded file';
  }
});

btnClassify.addEventListener('click', async () => {
  if (!currentBlob) return;
  btnClassify.disabled = true;
  try {
    const fd = new FormData();
    fd.append('audio', currentBlob, 'input.wav');
    fd.append('top_k', topk.value);
    fd.append('fixed_seconds', fixed.value);

    const res = await fetch('/infer', { method: 'POST', body: fd });
    if (!res.ok) {
      const text = await res.text();
      alert('Request failed: ' + text);
      btnClassify.disabled = false;
      return;
    }
    const data = await res.json();

    // Update UI
    top1Label.textContent = data.top1_label;
    top1Prob.textContent  = data.top1_prob.toFixed(6);
    latency.textContent   = data.latency_ms.toFixed(1) + ' ms';
    modelInfo.innerHTML   = data.model_info_html;
    probImg.src           = data.prob_b64;
    waveImg.src           = data.wave_b64;

    // Top-K table
    topkTable.innerHTML = '';
    data.topk.forEach((row, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td style="text-align:right">${idx+1}</td><td>${row[0]}</td><td>${row[1].toFixed(6)}</td>`;
      topkTable.appendChild(tr);
    });

    // Diagnostics
    if (data.confusion_available) {
      confImg.style.display = 'block';
      confImg.src = '/confusion';
      reportLink.style.display = data.per_class_available ? 'inline-block' : 'none';
      reportLink.href = '/per_class_report';
      diagNote.textContent = '';
    } else {
      confImg.style.display = 'none';
      reportLink.style.display = 'none';
      diagNote.textContent = 'No training confusion matrix/report found.';
    }
  } catch (err) {
    alert('Error: ' + err);
  } finally {
    btnClassify.disabled = false;
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    _load_once()
    return HTMLResponse(INDEX_HTML)


# --------------------------- Inference endpoint ---------------------------

@app.post("/infer", response_class=JSONResponse)
async def infer(audio: UploadFile = File(...), top_k: str = "5", fixed_seconds: str = "1.0"):
    model = _load_once()

    # Read WAV (client encodes WAV; server needs only soundfile)
    content = await audio.read()
    try:
        y, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
    except Exception as e:
        return PlainTextResponse(f"Could not read audio (expect WAV). Error: {type(e).__name__}: {e}", status_code=400)

    if y.ndim > 1:
        y = y.mean(axis=1)

    try:
        tk = int(top_k)
    except:
        tk = 5
    try:
        fx = float(fixed_seconds)
    except:
        fx = 1.0

    out = model.infer_numpy(y, sr_in=sr, top_k=tk, fixed_seconds=fx)

    # Visuals
    wave_b64 = plot_waveform_b64(y, sr)
    labels_all = [model.id2label[i] for i in range(len(model.id2label))]
    prob_b64 = plot_prob_bars_b64(labels_all, np.asarray(out["probs"]), top=min(tk, len(labels_all)))

    # Stats
    info_html = (
        f"<div><b>Model</b>: facebook/wav2vec2-base</div>"
        f"<div><b>Device</b>: {model.device.type}</div>"
        f"<div><b>Sampling rate</b>: {model.sampling_rate} Hz</div>"
        f"<div><b>Num labels</b>: {len(model.id2label)}</div>"
        f"<div><b>Parameters</b>: {model.num_params:,}</div>"
        f"<div><b>Size</b>: {model.size_mb} MB</div>"
    )

    return JSONResponse({
        "top1_label": out["top1_label"],
        "top1_prob": float(out["top1_prob"]),
        "topk": out["topk"],
        "latency_ms": float(out["latency_ms"]),
        "prob_b64": prob_b64,
        "wave_b64": wave_b64,
        "model_info_html": info_html,
        "confusion_available": _CONF_PNG is not None,
        "per_class_available": _CONF_CSV is not None,
    })


# --------------------------- CLI & optional ngrok share ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--share", action="store_true", help="Expose a public link via ngrok (requires NGROK_AUTHTOKEN).")
    return ap.parse_args()

def start_uvicorn_in_thread(host: str, port: int, log_level: str):
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, workers=1)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread

def maybe_start_ngrok(host: str, port: int):
    # Requires: pip install pyngrok && export NGROK_AUTHTOKEN=...
    try:
        from pyngrok import ngrok, conf
    except Exception:
        print("[WARN] --share requested but pyngrok not installed. `pip install pyngrok` and set NGROK_AUTHTOKEN.")
        return None
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        print("[WARN] --share requested but NGROK_AUTHTOKEN is not set. Set it to enable public link.")
        return None
    conf.get_default().auth_token = token
    tunnel = ngrok.connect(addr=f"http://{host}:{port}", proto="http")
    print(f"[SHARE] Public URL: {tunnel.public_url}")
    return tunnel

if __name__ == "__main__":
    args = parse_args()
    # Ensure model loads before serving, so first request isn't slow/failing.
    _load_once()

    if args.share:
        # Start server in background and open ngrok tunnel
        srv, th = start_uvicorn_in_thread(args.host, args.port, args.log_level)
        maybe_start_ngrok(args.host, args.port)
        th.join()
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
