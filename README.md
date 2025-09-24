# Arabic Alphabet Realtime Recognition

> **A production-ready FastAPI application for real-time Arabic letter recognition using wav2vec2-base**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/🤗%20Model-yansari/arabic--letters--wav2vec2--base-yellow)](https://huggingface.co/yansari/arabic-letters-wav2vec2-base)

## 🎯 Overview

A lean, production-style system that recognizes **28 Arabic letters** from short audio snippets in real-time. Features:

- 🎙️ **Browser-based recording** - No ffmpeg required
- ⚡ **GPU/CPU support** - Automatic detection and optimization
- ☁️ **Pre-trained models** - Hosted on Hugging Face for easy deployment
- 🔄 **Real-time inference** - Low-latency audio classification
- 📊 **Rich visualization** - Waveforms, probability charts, confusion matrices

## 🚀 Quick Start

### Linux/macOS (Tested)

```bash
# Clone the repository
git clone https://github.com/YaqoobAnsari/arabic-letters-realtime.git
cd arabic-letters-realtime

# Install dependencies and download model
bash scripts/install.sh

# Run the server
source .venv/bin/activate
python Code/serve_realtime_fastapi.py --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 in your browser and start recording!

### Windows (Git Bash - Experimental)

```bash
bash scripts/install.sh
source .venv/Scripts/activate
python Code/serve_realtime_fastapi.py --host 0.0.0.0 --port 7860
```

### Public Sharing (Optional)

```bash
pip install pyngrok
export NGROK_AUTHTOKEN="YOUR_TOKEN"
python Code/serve_realtime_fastapi.py --host 0.0.0.0 --port 7860 --share
```

## 🏗️ Project Structure

```
arabic-letters-realtime/
├── Code/
│   ├── serve_realtime_fastapi.py    # FastAPI server
│   ├── finetune_eval.py             # Training pipeline
│   └── dataset_reports/             # CSV manifests
├── Models/
│   └── facebook__wav2vec2-base/     # Downloaded model weights
├── Results/
│   └── facebook__wav2vec2-base/     # Training results & metrics
└── scripts/
    ├── install.sh                   # Cross-platform installer
    └── run.sh                       # Server launcher
```

## 🔧 Model Details

| Feature | Value |
|---------|--------|
| **Base Model** | facebook/wav2vec2-base |
| **Classes** | 28 Arabic letters |
| **Parameters** | ~95M |
| **Sample Rate** | 16 kHz |
| **Window Size** | 1.0 second |
| **Model Size** | ~350-400 MB |

## 🎯 Training Your Own Model

### Dataset Format

Create CSV files with columns: `path,label` (optional: `duration_s,samplerate,channels`)

```csv
path,label
audio/alef_001.wav,ا
audio/baa_001.wav,ب
audio/taa_001.wav,ت
```

### Training Command

```bash
source .venv/bin/activate

python Code/finetune_eval.py \
  --models facebook/wav2vec2-base \
  --num_epochs 8 \
  --batch_size 32 \
  --lr 2e-5 \
  --fp16 \
  --window_seconds 1.0
```

### Training Outputs

- `Results/facebook__wav2vec2-base/summary.json` - Overall metrics
- `Results/facebook__wav2vec2-base/confusion_matrix.png` - Confusion matrix
- `Results/facebook__wav2vec2-base/per_class_report.csv` - Per-class metrics
- `Results/facebook__wav2vec2-base/model/` - Inference-ready model

## 📊 Features

### Web Interface
- **Real-time recording** with visual feedback
- **Top-K predictions** with confidence scores
- **Waveform visualization** of recorded audio
- **Performance metrics** (accuracy, latency)
- **Confusion matrix** display (if available)

### API Endpoints
- `POST /predict` - Audio classification
- `GET /` - Web interface
- `GET /health` - Health check

## 🛠️ Advanced Configuration

### Training Parameters

```bash
python Code/finetune_eval.py \
  --models facebook/wav2vec2-base \
  --num_epochs 12 \
  --batch_size 16 \
  --lr 1e-5 \
  --grad_accum 2 \
  --window_seconds 1.5 \
  --force_retrain \
  --train_csv custom_train.csv \
  --val_csv custom_val.csv \
  --test_csv custom_test.csv
```

### Server Options

```bash
python Code/serve_realtime_fastapi.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --reload
```

## 📋 System Requirements

- **OS**: Linux (tested), macOS, Windows (Git Bash - experimental)
- **Python**: 3.9+
- **Memory**: 4GB+ RAM recommended
- **GPU**: Optional but recommended (NVIDIA with CUDA)

## 🔍 Troubleshooting

### Common Issues

**Port already in use**
```bash
python Code/serve_realtime_fastapi.py --port 8000
```

**Linux: Missing libsndfile**
```bash
sudo apt-get update && sudo apt-get install -y libsndfile1
```

**CUDA not detected**
- Ensure `nvidia-smi` works
- Installer falls back to CPU if CUDA unavailable

**Hugging Face download errors**
- Check internet connection
- Verify Hugging Face Hub access

## 🔒 Security & Privacy

- ✅ **Local processing** - Audio stays on your device/server
- ✅ **No telemetry** - No data sent to third parties  
- ✅ **Mic permissions** - Standard browser security model

## 🗺️ Roadmap

- [ ] Streaming audio endpoint for continuous recognition
- [ ] Mobile-optimized interface
- [ ] ONNX export for edge deployment
- [ ] Model quantization for faster inference
- [ ] Support for Arabic word recognition
- [ ] Docker containerization

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{ansari2025arabic,
  title={Arabic Letters Realtime Recognition},
  author={Ansari, Y.},
  year={2025},
  url={https://github.com/YaqoobAnsari/arabic-letters-realtime}
}

@inproceedings{baevski2020wav2vec,
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  booktitle={Advances in neural information processing systems},
  year={2020}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the wav2vec2-base model
- **Hugging Face** for the transformers library and model hosting
- **FastAPI** team for the excellent web framework

---

<div align="center">

**[🤗 Model Card](https://huggingface.co/yansari/arabic-letters-wav2vec2-base)** • 
**[📖 Documentation](#)** • 
**[🐛 Issues](../../issues)** • 
**[💬 Discussions](../../discussions)**

Made with ❤️ for the Arabic NLP community

</div>