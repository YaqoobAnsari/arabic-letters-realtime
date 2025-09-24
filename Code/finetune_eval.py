#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_eval.py — production-grade finetuning + evaluation for Arabic alphabet (28-class) audio classification.

Repo layout (auto-detected relative to this file):
  <root>/
    Code/
      finetune_eval.py
      dataset_reports/{train,val,test}.csv
    Dataset/<Class>/*.wav
    Models/<model_name>/*
    Results/<model_name>/*

Requirements:
  pip install -U transformers datasets torchaudio soundfile numpy scipy scikit-learn matplotlib accelerate evaluate
  # optional:
  pip install resampy
"""

from __future__ import annotations
import json, time, shutil, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from inspect import signature

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import resampy
    HAVE_RESAMPY = True
except Exception:
    HAVE_RESAMPY = False

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------- Paths & Defaults --------------------------

DEFAULT_MODELS = [
    "facebook/wav2vec2-base",
    "facebook/hubert-large-ls960-ft",
    "microsoft/wavlm-base-plus",
    "facebook/data2vec-audio-base",
]

def resolve_repo_paths() -> Dict[str, Path]:
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    return {
        "root": root,
        "code": code_dir,
        "dataset": root / "Dataset",
        "models": root / "Models",
        "results": root / "Results",
        "reports": code_dir / "dataset_reports",
    }


# -------------------------- Data Utilities --------------------------

class AudioCSV(Dataset):
    """
    Manifest columns: path,label,(duration_s,samplerate,channels optional)
    Returns raw mono PCM float32 @ target_sr.
    """
    def __init__(self, csv_path: Path, label2id: Dict[str, int], target_sr: int):
        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"{csv_path} must have 'path' and 'label' columns.")
        self.label2id = label2id
        self.target_sr = target_sr

        if self.df["path"].isna().any():
            raise ValueError(f"{csv_path} has rows with missing path.")
        if self.df["label"].isna().any():
            raise ValueError(f"{csv_path} has rows with missing label.")
        unknown = set(self.df["label"].unique()) - set(label2id.keys())
        if unknown:
            raise ValueError(f"Unknown labels in {csv_path}: {unknown}")

        self.paths = self.df["path"].tolist()
               # ensure strings (robustness to numbers in CSV)
        self.paths = [str(p) for p in self.paths]
        self.labels = [label2id[l] for l in self.df["label"].tolist()]

    def __len__(self):
        return len(self.paths)

    def _read_audio(self, p: str) -> Tuple[np.ndarray, int]:
        try:
            y, sr = sf.read(p, always_2d=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio: {p} ({type(e).__name__}: {e})")
        if hasattr(y, "ndim") and y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != self.target_sr:
            if HAVE_RESAMPY:
                y = resampy.resample(y, sr, self.target_sr)
            else:
                duration = len(y) / float(sr)
                new_len = max(1, int(round(duration * self.target_sr)))
                y = np.interp(np.linspace(0, len(y), new_len, endpoint=False),
                              np.arange(len(y)), y)
            sr = self.target_sr
        y = np.asarray(y, dtype=np.float32)
        return y, sr

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        audio, _ = self._read_audio(path)
        return {"audio": audio, "label": label, "path": path}


@dataclass
class Collator:
    processor: AutoProcessor
    fixed_seconds: float = 1.0  # window length (e.g., 1.0 or 2.0 sec)

    def __call__(self, batch: List[Dict]):
        audios, labels = [], []
        for b in batch:
            if "audio" in b:
                audios.append(b["audio"])
            elif "input_values" in b:
                audios.append(b["input_values"])
            else:
                raise KeyError(f"Batch item missing audio. Keys={list(b.keys())}")
            if "label" not in b:
                raise KeyError(f"Batch item missing label. Keys={list(b.keys())}")
            labels.append(b["label"])

        sr = getattr(self.processor, "sampling_rate", None)
        if sr is None and hasattr(self.processor, "feature_extractor"):
            sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        if sr is None:
            sr = 16000

        max_len = int(round(self.fixed_seconds * sr))  # fixed window in samples

        inputs = self.processor(
            audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
        )
        # be explicit to silence mask dtype warnings on some models
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs


# -------------------------- Metrics & Reporting --------------------------

def compute_metrics_builder(id2label: Dict[int, str]):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "mcc": matthews_corrcoef(labels, preds),
        }
    return _compute_metrics


def save_confusion_and_report(y_true, y_pred, id2label: Dict[int, str], out_dir: Path):
    labels_order = list(range(len(id2label)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[id2label[i] for i in labels_order], columns=[id2label[i] for i in labels_order])
    (out_dir / "confusion_matrix.csv").write_text(cm_df.to_csv())
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(labels_order))
    plt.xticks(ticks, [id2label[i] for i in labels_order], rotation=90)
    plt.yticks(ticks, [id2label[i] for i in labels_order])
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Predicted")
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in labels_order], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "per_class_report.csv")


def measure_latency(model, processor, ds: Dataset, device: torch.device, batch_size: int = 16, max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    torch.set_grad_enabled(False)
    n = min(len(ds), batch_size * max_batches)
    if n == 0:
        return {"avg_ms_per_example": float("nan"), "n_examples": 0}
    idxs = list(range(n))
    total_time = 0.0
    total_examples = 0
    collate = Collator(processor)
    for i in range(0, n, batch_size):
        batch_idx = idxs[i:i+batch_size]
        batch = [ds[j] for j in batch_idx]
        inputs = collate(batch)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_examples += len(batch_idx)
    avg_ms = (total_time / max(1, total_examples)) * 1000.0
    return {"avg_ms_per_example": avg_ms, "n_examples": total_examples}


def human_model_size_bytes(model: AutoModelForAudioClassification) -> int:
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    return total


# -------------------------- Training / Eval Pipeline --------------------------

def build_training_args(output_dir, **kw) -> TrainingArguments:
    """
    Version-agnostic TrainingArguments:
    - Filter unsupported kwargs
    - Align eval/save strategies when load_best_model_at_end=True
    - Keep raw columns (remove_unused_columns=False) so our 'audio' never gets dropped
    """
    allowed = set(signature(TrainingArguments).parameters.keys())
    filtered = {k: v for k, v in kw.items() if k in allowed}

    eval_supported = "evaluation_strategy" in allowed
    save_supported = "save_strategy" in allowed

    if "remove_unused_columns" in allowed:
        filtered["remove_unused_columns"] = False

    if not eval_supported:
        for k in ("load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
            filtered.pop(k, None)
    else:
        lbme = filtered.get("load_best_model_at_end", False)
        if lbme:
            save_strat = filtered.get("save_strategy", "no") if save_supported else "no"
            eval_strat = filtered.get("evaluation_strategy", None)
            if eval_strat is None or eval_strat == "no":
                if save_strat in ("epoch", "steps"):
                    filtered["evaluation_strategy"] = save_strat
                else:
                    for k in ("load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
                        filtered.pop(k, None)
            elif save_supported and save_strat != eval_strat:
                filtered["evaluation_strategy"] = save_strat

    if "per_device_eval_batch_size" not in allowed and "per_device_eval_batch_size" in filtered:
        filtered.pop("per_device_eval_batch_size", None)

    return TrainingArguments(output_dir=str(output_dir), **filtered)


def build_trainer(model, training_args, ds_train, ds_val, collator, compute_metrics, processor):
    """
    Version-agnostic Trainer init + robust loss:
    - prefer `processing_class`, fall back to `tokenizer`
    - custom CETrainer.compute_loss handles HF passing extra kwargs (e.g., num_items_in_batch)
    - computes CE on logits explicitly to guarantee a proper autograd path
    """
    class CETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            # get logits from ModelOutput or tuple/dict
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs[0]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    allowed = set(signature(Trainer.__init__).parameters.keys())
    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    if "processing_class" in allowed:
        kwargs["processing_class"] = processor
    elif "tokenizer" in allowed:
        kwargs["tokenizer"] = None
    return CETrainer(**kwargs)


def already_trained(result_dir: Path) -> bool:
    """Decide if a model should be skipped. We consider a run done if summary.json exists."""
    return (result_dir / "summary.json").exists()


def train_one_model(
    model_id: str,
    paths: Dict[str, Path],
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    seed: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    fp16: bool,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    force_retrain: bool = False,
    window_seconds: float = 1.0,
) -> None:

    safe_name = model_id.replace("/", "__")
    model_out_dir = paths["models"] / safe_name
    result_dir = paths["results"] / safe_name
    paths["models"].mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already trained (unless forced)
    if already_trained(result_dir) and not force_retrain:
        print(f"[SKIP] {model_id} — found {result_dir/'summary.json'}; use --force_retrain to override.")
        return

    # Labels
    train_df = pd.read_csv(train_csv)
    labels_sorted = sorted(train_df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels_sorted)

    # Processor
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        fe = AutoFeatureExtractor.from_pretrained(model_id)
        class Wrapper:
            def __init__(self, fe):
                self.feature_extractor = fe
                self.sampling_rate = getattr(fe, "sampling_rate", 16000)
            def __call__(self, *args, **kwargs):
                return self.feature_extractor(*args, **kwargs)
        processor = Wrapper(fe)

    target_sr = getattr(processor, "sampling_rate", None)
    if target_sr is None and hasattr(processor, "feature_extractor"):
        target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    if target_sr is None:
        target_sr = 16000

    # Datasets
    ds_train = AudioCSV(train_csv, label2id, target_sr)
    ds_val   = AudioCSV(val_csv,   label2id, target_sr)
    ds_test  = AudioCSV(test_csv,  label2id, target_sr)

    # Model
    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Device + ensure trainable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    training_args = build_training_args(
        model_out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=fp16 and (device.type == "cuda"),
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        save_total_limit=2,
        report_to=[],
        max_grad_norm=1.0,
        seed=seed,
    )

    # Trainer (version-safe) with fixed window seconds in collator
    collator = Collator(processor, fixed_seconds=window_seconds)
    compute_metrics = compute_metrics_builder(id2label)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        ds_train=ds_train,
        ds_val=ds_val,
        collator=collator,
        compute_metrics=compute_metrics,
        processor=processor,
    )

    # Train
    train_result = trainer.train()
    trainer.save_model(model_out_dir)

    # Evaluate
    val_metrics = trainer.evaluate(eval_dataset=ds_val)
    test_metrics = trainer.evaluate(eval_dataset=ds_test)

    # Predictions
    logits, labels, _ = trainer.predict(ds_test)
    preds = np.argmax(logits, axis=-1)

    # Reports
    save_confusion_and_report(labels, preds, id2label, result_dir)
    latency_info = measure_latency(trainer.model, processor, ds_test, device=device, batch_size=16, max_batches=50)
    size_mb = round(human_model_size_bytes(trainer.model) / (1024 * 1024), 2)

    summary = {
        "model_id": model_id,
        "safe_name": safe_name,
        "num_labels": num_labels,
        "labels": labels_sorted,
        "training": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "fp16": training_args.fp16 if hasattr(training_args, "fp16") else False,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "window_seconds": window_seconds,
        },
        "metrics": {
            "val": {k: float(v) for k, v in val_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()},
        },
        "latency_ms_per_example": latency_info.get("avg_ms_per_example"),
        "latency_examples_measured": latency_info.get("n_examples"),
        "model_size_mb": size_mb,
        "paths": {"models_dir": str(model_out_dir), "results_dir": str(result_dir)},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pred_df = pd.DataFrame({
        "path": [ds_test.paths[i] for i in range(len(ds_test))],
        "true_label_id": labels,
        "true_label": [id2label[int(x)] for x in labels],
        "pred_label_id": preds,
        "pred_label": [id2label[int(x)] for x in preds],
    })
    pred_df.to_csv(result_dir / "test_predictions.csv", index=False, encoding="utf-8")

    # Convenience copy
    dest_model_copy = result_dir / "model"
    if dest_model_copy.exists():
        shutil.rmtree(dest_model_copy)
    shutil.copytree(model_out_dir, dest_model_copy)


# -------------------------- Main --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Finetune & evaluate multiple audio models for Arabic letter classification.")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="List of HF model ids.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true", help="Use fp16 if CUDA available.")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--force_retrain", action="store_true", help="Ignore existing Results/<model>/summary.json and retrain.")
    p.add_argument("--window_seconds", type=float, default=1.0, help="Fixed audio window length used by the collator.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    paths = resolve_repo_paths()
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_csv) if args.train_csv else (paths["code"] / "dataset_reports" / "train.csv")
    val_csv   = Path(args.val_csv)   if args.val_csv   else (paths["code"] / "dataset_reports" / "val.csv")
    test_csv  = Path(args.test_csv)  if args.test_csv  else (paths["code"] / "dataset_reports" / "test.csv")

    for pth in [train_csv, val_csv, test_csv]:
        if not pth.exists():
            raise SystemExit(f"[FATAL] Missing manifest file: {pth}")

    print(f"[INFO] Using manifests:\n  train={train_csv}\n  val  ={val_csv}\n  test ={test_csv}")
    print(f"[INFO] Saving models under: {paths['models']}")
    print(f"[INFO] Saving results under: {paths['results']}")

    for mid in args.models:
        print("\n" + "="*80)
        print(f"[RUN] Fine-tuning: {mid}")
        print("="*80)
        train_one_model(
            model_id=mid,
            paths=paths,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            seed=args.seed,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            gradient_accumulation_steps=args.grad_accum,
            dataloader_num_workers=args.workers,
            force_retrain=args.force_retrain,
            window_seconds=args.window_seconds,
        )

    print("\n[OK] All done. Check the Results/ folder for per-model summaries.")


if __name__ == "__main__":
    main()
