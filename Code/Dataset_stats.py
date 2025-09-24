#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_stats.py

Verify and prepare an Arabic alphabet speech dataset organized as:

Dataset/
  ALIF/
    *.wav
  BAA/
    *.wav
  ...
(28 class subfolders, each holding .wav files)

What this script does:
1) Validates dataset structure & audio files
2) Computes per-class & global stats (counts, durations, hours)
3) Detects duplicate files (by content hash)
4) Writes:
   - stats.json (machine-readable summary)
   - stats.csv (per-class table)
   - manifest_full.csv (all valid files with label, duration, sr)
   - train.csv / val.csv / test.csv (stratified splits)
5) (Optional) Prepares a cleaned copy: mono, target sample rate, 16-bit PCM,
   peak-normalized, written into --prepared_dir, and manifests point there.

Dependencies:
- Pure Python stdlib validation uses the `wave` module.
- If installed, `soundfile` (pysoundfile) improves robustness & enables audio prep.
- If installed, `numpy` and `scipy` (or `resampy`) allow resampling on --prepare-audio.

Usage examples:
    python dataset_stats.py --dataset_dir Dataset
    python dataset_stats.py --dataset_dir Dataset --out_dir Code/dataset_reports
    python dataset_stats.py --dataset_dir Dataset --prepare-audio \
        --prepared_dir Dataset_prepared --target-sr 16000 --mono --normalize-peak

Tips:
- Run first WITHOUT --prepare-audio to inspect stats & warnings.
- If happy, run with --prepare-audio to create a consistent training set.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional libraries
HAVE_SF = False
HAVE_NUMPY = False
HAVE_SCIPY = False
HAVE_RESAMPY = False

try:
    import soundfile as sf  # type: ignore
    HAVE_SF = True
except Exception:
    pass

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:
    pass

try:
    from scipy.signal import resample_poly  # type: ignore
    HAVE_SCIPY = True
except Exception:
    pass

try:
    import resampy  # type: ignore
    HAVE_RESAMPY = True
except Exception:
    pass

import contextlib
import wave

# -------------------------- Helpers & Data Structures --------------------------

VALID_EXTS = {".wav", ".wave"}  # keep it simple and robust

@dataclass
class FileRecord:
    path: str
    rel_path: str
    label: str
    samplerate: Optional[int]
    n_channels: Optional[int]
    sample_width: Optional[int]  # bytes per sample (wave-only)
    duration_sec: Optional[float]
    valid: bool
    reason: Optional[str] = None
    md5: Optional[str] = None

def read_wav_info_stdlib(filepath: Path) -> Tuple[bool, Optional[int], Optional[int], Optional[int], Optional[float], Optional[str]]:
    """Read WAV info using Python's stdlib wave module."""
    try:
        with contextlib.closing(wave.open(str(filepath), 'rb')) as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()  # bytes per sample
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = float(n_frames) / float(framerate) if framerate > 0 else 0.0
            # Basic sanity
            if framerate <= 0 or n_frames <= 0 or duration <= 0.0:
                return False, framerate, n_channels, sampwidth, duration, "zero_or_invalid_duration"
            return True, framerate, n_channels, sampwidth, duration, None
    except Exception as e:
        return False, None, None, None, None, f"wave_error:{type(e).__name__}"

def read_audio_sf(filepath: Path) -> Tuple[bool, Optional[int], Optional[int], Optional[float], Optional[str]]:
    """Use soundfile (if available) to read meta; returns (ok, sr, channels, duration, reason)."""
    try:
        info = sf.info(str(filepath))
        sr = info.samplerate
        ch = info.channels
        frames = info.frames
        dur = frames / float(sr) if sr > 0 else 0.0
        if sr <= 0 or frames <= 0 or dur <= 0.0:
            return False, sr, ch, dur, "zero_or_invalid_duration"
        return True, sr, ch, dur, None
    except Exception as e:
        return False, None, None, None, f"sf_error:{type(e).__name__}"

def md5_of_file(filepath: Path, chunk_size: int = 1 << 20) -> str:
    m = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

def ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def natural_class_name(name: str) -> str:
    # Keep folder name as-is; you can normalize here if needed.
    return name.strip()

def resample_and_prepare_audio(
    in_path: Path,
    out_path: Path,
    target_sr: int,
    to_mono: bool,
    normalize_peak: bool
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Prepare audio using soundfile + numpy + (resampy or scipy.signal.resample_poly).
    - Reads the audio fully
    - Converts to mono if requested
    - Resamples to target_sr if needed
    - Peak-normalizes if requested
    - Saves as 16-bit PCM WAV
    Returns: (ok, reason_if_any, duration_sec)
    """
    if not HAVE_SF or not HAVE_NUMPY:
        return False, "prepare_requires_soundfile_numpy", None

    try:
        y, sr = sf.read(str(in_path), always_2d=True)  # shape: (n_frames, n_channels)
        # to mono
        if to_mono and y.shape[1] > 1:
            y = np.mean(y, axis=1, keepdims=True)
        else:
            # keep as-is
            pass

        y = y.squeeze(-1)  # (n_frames,)

        # resample if needed
        if target_sr and sr != target_sr:
            if HAVE_RESAMPY:
                y = resampy.resample(y, sr, target_sr)
            elif HAVE_SCIPY:
                # resample_poly is efficient & decent quality
                from math import gcd
                g = gcd(sr, target_sr)
                up = target_sr // g
                down = sr // g
                y = resample_poly(y, up, down)
            else:
                return False, "resampling_requires_resampy_or_scipy", None
            sr = target_sr

        # peak normalize
        if normalize_peak and y.size > 0:
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak * 0.98  # avoid clipping margin

        # write as 16-bit PCM
        ensure_out_dir(out_path.parent)
        sf.write(str(out_path), y, sr, subtype="PCM_16")

        duration = float(len(y)) / float(sr) if sr > 0 else 0.0
        return True, None, duration

    except Exception as e:
        return False, f"prepare_error:{type(e).__name__}", None

# -------------------------- Core Processing --------------------------

def scan_dataset(
    dataset_dir: Path,
    accept_exts: set[str],
    min_duration: float,
    max_duration: float,
) -> Tuple[List[FileRecord], Dict[str, List[FileRecord]], List[FileRecord], List[str]]:
    """
    Walk the dataset folder and collect file records.
    Returns:
      - all_records: every file record (valid+invalid)
      - by_class: valid records grouped by class label
      - invalid_records: invalid/bad files
      - warnings: list of warning messages
    """
    warnings: List[str] = []
    all_records: List[FileRecord] = []
    invalid_records: List[FileRecord] = []
    by_class: Dict[str, List[FileRecord]] = defaultdict(list)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"[FATAL] --dataset_dir '{dataset_dir}' does not exist or is not a directory.")

    class_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise SystemExit("[FATAL] No class subfolders were found in dataset_dir.")

    for cdir in class_dirs:
        label = natural_class_name(cdir.name)
        audio_files = sorted([p for p in cdir.rglob("*") if p.suffix.lower() in accept_exts and p.is_file()])

        if not audio_files:
            warnings.append(f"[WARN] Class '{label}' is EMPTY (no .wav files).")

        for f in audio_files:
            rel_path = str(f.relative_to(dataset_dir))
            ext = f.suffix.lower()

            # Basic file presence
            if not f.exists():
                rec = FileRecord(str(f), rel_path, label, None, None, None, None, False, "missing")
                all_records.append(rec)
                invalid_records.append(rec)
                continue

            # Read metadata: prefer soundfile if available
            ok = False
            sr = ch = sampwidth = None
            dur = None
            reason = None

            if HAVE_SF:
                ok_sf, sr_sf, ch_sf, dur_sf, reason_sf = read_audio_sf(f)
                if ok_sf:
                    ok = True
                    sr, ch, dur = sr_sf, ch_sf, dur_sf
                    sampwidth = None  # not provided by SF
                else:
                    # Fall back to wave
                    ok_w, sr_w, ch_w, sw_w, dur_w, reason_w = read_wav_info_stdlib(f)
                    ok = ok_w
                    sr, ch, sampwidth, dur, reason = sr_w, ch_w, sw_w, dur_w, reason_w or reason_sf
            else:
                ok_w, sr_w, ch_w, sw_w, dur_w, reason_w = read_wav_info_stdlib(f)
                ok = ok_w
                sr, ch, sampwidth, dur, reason = sr_w, ch_w, sw_w, dur_w, reason_w

            # Duration sanity
            if ok and dur is not None:
                if dur <= 0.0:
                    ok = False
                    reason = "zero_duration"
                elif dur < min_duration:
                    # Keep as valid but warn: often short samples are OK for phonemes; you decide
                    warnings.append(f"[WARN] Very short file ({dur:.3f}s): {rel_path}")
                elif dur > max_duration:
                    warnings.append(f"[WARN] Very long file ({dur:.1f}s): {rel_path}")

            # Hash for duplicates (only compute for readable files)
            file_md5 = None
            if ok:
                try:
                    file_md5 = md5_of_file(f)
                except Exception as e:
                    ok = False
                    reason = f"hash_error:{type(e).__name__}"

            rec = FileRecord(str(f), rel_path, label, sr, ch, sampwidth, dur, ok, reason, file_md5)
            all_records.append(rec)
            if ok:
                by_class[label].append(rec)
            else:
                invalid_records.append(rec)

    # Class existence warnings
    for label in sorted(set(p.name for p in class_dirs)):
        if label not in by_class or len(by_class[label]) == 0:
            warnings.append(f"[WARN] No valid files in class '{label}'.")

    return all_records, by_class, invalid_records, warnings

def find_duplicates(valid_records: List[FileRecord]) -> Dict[str, List[FileRecord]]:
    """Group files by MD5 hash; return only groups with duplicates."""
    dup_map: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in valid_records:
        if r.md5:
            dup_map[r.md5].append(r)
    return {k: v for k, v in dup_map.items() if len(v) > 1}

def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_out_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def stratified_split(
    records: List[FileRecord],
    train: float,
    val: float,
    test: float,
    seed: int
) -> Tuple[List[FileRecord], List[FileRecord], List[FileRecord]]:
    import random
    random.seed(seed)
    by_label: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)

    train_set: List[FileRecord] = []
    val_set: List[FileRecord] = []
    test_set: List[FileRecord] = []

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(train * n))
        n_val = int(round(val * n))
        n_test = n - n_train - n_val
        # adjust if rounding off
        if n_test < 0:
            n_test = 0
        # If rounding caused mismatch, fix last
        if n_train + n_val + n_test != n:
            n_test = n - n_train - n_val
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train+n_val])
        test_set.extend(items[n_train+n_val:])
    return train_set, val_set, test_set

def maybe_prepare_audio(
    records: List[FileRecord],
    dataset_dir: Path,
    prepared_dir: Path,
    target_sr: int,
    to_mono: bool,
    normalize_peak: bool
) -> Tuple[List[FileRecord], List[str]]:
    """
    If preparation requested, create cleaned copies and return updated records
    pointing to prepared paths. Also collect warnings.
    """
    warnings: List[str] = []
    prepared_records: List[FileRecord] = []

    for r in records:
        in_path = Path(r.path)
        out_path = prepared_dir / Path(r.rel_path)
        ok, reason, new_dur = resample_and_prepare_audio(
            in_path, out_path, target_sr, to_mono, normalize_peak
        )
        if ok:
            # Update record to point to prepared version
            new_md5 = md5_of_file(out_path)
            new_rec = FileRecord(
                str(out_path),
                str(out_path.relative_to(prepared_dir)),
                r.label,
                target_sr if target_sr else r.samplerate,
                1 if to_mono else r.n_channels,
                None,  # sample width not tracked post-prep
                new_dur if new_dur is not None else r.duration_sec,
                True,
                None,
                new_md5
            )
            prepared_records.append(new_rec)
        else:
            warnings.append(f"[WARN] Failed to prepare '{r.rel_path}': {reason}. Using original.")
            prepared_records.append(r)  # fall back to original

    return prepared_records, warnings

# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify and prepare an Arabic alphabet dataset for training."
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to Dataset/ root.")
    parser.add_argument("--out_dir", type=str, default="dataset_reports", help="Where to write reports & manifests.")
    parser.add_argument("--min-duration", type=float, default=0.05, help="Warn if shorter than this (seconds).")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Warn if longer than this (seconds).")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for splits.")
    parser.add_argument("--accept-exts", type=str, nargs="*", default=list(VALID_EXTS), help="Accepted audio extensions.")
    # Preparation options
    parser.add_argument("--prepare-audio", action="store_true", help="Create cleaned copies for consistent training.")
    parser.add_argument("--prepared_dir", type=str, default="Dataset_prepared", help="Output root for prepared audio.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate for prepared audio.")
    parser.add_argument("--mono", action="store_true", help="Convert to mono when preparing audio.")
    parser.add_argument("--normalize-peak", action="store_true", help="Peak-normalize prepared audio.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_out_dir(out_dir)

    # Validate split ratios
    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("[FATAL] train+val+test must equal 1.0")

    # Scan dataset
    start = time.time()
    all_records, by_class, invalid_records, warnings = scan_dataset(
        dataset_dir,
        accept_exts={e.lower() if e.startswith('.') else f'.{e.lower()}' for e in args.accept_exts},
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    valid_records = [r for r in all_records if r.valid]
    dup_groups = find_duplicates(valid_records)

    # Optional: prepare audio copies
    if args.prepare_audio:
        prepared_dir = Path(args.prepared_dir).resolve()
        ensure_out_dir(prepared_dir)
        if not HAVE_SF or not HAVE_NUMPY:
            warnings.append("[WARN] --prepare-audio requested but soundfile/numpy not available; skipping audio prep.")
        else:
            valid_records, prep_warnings = maybe_prepare_audio(
                valid_records, dataset_dir, prepared_dir, args.target_sr, args.mono, args.normalize_peak
            )
            warnings.extend(prep_warnings)

    # Recompute class grouping after (optional) prep
    by_class_final: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in valid_records:
        by_class_final[r.label].append(r)

    # Summaries
    class_rows = []
    total_files = 0
    total_duration = 0.0
    sr_counter = Counter()
    ch_counter = Counter()

    for label, recs in sorted(by_class_final.items(), key=lambda kv: kv[0]):
        n = len(recs)
        durs = [r.duration_sec for r in recs if r.duration_sec is not None]
        dur_sum = float(sum(durs)) if durs else 0.0
        dur_min = min(durs) if durs else 0.0
        dur_max = max(durs) if durs else 0.0

        # SR/Channels distribution (rough)
        for r in recs:
            if r.samplerate:
                sr_counter[r.samplerate] += 1
            if r.n_channels:
                ch_counter[r.n_channels] += 1

        total_files += n
        total_duration += dur_sum
        class_rows.append({
            "label": label,
            "num_files": n,
            "total_duration_s": round(dur_sum, 3),
            "total_duration_h": round(dur_sum / 3600.0, 4),
            "min_dur_s": round(dur_min, 3),
            "max_dur_s": round(dur_max, 3),
            "avg_dur_s": round((dur_sum / n) if n > 0 else 0.0, 3),
        })

    # Write per-class CSV
    stats_csv_path = out_dir / "stats.csv"
    write_csv(stats_csv_path, class_rows,
              ["label", "num_files", "total_duration_s", "total_duration_h", "min_dur_s", "max_dur_s", "avg_dur_s"])

    # Write full manifest of valid files
    manifest_rows = []
    for r in valid_records:
        manifest_rows.append({
            "path": r.path,
            "rel_path": r.rel_path if not args.prepare_audio else r.rel_path,  # relative to prepared_dir if used
            "label": r.label,
            "samplerate": r.samplerate if r.samplerate is not None else "",
            "channels": r.n_channels if r.n_channels is not None else "",
            "duration_s": round(r.duration_sec, 6) if r.duration_sec is not None else "",
            "md5": r.md5 or "",
        })
    manifest_full_path = out_dir / "manifest_full.csv"
    write_csv(manifest_full_path, manifest_rows,
              ["path", "rel_path", "label", "samplerate", "channels", "duration_s", "md5"])

    # Write invalid file report
    if invalid_records:
        invalid_rows = []
        for r in invalid_records:
            invalid_rows.append({
                "path": r.path,
                "rel_path": r.rel_path,
                "label": r.label,
                "reason": r.reason or "unknown",
            })
        invalid_csv_path = out_dir / "invalid_files.csv"
        write_csv(invalid_csv_path, invalid_rows, ["path", "rel_path", "label", "reason"])

    # Duplicate groups report
    if dup_groups:
        dup_rows = []
        for md5sum, group in dup_groups.items():
            for r in group:
                dup_rows.append({"md5": md5sum, "label": r.label, "rel_path": r.rel_path, "duration_s": r.duration_sec})
        write_csv(out_dir / "duplicates.csv", dup_rows, ["md5", "label", "rel_path", "duration_s"])

    # Stratified splits
    train_set, val_set, test_set = stratified_split(valid_records, args.train, args.val, args.test, args.seed)
    for split_name, split_records in [("train", train_set), ("val", val_set), ("test", test_set)]:
        split_rows = []
        for r in split_records:
            split_rows.append({
                "path": r.path,
                "label": r.label,
                "duration_s": round(r.duration_sec, 6) if r.duration_sec is not None else "",
                "samplerate": r.samplerate if r.samplerate is not None else "",
                "channels": r.n_channels if r.n_channels is not None else "",
            })
        write_csv(out_dir / f"{split_name}.csv", split_rows, ["path", "label", "duration_s", "samplerate", "channels"])

    # Write JSON summary
    summary = {
        "dataset_dir": str(dataset_dir),
        "prepared_audio": bool(args.prepare_audio),
        "prepared_dir": str(Path(args.prepared_dir).resolve()) if args.prepare_audio else None,
        "num_classes": len(by_class_final),
        "classes": sorted(by_class_final.keys()),
        "total_valid_files": len(valid_records),
        "total_invalid_files": len(invalid_records),
        "total_duration_s": round(total_duration, 3),
        "total_duration_h": round(total_duration / 3600.0, 4),
        "samplerate_distribution": dict(sr_counter.most_common()),
        "channel_distribution": dict(ch_counter.most_common()),
        "num_duplicate_groups": len(dup_groups),
        "split_sizes": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "warnings": warnings,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0"
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"[OK] Finished in {elapsed:.2f}s")
    print(f"[OK] Reports written to: {out_dir}")
    if warnings:
        print(f"[NOTE] {len(warnings)} warnings. See stats.json for details.")
    if invalid_records:
        print(f"[NOTE] {len(invalid_records)} invalid files. See invalid_files.csv.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        raise
    except Exception:
        print("[FATAL] Uncaught exception:\n" + traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
