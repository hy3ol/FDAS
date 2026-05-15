"""Reimplementation of scripts/01_data_preparation.py split + scaler logic,
exposed as a function so the horizon ablation runner can vary pred_len without
modifying production code.

Mirrors load_and_prepare_data() / split + StandardScaler / metadata + bundle_meta
write-out, but lets the caller pass (lookback, pred_len) explicitly and write
the bundle_meta to an arbitrary location instead of results/<key>/.

This module is intentionally a near-verbatim copy of the production split logic
so that ablation runs are bit-for-bit identical to what 01 would produce given
the same (lookback, pred_len). Drift risk: if production 01 changes its split
or scaler logic, this file must be re-synced. Tagged with PRODUCTION_REF for
diff-tracking.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Reach production helpers (artifact_paths.resolve_dataset, DATA_DIR).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from artifact_paths import DATA_DIR, resolve_dataset, save_data_metadata  # type: ignore

# PRODUCTION_REF: scripts/01_data_preparation.py
TRAIN_RATIO_WITHIN_TRAINVAL = 0.70


class SegmentTooShortError(ValueError):
    """Raised when train/val/test segment can't fit a single (L+H) window."""


@dataclass
class PreparedSplit:
    dataset_key: str
    num_channels: int
    train_data_norm: np.ndarray
    val_data_norm: np.ndarray
    test_data_norm: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    metadata: dict
    bundle_meta: dict


# PRODUCTION_REF: scripts/01_data_preparation.py::parse_filename
def _parse_filename(filename: str) -> dict:
    parts = filename.replace(".csv", "").split("_")
    tr_idx = parts.index("tr")
    first_idx = parts.index("1st")
    return {
        "id": int(parts[0]),
        "dataset": parts[1],
        "train_size": int(parts[tr_idx + 1]),
        "first_anomaly_idx": int(parts[first_idx + 1]),
        "filename": filename,
    }


def prepare_split(dataset_key: str, lookback: int, pred_len: int) -> PreparedSplit:
    """Bit-for-bit equivalent to running 01_data_preparation.py with the
    requested (lookback, pred_len), but returns the splits in memory instead
    of writing to data/.

    Raises SegmentTooShortError if any of train / val / test fails the
    required-span guard. Caller catches and logs as skip_too_short.
    """
    info = resolve_dataset(dataset_key)
    data_path = info["path"]
    dataset_name = info["filename"]
    parsed = _parse_filename(dataset_name)

    df = pd.read_csv(data_path)
    label_col = "Label" if "Label" in df.columns else "is_anomaly"
    value_cols = [c for c in df.columns if c != label_col]
    values = df[value_cols].values
    labels = df[label_col].values
    num_channels = len(value_cols)
    total_timesteps = len(values)

    anomaly_indices = np.where(labels == 1)[0]
    first_anomaly_idx = (
        int(anomaly_indices[0]) if len(anomaly_indices) > 0 else len(labels)
    )

    required_span = lookback + pred_len
    official_train_end = min(parsed["train_size"], total_timesteps)
    if official_train_end < required_span:
        raise SegmentTooShortError(
            f"train_too_short: train={official_train_end} < required={required_span}"
        )

    train_end_nominal = int(official_train_end * TRAIN_RATIO_WITHIN_TRAINVAL)
    train_end_nominal = max(train_end_nominal, required_span)
    disjoint_val_len = official_train_end - train_end_nominal
    uses_overlap_val = disjoint_val_len < required_span

    if uses_overlap_val:
        train_end = official_train_end
        val_start = official_train_end - required_span
    else:
        train_end = train_end_nominal
        val_start = train_end

    train_val_end = official_train_end
    test_start = official_train_end
    train_data = values[:train_end]
    train_labels = labels[:train_end]
    val_data = values[val_start:train_val_end]
    val_labels = labels[val_start:train_val_end]
    test_data = values[test_start:]
    test_labels = labels[test_start:]

    if len(val_data) < required_span:
        raise SegmentTooShortError(
            f"val_too_short: val={len(val_data)} < required={required_span}"
        )
    if len(test_data) < required_span:
        raise SegmentTooShortError(
            f"test_too_short: test={len(test_data)} < required={required_span}"
        )

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data_norm = scaler.transform(train_data)
    val_data_norm = scaler.transform(val_data)
    test_data_norm = scaler.transform(test_data)

    metadata = {
        "dataset_name": dataset_name,
        "dataset": parsed["dataset"],
        "dataset_key": dataset_key,
        "dataset_relative_path": info["relative_path"],
        "dataset_source_path": str(data_path),
        "num_channels": num_channels,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "total_timesteps": total_timesteps,
        "anomaly_ratio": float(test_labels.mean()),
        "first_anomaly_idx": first_anomaly_idx,
        "official_train_end": official_train_end,
        "train_end": train_end,
        "train_val_end": train_val_end,
        "val_start": val_start,
        "test_start": test_start,
        "val_mode": "overlap-tail" if uses_overlap_val else "disjoint",
        "train_anomalies": int(train_labels.sum()),
        "val_anomalies": int(val_labels.sum()),
        "test_anomalies": int(test_labels.sum()),
        "lookback": lookback,
        "pred_len": pred_len,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    bundle_meta = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "official_train_end": official_train_end,
        "scaler_train_end": train_end,
        "train_end": train_end,
        "val_start": val_start,
        "test_start": test_start,
        "val_mode": "overlap-tail" if uses_overlap_val else "disjoint",
        "total_timesteps": total_timesteps,
        "num_channels": num_channels,
        "lookback": lookback,
        "pred_len": pred_len,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return PreparedSplit(
        dataset_key=dataset_key,
        num_channels=num_channels,
        train_data_norm=train_data_norm,
        val_data_norm=val_data_norm,
        test_data_norm=test_data_norm,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        metadata=metadata,
        bundle_meta=bundle_meta,
    )


def write_data_dir(split: PreparedSplit) -> None:
    """Write the in-memory split to data/ so production 02_train.py / 03_inference.py
    pick it up (they read from data/metadata.json + data/{train,val,test}_*.npy).

    Overwrites data/ each call; orchestrator runs (H, key) iterations serially
    so no concurrency. Production 01 is also single-active-dataset so this
    mirrors the established pattern.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "train_data.npy", split.train_data_norm)
    np.save(DATA_DIR / "train_labels.npy", split.train_labels)
    np.save(DATA_DIR / "val_data.npy", split.val_data_norm)
    np.save(DATA_DIR / "val_labels.npy", split.val_labels)
    np.save(DATA_DIR / "test_data.npy", split.test_data_norm)
    np.save(DATA_DIR / "test_labels.npy", split.test_labels)
    save_data_metadata(split.metadata)


def write_bundle_meta(split: PreparedSplit, bundle_meta_path: Path) -> None:
    """Write the bundle_meta to an ablation-specific location.

    Production location (results/<key>/bundle_meta.json) is reserved for the
    H=96 paper-grade artifact and not touched.
    """
    bundle_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bundle_meta_path, "w") as f:
        json.dump(split.bundle_meta, f, indent=4)
