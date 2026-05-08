"""
_ablation_channel_mask.py — channel-filter isolation ablation.

For every dataset (predictions_train + scores_per_ch present), compute three
variants on the SAME D_w_c table:

  raw_max         — current: max over ALL channels
  raw_max_intersect  — max only over channels with σ_c^train > ε  (same mask as z)
  z_train_max     — current: train-baseline z + max over σ-valid channels

Headline question: how much of the +0.020 VUS-PR gap between z_train_max and
raw_max comes from the channel filter (σ ≤ ε channels dropped) vs from the
z-score itself? raw_max_intersect isolates the filter effect.

Outputs (under V13/results/04_metrics/):
  ablation_channel_mask_per_dataset.csv
  ablation_channel_mask_summary.csv     family means + Δ
"""
from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# Production scripts live at V13/scripts/; this file is at V13/ablations/scripts/.
sys.path.append(str(SCRIPT_DIR.parent.parent / "scripts"))

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Mean of empty slice")

from artifact_paths import RESULTS_ROOT
from score_utils import (
    compute_train_baseline_stats,
    compute_tsb_metrics,
    compute_tsb_sliding_window,
    load_per_channel_scores,
    prepare_dataset_bundle,
)

EPS = 1e-8
METRIC_KEYS = ["VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC"]


def edge_fill(length, t_idx, vals):
    out = np.full(length, np.nan, dtype=np.float64)
    out[t_idx] = vals
    fin = np.isfinite(out)
    if not np.any(fin): return np.zeros_like(out)
    valid = np.flatnonzero(fin); f, l = int(valid[0]), int(valid[-1])
    out[:f] = out[f]; out[l+1:] = out[l]
    last = out[f]
    for i in range(f, l+1):
        if np.isfinite(out[i]): last = out[i]
        else: out[i] = last
    return out


def evaluate_one(key: str) -> dict | None:
    pc_path = RESULTS_ROOT / key / "scores_per_ch.npz"
    train_pred_path = RESULTS_ROOT / key / "predictions_train.npy"
    if not pc_path.exists() or not train_pred_path.exists():
        return None
    bundle = prepare_dataset_bundle(key)
    pc = load_per_channel_scores(pc_path.parent)
    Dwc = pc["D_w_c"]
    t_idx = pc["t"].astype(np.int64)
    label = bundle.test_labels.astype(np.int64)
    test_len = label.size
    if np.unique(label).size < 2: return None
    sw = compute_tsb_sliding_window(bundle.full_values_raw)

    # train baseline
    base = compute_train_baseline_stats(
        np.load(train_pred_path), bundle.train_values_norm,
        lookback=192, pred_len=96,
    )
    mu, sigma = base["mean"], base["std"]
    valid_c = np.isfinite(sigma) & (sigma > EPS) & np.isfinite(mu)
    n_total = int(Dwc.shape[1])
    n_valid = int(valid_c.sum())

    aggs = {}
    # raw_max — ALL channels
    aggs["raw_max_all"] = np.nanmax(Dwc, axis=1)
    # raw_max — INTERSECT (only train-σ-valid channels)
    if valid_c.any():
        aggs["raw_max_intersect"] = np.nanmax(Dwc[:, valid_c], axis=1)
    else:
        aggs["raw_max_intersect"] = np.full(Dwc.shape[0], np.nan)
    # z_train_max
    if valid_c.any():
        Z = (Dwc[:, valid_c] - mu[valid_c][None,:]) / sigma[valid_c][None,:]
        aggs["z_train_max"] = np.nanmax(Z, axis=1)
    else:
        aggs["z_train_max"] = np.full(Dwc.shape[0], np.nan)

    rec = {
        "dataset_id": key,
        "family": key.split("_id_")[0] if "_id_" in key else key,
        "n_total_channels": n_total,
        "n_valid_channels": n_valid,
        "n_dropped": n_total - n_valid,
    }
    for name, vals in aggs.items():
        full = edge_fill(test_len, t_idx, vals)
        m = compute_tsb_metrics(full, label, sw)
        for k in METRIC_KEYS:
            rec[f"{k.lower().replace('-','_')}_{name}"] = float(m.get(k, np.nan))
    return rec


def _wrap(key):
    try:
        return key, evaluate_one(key), None
    except Exception as exc:
        return key, None, f"{type(exc).__name__}: {exc}"


def main():
    keys = sorted([d.name for d in RESULTS_ROOT.iterdir()
                   if d.is_dir()
                   and (d/"scores_per_ch.npz").exists()
                   and (d/"predictions_train.npy").exists()])
    workers = max(1, (os.cpu_count() or 2) // 2)
    print(f"ablation — {len(keys)} datasets, workers={workers}")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_wrap, k): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            k, rec, err = fut.result()
            if err: print(f"  [{i:>3}] {k} FAIL {err}"); continue
            if rec is None: print(f"  [{i:>3}] {k} SKIP"); continue
            rows.append(rec)
            print(f"  [{i:>3}/{len(keys)}] {k}  drop={rec['n_dropped']:>3}/{rec['n_total_channels']:>3}  "
                  f"VUS-PR raw_all={rec['vus_pr_raw_max_all']:.3f}  "
                  f"raw_∩={rec['vus_pr_raw_max_intersect']:.3f}  "
                  f"z={rec['vus_pr_z_train_max']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / "04_metrics"
    df.to_csv(out_dir / "ablation_channel_mask_per_dataset.csv", index=False)

    # family summary
    rows_s = []
    for fam, g in df.groupby("family"):
        row = {"family": fam, "n": int(len(g))}
        for k in METRIC_KEYS:
            kk = k.lower().replace("-","_")
            row[f"{k}_raw_all"]       = float(np.nanmean(g[f"{kk}_raw_max_all"]))
            row[f"{k}_raw_intersect"] = float(np.nanmean(g[f"{kk}_raw_max_intersect"]))
            row[f"{k}_z_train_max"]   = float(np.nanmean(g[f"{kk}_z_train_max"]))
        rows_s.append(row)
    fam_df = pd.DataFrame(rows_s).sort_values("family")
    fam_df.to_csv(out_dir / "ablation_channel_mask_summary.csv", index=False)

    # console: VUS-PR overall
    print("\n=== overall mean (n={}) ===".format(len(df)))
    for k in METRIC_KEYS:
        kk = k.lower().replace("-","_")
        a = df[f"{kk}_raw_max_all"].mean()
        b = df[f"{kk}_raw_max_intersect"].mean()
        c = df[f"{kk}_z_train_max"].mean()
        print(f"  {k:<10}  raw_all={a:.4f}  raw_intersect={b:.4f}  "
              f"z_train_max={c:.4f}    Δ(z−raw_all)={c-a:+.4f}  "
              f"Δ(z−raw_intersect)={c-b:+.4f}")
    print("\nsaved:")
    print("  ", out_dir / "ablation_channel_mask_per_dataset.csv")
    print("  ", out_dir / "ablation_channel_mask_summary.csv")


if __name__ == "__main__":
    main()
