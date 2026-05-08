"""
_ablation_no_drop.py — approach B isolation ablation.

For every dataset (predictions_train + predictions_val + scores_per_ch present),
compute the same 9 variants as compare_agg_normalize.py BUT WITHOUT dropping
σ ≤ ε channels. Instead apply a numerical floor:

    σ_safe = max(σ, ε_floor)        ε_floor = 1e-8

This way every channel survives into the aggregation. Bad channels with
σ ≈ 0 produce large z values (because (D − μ)/ε is huge). The point is to
quantify how the channel-drop policy affects each aggregation:

  - max  : sensitive — large z values from bad channels dominate.
  - mean : sensitive — bad channels pull the mean.
  - median : robust — large z values become outliers ignored by median.

Compare these "no-drop" results to the current "with-drop" numbers in
agg_normalize_per_dataset.csv.

Outputs (under V13/results/04_metrics/):
  ablation_no_drop_per_dataset.csv
  ablation_no_drop_summary.csv     family means
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
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="overflow encountered")

from artifact_paths import RESULTS_ROOT
from score_utils import (
    compute_train_baseline_stats,
    compute_tsb_metrics,
    compute_tsb_sliding_window,
    load_per_channel_scores,
    prepare_dataset_bundle,
)

EPS_FLOOR = 1e-8
METRIC_KEYS = ["VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
               "Standard-F1", "PA-F1"]
VARIANTS = [
    "raw_max", "raw_mean", "raw_median",
    "z_train_max_nodrop", "z_train_mean_nodrop", "z_train_median_nodrop",
]


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


def _zscore_nodrop(Dwc: np.ndarray, base: dict) -> np.ndarray:
    """Z-score WITHOUT dropping channels — use σ_safe = max(σ, ε)."""
    mu = np.asarray(base["mean"], dtype=np.float64)
    sigma = np.asarray(base["std"], dtype=np.float64)
    # numerical floor: keep every channel, just protect divide
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > EPS_FLOOR),
                          sigma, EPS_FLOOR)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    return (Dwc - mu_safe[None, :]) / sigma_safe[None, :]


def evaluate_one(key: str) -> dict | None:
    pc_path = RESULTS_ROOT / key / "scores_per_ch.npz"
    train_pred_path = RESULTS_ROOT / key / "predictions_train.npy"
    if not (pc_path.exists() and train_pred_path.exists()):
        return None
    bundle = prepare_dataset_bundle(key)
    pc = load_per_channel_scores(pc_path.parent)
    Dwc = pc["D_w_c"]
    t_idx = pc["t"].astype(np.int64)
    label = bundle.test_labels.astype(np.int64)
    test_len = label.size
    if np.unique(label).size < 2: return None
    sw = compute_tsb_sliding_window(bundle.full_values_raw)

    base_train = compute_train_baseline_stats(
        np.load(train_pred_path), bundle.train_values_norm,
        lookback=192, pred_len=96,
    )
    Zt = _zscore_nodrop(Dwc, base_train)

    aggs = {
        "raw_max":    np.nanmax(Dwc, axis=1),
        "raw_mean":   np.nanmean(Dwc, axis=1),
        "raw_median": np.nanmedian(Dwc, axis=1),
        "z_train_max_nodrop":    np.nanmax(Zt, axis=1),
        "z_train_mean_nodrop":   np.nanmean(Zt, axis=1),
        "z_train_median_nodrop": np.nanmedian(Zt, axis=1),
    }

    sigma_t = base_train["std"]
    rec = {
        "dataset_id": key,
        "family": key.split("_id_")[0] if "_id_" in key else key,
        "n_channels": int(Dwc.shape[1]),
        "n_train_above_floor": int(np.sum(np.isfinite(sigma_t) & (sigma_t > EPS_FLOOR))),
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
    print(f"ablation no-drop — {len(keys)} datasets, workers={workers}")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_wrap, k): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            k, rec, err = fut.result()
            if err: print(f"  [{i:>3}] {k} FAIL {err}"); continue
            if rec is None: print(f"  [{i:>3}] {k} SKIP"); continue
            rows.append(rec)
            print(f"  [{i:>3}/{len(keys)}] {k}  "
                  f"VUS-PR raw_max={rec['vus_pr_raw_max']:.3f}  "
                  f"z_tr_max_nd={rec['vus_pr_z_train_max_nodrop']:.3f}  "
                  f"z_tr_med_nd={rec['vus_pr_z_train_median_nodrop']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / "04_metrics"
    df.to_csv(out_dir / "ablation_no_drop_per_dataset.csv", index=False)

    # family summary
    rows_s = []
    for fam, g in df.groupby("family"):
        row = {"family": fam, "n": int(len(g))}
        for k in METRIC_KEYS:
            kk = k.lower().replace("-","_")
            for v in VARIANTS:
                row[f"{k}_{v}"] = float(np.nanmean(g[f"{kk}_{v}"]))
        rows_s.append(row)
    fam_df = pd.DataFrame(rows_s).sort_values("family")
    fam_df.to_csv(out_dir / "ablation_no_drop_summary.csv", index=False)

    print("\n=== overall mean (n={}) — VUS-PR ===".format(len(df)))
    for v in VARIANTS:
        a = df[f"vus_pr_{v}"].mean()
        print(f"  {v:<26}  {a:.4f}")
    print("\nsaved:")
    print("  ", out_dir / "ablation_no_drop_per_dataset.csv")
    print("  ", out_dir / "ablation_no_drop_summary.csv")


if __name__ == "__main__":
    main()
