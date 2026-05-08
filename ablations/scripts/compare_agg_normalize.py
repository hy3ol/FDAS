"""
compare_agg_normalize.py — channel aggregation + normalization comparison.

For every dataset that has scores_per_ch.npz AND predictions_{train,val}.npy
under results/{dataset_id}/, compute nine scoring variants:

  raw_{max,mean,median}        — D_w_c(t) aggregated across channels
                                  (no normalization)
  z_train_{max,mean,median}    — train-baseline z-score per channel,
                                  then aggregated
  z_val_{max,mean,median}      — val-baseline z-score per channel,
                                  then aggregated

For each variant TSB-AD metrics are computed (VUS-PR / VUS-ROC / AUC-PR /
AUC-ROC / Standard-F1 / PA-F1) via the same edge-fill pipeline as 05_metrics.

Output:
  results/04_metrics/agg_normalize_per_dataset.csv  (per-dataset metrics)
  results/04_metrics/agg_normalize_summary.csv      (family-level mean per variant)

Usage:
  python scripts/compare_agg_normalize.py                          # all 199
  python scripts/compare_agg_normalize.py --family Exathlon
  python scripts/compare_agg_normalize.py --only KEY1 KEY2 ...

Real-time / GT-free properties of the variants:
  - raw_*:       GT-free, real-time, no normalization (channel scale dominates).
  - z_train_*:   GT-free, real-time. Baseline from train predictions
                 (model-overfit bias).
  - z_val_*:     GT-free, real-time. Baseline from val predictions
                 (held-out, no overfit bias — recommended).
"""
from __future__ import annotations

import argparse
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
# np.nanmax/nanmean/nanmedian on all-NaN rows return NaN with a warning.
# We intentionally allow some all-NaN rows (NaN_in_Y_hat propagation) and
# the downstream edge-fill handles them — the warnings are pure noise.
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

OUT_DIR = RESULTS_ROOT / "04_metrics"
EPS = 1e-8

VARIANTS = [
    "raw_max", "raw_mean", "raw_median",
    "z_train_max", "z_train_mean", "z_train_median",
]
METRIC_KEYS = ["VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
               "Standard-F1", "PA-F1"]


def edge_fill(length: int, t_idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.full(length, np.nan, dtype=np.float64)
    out[t_idx] = vals
    fin = np.isfinite(out)
    if not np.any(fin):
        return np.zeros_like(out)
    valid = np.flatnonzero(fin)
    f, l = int(valid[0]), int(valid[-1])
    out[:f] = out[f]
    out[l + 1 :] = out[l]
    last = out[f]
    for i in range(f, l + 1):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out


def _zscore(Dwc: np.ndarray, base: dict) -> tuple[np.ndarray, int]:
    mu, sigma = base["mean"], base["std"]
    valid_c = np.isfinite(sigma) & (sigma > EPS)
    Z = np.full_like(Dwc, np.nan)
    if valid_c.any():
        Z[:, valid_c] = (Dwc[:, valid_c] - mu[valid_c][None, :]) / sigma[valid_c][None, :]
    return Z, int(valid_c.sum())


def evaluate_one(key: str) -> dict | None:
    pc_path = RESULTS_ROOT / key / "scores_per_ch.npz"
    val_pred_path = RESULTS_ROOT / key / "predictions_val.npy"
    train_pred_path = RESULTS_ROOT / key / "predictions_train.npy"
    if not pc_path.exists() or not val_pred_path.exists() or not train_pred_path.exists():
        return None
    bundle = prepare_dataset_bundle(key)
    pc = load_per_channel_scores(pc_path.parent)
    Dwc = pc["D_w_c"]                                # (T_eval, C) — train+test (V13 full-series)
    t_idx = pc["t"].astype(np.int64)                  # global indices into full series
    # V13 full-series eval — TSB-AD-M aligned. Use bundle.full_labels (length
    # T_full = train + test concatenated) and edge_fill on length T_full.
    label = bundle.full_labels.astype(np.int64)
    full_len = int(bundle.total_timesteps)
    if np.unique(label).size < 2:
        return None
    sw = compute_tsb_sliding_window(bundle.full_values_raw)

    base_train = compute_train_baseline_stats(
        np.load(train_pred_path), bundle.train_values_norm,
        lookback=192, pred_len=96,
    )
    Zt, n_kept_train = _zscore(Dwc, base_train)

    aggs = {
        "raw_max":          np.nanmax(Dwc, axis=1),
        "raw_mean":         np.nanmean(Dwc, axis=1),
        "raw_median":       np.nanmedian(Dwc, axis=1),
        "z_train_max":      np.nanmax(Zt, axis=1),
        "z_train_mean":     np.nanmean(Zt, axis=1),
        "z_train_median":   np.nanmedian(Zt, axis=1),
    }

    rec: dict = {
        "dataset_id": key,
        "family":     key.split("_id_")[0] if "_id_" in key else key,
        "n_eval_rows": int(t_idx.size),
        "n_channels":  int(Dwc.shape[1]),
        "n_kept_train_baseline": n_kept_train,
        "sliding_window":  int(sw),
    }
    for name, vals in aggs.items():
        full = edge_fill(full_len, t_idx, vals)
        m = compute_tsb_metrics(full, label, sw)
        for k in METRIC_KEYS:
            rec[f"{k.lower().replace('-', '_')}_{name}"] = float(m.get(k, np.nan))
    return rec


def _evaluate_with_key(key: str) -> tuple[str, dict | None, str | None]:
    """Worker entry point — wraps evaluate_one with the key in the result tuple."""
    try:
        rec = evaluate_one(key)
        return (key, rec, None)
    except Exception as exc:
        return (key, None, f"{type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None, help="specific dataset keys")
    p.add_argument("--family", default=None,
                   help="restrict to one family prefix (e.g. Exathlon, MSL, CATSv2)")
    p.add_argument("--workers", type=int, default=None,
                   help="number of parallel processes. default: cpu_count() // 2 "
                        "(set 1 to disable multiprocessing).")
    args = p.parse_args()

    if args.only:
        keys = args.only
    else:
        keys = sorted([d.name for d in RESULTS_ROOT.iterdir()
                       if d.is_dir()
                       and (d / "scores_per_ch.npz").exists()
                       and (d / "predictions_val.npy").exists()
                       and (d / "predictions_train.npy").exists()])
    if args.family:
        keys = [k for k in keys if k.startswith(args.family)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) // 2)
    workers = max(1, min(workers, len(keys)))
    print(f"compare_agg_normalize — {len(keys)} dataset(s), workers={workers}")

    rows: list[dict] = []

    if workers == 1:
        # serial path (debug / single-CPU)
        for i, key in enumerate(keys, 1):
            print(f"  [{i:>3}/{len(keys)}] {key} ...", end="", flush=True)
            _, rec, err = _evaluate_with_key(key)
            if err is not None:
                print(f"  FAIL ({err})")
                continue
            if rec is None:
                print("  SKIP (missing artifacts or single-class)")
                continue
            rows.append(rec)
            print(f"  ✓ VUS-PR raw_max={rec['vus_pr_raw_max']:.3f}  "
                  f"z_train_max={rec['vus_pr_z_train_max']:.3f}  "
                  f"z_train_med={rec['vus_pr_z_train_median']:.3f}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_evaluate_with_key, k): k for k in keys}
            for i, fut in enumerate(as_completed(futures), 1):
                key, rec, err = fut.result()
                if err is not None:
                    print(f"  [{i:>3}/{len(keys)}] {key} FAIL ({err})")
                    continue
                if rec is None:
                    print(f"  [{i:>3}/{len(keys)}] {key} SKIP "
                          f"(missing artifacts or single-class)")
                    continue
                rows.append(rec)
                print(f"  [{i:>3}/{len(keys)}] {key} ✓ "
                      f"VUS-PR raw_max={rec['vus_pr_raw_max']:.3f}  "
                      f"z_train_med={rec['vus_pr_z_train_median']:.3f}  "
                      f"z_val_med={rec['vus_pr_z_val_median']:.3f}")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / ("agg_normalize_per_dataset.csv"
                          if not args.family
                          else f"agg_normalize_{args.family}_per_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ {out_path}")

    # Family summary on each metric (9 variants)
    summary_rows: list[dict] = []
    for fam, g in df.groupby("family"):
        row = {"family": fam, "n": int(len(g))}
        for k in METRIC_KEYS:
            kk = k.lower().replace("-", "_")
            for v in VARIANTS:
                row[f"{k}_{v}"] = float(np.nanmean(g[f"{kk}_{v}"]))
        summary_rows.append(row)
    fam_df = pd.DataFrame(summary_rows).sort_values("family")
    fam_path = OUT_DIR / ("agg_normalize_summary.csv"
                          if not args.family
                          else f"agg_normalize_{args.family}_summary.csv")
    fam_df.to_csv(fam_path, index=False)
    print(f"  ✓ {fam_path}")

    # Console: VUS-PR family table (headline metric)
    print("\n=== VUS-PR by family (mean across datasets) ===")
    header = (
        f"{'family':<14} {'n':>3}  "
        f"{'raw_max':>7} {'raw_mn':>7} {'raw_med':>7}  "
        f"{'tr_max':>7} {'tr_mn':>7} {'tr_med':>7}  "
        f"{'val_max':>7} {'val_mn':>7} {'val_med':>7}"
    )
    print(header)
    for _, r in fam_df.iterrows():
        cells = [
            f"{r['family']:<14}",
            f"{int(r['n']):>3}",
        ]
        for v in VARIANTS:
            cells.append(f"{r[f'VUS-PR_{v}']:>7.4f}")
        # group by spaces
        line = (cells[0] + " " + cells[1] + "  "
                + " ".join(cells[2:5]) + "  "
                + " ".join(cells[5:8]) + "  "
                + " ".join(cells[8:11]))
        print(line)

    print("\n=== overall mean (all datasets) ===")
    for k in METRIC_KEYS:
        kk = k.lower().replace("-", "_")
        means = {v: float(np.nanmean(df[f"{kk}_{v}"])) for v in VARIANTS}
        print(f"  {k:<12}: " +
              "  ".join(f"{v}={means[v]:.4f}" for v in VARIANTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
