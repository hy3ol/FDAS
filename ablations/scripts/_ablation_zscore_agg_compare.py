"""
_ablation_zscore_agg_compare.py — z_train aggregation comparison.

For each of the 200 datasets, takes the existing per-channel D_w_c
(full-series global indexing) and the existing train-only baseline
(μ_c, σ_c from predictions_train), and aggregates the channel-z-scores
three ways:

    z_train_max(t)    = max_c    (D_w_c(t) − μ_c) / σ_c
    z_train_median(t) = median_c (D_w_c(t) − μ_c) / σ_c
    z_train_mean(t)   = mean_c   (D_w_c(t) − μ_c) / σ_c

Edge-fills each to length T_full, evaluates TSB-AD-M metrics
(VUS-PR, VUS-ROC, AUC-PR, AUC-ROC, Standard-F1, PA-F1) on bundle.full_labels.

Output: results/04_metrics/_ablation_zscore_agg_compare.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Production scripts live at V13/scripts/; this file is at V13/ablations/scripts/.
_PROD_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.append(str(_PROD_SCRIPTS))

from artifact_paths import RESULTS_ROOT
from score_utils import (
    compute_train_baseline_stats,
    compute_tsb_metrics,
    compute_tsb_sliding_window,
    load_per_channel_scores,
    prepare_dataset_bundle,
)


METRICS_DIR = RESULTS_ROOT / "04_metrics"
EPS = 1e-8

AGG_MODES = ["max", "median", "mean"]
TSB_KEYS = ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC", "Standard-F1", "PA-F1")


def _edge_fill(score: np.ndarray) -> np.ndarray:
    """TSB-AD-style edge-fill (forward-fill internal NaNs, edge-pad both ends)."""
    out = np.asarray(score, dtype=np.float64).copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float64)
    valid_idx = np.flatnonzero(valid)
    first, last = int(valid_idx[0]), int(valid_idx[-1])
    out[:first] = out[first]
    out[last + 1:] = out[last]
    last_val = out[first]
    for i in range(first, last + 1):
        if np.isfinite(out[i]):
            last_val = out[i]
        else:
            out[i] = last_val
    return out


def _aggregate(z: np.ndarray, mode: str) -> np.ndarray:
    """Per-row channel aggregation. z: (T_eval, C')."""
    if mode == "max":
        return np.nanmax(z, axis=1)
    if mode == "median":
        return np.nanmedian(z, axis=1)
    if mode == "mean":
        return np.nanmean(z, axis=1)
    raise ValueError(f"unknown agg mode: {mode}")


def process_one(dataset_id: str) -> dict:
    rec: dict = {"dataset_id": dataset_id, "status": "ok", "error": ""}
    try:
        bundle = prepare_dataset_bundle(dataset_id)
        ds_dir = RESULTS_ROOT / dataset_id

        # Load full-series per-channel D_w_c (already includes train + test
        # regions in global indexing).
        per_ch = load_per_channel_scores(ds_dir)
        Dwc_full = per_ch["D_w_c"]
        ts_global = per_ch["t"].astype(np.int64)

        # Compute train-only baseline (production V13/V14 choice) from
        # predictions_train. Train is preferred — V14 200/200 datasets
        # use train baseline.
        train_pred_path = ds_dir / "predictions_train.npy"
        if not train_pred_path.exists():
            rec["status"] = "skip"
            rec["error"] = "no_predictions_train"
            return rec
        preds_train = np.load(train_pred_path)
        baseline = compute_train_baseline_stats(
            predictions_train=preds_train,
            train_values_norm=bundle.train_values_norm,
        )
        mu = baseline["mean"]
        sigma = baseline["std"]
        valid_c = np.isfinite(sigma) & (sigma > EPS) & np.isfinite(mu)
        if not valid_c.any():
            rec["status"] = "skip"
            rec["error"] = "no_valid_channels"
            return rec

        # z-score per-channel
        z = (Dwc_full[:, valid_c] - mu[valid_c]) / sigma[valid_c]   # (T_eval, C')

        # TSB-AD slidingWindow (raw full series, first channel ACF rank-1)
        sw = compute_tsb_sliding_window(bundle.full_values_raw)
        full_len = int(bundle.total_timesteps)
        label = bundle.full_labels.astype(np.int64)

        rec["n_train_baseline_rows"] = int(baseline.get("n", 0))
        rec["n_channels_used"] = int(valid_c.sum())
        rec["sliding_window"] = int(sw)

        for mode in AGG_MODES:
            agg = _aggregate(z, mode)
            score_full = np.full(full_len, np.nan, dtype=np.float64)
            score_full[ts_global] = agg
            score_full = _edge_fill(score_full)
            m = compute_tsb_metrics(score_full, label, sw)
            for k in TSB_KEYS:
                rec[f"{k.replace('-','_')}_{mode}"] = float(m.get(k, float("nan")))
            rec[f"err_{mode}"] = m.get("error", "") or ""
        return rec

    except Exception as exc:
        rec["status"] = "fail"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


# CSV columns: dataset_id + per-mode metric columns + status
def _columns() -> list[str]:
    cols = ["dataset_id", "n_train_baseline_rows", "n_channels_used", "sliding_window"]
    for k in TSB_KEYS:
        for mode in AGG_MODES:
            cols.append(f"{k.replace('-','_')}_{mode}")
    for mode in AGG_MODES:
        cols.append(f"err_{mode}")
    cols.extend(["status", "error"])
    return cols


def _process_one_pickleable(dataset_id: str) -> tuple[str, dict]:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    return dataset_id, process_one(dataset_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / "_ablation_zscore_agg_compare.csv"

    datasets = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "scores_per_ch.npz").exists():
            continue
        datasets.append(d.name)
    if args.only:
        only = set(args.only)
        datasets = [d for d in datasets if d in only]

    workers = max(1, min(int(args.workers), len(datasets))) if datasets else 1
    print(f"z_train aggregation comparison — {len(datasets)} dataset(s), workers={workers}")

    cols = _columns()
    rows: list[dict] = []
    n_ok = n_skip = n_fail = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()

        def _emit(i: int, ds: str, rec: dict) -> None:
            nonlocal n_ok, n_skip, n_fail
            for k in cols:
                rec.setdefault(k, "")
            w.writerow({k: rec.get(k, "") for k in cols})
            fh.flush()
            rows.append(rec)
            if rec["status"] == "ok":
                n_ok += 1
                msg = (f"max={rec['VUS_PR_max']:.3f} "
                       f"med={rec['VUS_PR_median']:.3f} "
                       f"mean={rec['VUS_PR_mean']:.3f}")
            elif rec["status"] == "skip":
                n_skip += 1
                msg = f"SKIP ({rec['error']})"
            else:
                n_fail += 1
                msg = f"FAIL ({rec['error']})"
            print(f"  [{i:>3}/{len(datasets)}] {ds:<24} {msg}")

        if workers == 1:
            for i, ds in enumerate(datasets, 1):
                _emit(i, ds, process_one(ds))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_process_one_pickleable, ds): ds for ds in datasets}
                for i, fut in enumerate(as_completed(futures), 1):
                    ds, rec = fut.result()
                    _emit(i, ds, rec)

    print(f"\n  ok={n_ok}, skip={n_skip}, fail={n_fail}")
    print(f"  ✓ {out_csv}")

    # Summary
    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    print(f"\n=== Aggregation comparison (n={len(df)} datasets) ===\n")
    print(f"{'Metric':<14} | {'max mean':>9} | {'median mean':>11} | {'mean mean':>10} | "
          f"{'Δ med-max':>10} | {'Δ mean-max':>10} | {'p(med vs max)':>14} | {'p(mean vs max)':>14}")
    print("-" * 120)
    for k in TSB_KEYS:
        mx = pd.to_numeric(df[f"{k.replace('-','_')}_max"], errors="coerce").to_numpy(np.float64)
        md = pd.to_numeric(df[f"{k.replace('-','_')}_median"], errors="coerce").to_numpy(np.float64)
        mn = pd.to_numeric(df[f"{k.replace('-','_')}_mean"], errors="coerce").to_numpy(np.float64)
        fin = np.isfinite(mx) & np.isfinite(md) & np.isfinite(mn)
        mx, md, mn = mx[fin], md[fin], mn[fin]
        try:
            _, p_md = wilcoxon(md - mx, alternative="two-sided", zero_method="wilcox")
        except Exception:
            p_md = float("nan")
        try:
            _, p_mn = wilcoxon(mn - mx, alternative="two-sided", zero_method="wilcox")
        except Exception:
            p_mn = float("nan")
        print(f"{k:<14} | {mx.mean():>9.4f} | {md.mean():>11.4f} | {mn.mean():>10.4f} | "
              f"{(md-mx).mean():>+10.4f} | {(mn-mx).mean():>+10.4f} | "
              f"{p_md:>14.4f} | {p_mn:>14.4f}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
