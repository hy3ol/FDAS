"""
06_cross_dataset.py — V14 cross-dataset analysis (full-series, GT-free).

Reads results/04_metrics/per_dataset_metrics.csv and computes:
  - Main: Spearman ρ + Pearson r between (Normal MSE, <metric>(D_w_z))
          (using log10 MSE for Pearson per v7.0 §5.1)
  - Sub : Paired comparison D_w_z (production, z_train_max) vs D_w (raw_max)
          - Wilcoxon signed-rank test on (D_w_z − D_w)
          - Win count / mean Δ

V14 schema: the GT-using `*_base` columns were removed. The legacy "D_w vs
base" comparison is repurposed to "D_w_z vs D_w" — i.e. production
(z_train_max) vs raw_max baseline, which is the headline finding of
V14_RESULTS_REPORT §1.

Output: results/05_cross_dataset/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.path.append(str(Path(__file__).resolve().parent))

from artifact_paths import RESULTS_ROOT


METRICS_DIR = RESULTS_ROOT / "04_metrics"
OUT_DIR_BASE = RESULTS_ROOT / "05_cross_dataset"


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# (production_col, baseline_col, label).
# Production = D_w_z (z_train_max), baseline = D_w (raw_max).
METRIC_COLS = {
    "auroc": ("auroc_D_w_z", "auroc_D_w", "AUROC"),
    "vus_pr": ("vus_pr_D_w_z", "vus_pr_D_w", "VUS-PR"),
    "vus_roc": ("vus_roc_D_w_z", "vus_roc_D_w", "VUS-ROC"),
    "auc_pr": ("auc_pr_D_w_z", "auc_pr_D_w", "AUC-PR"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", type=str, default="iTransformer",
        help="Backbone whose per_dataset_metrics CSV to read. Default: iTransformer.",
    )
    parser.add_argument(
        "--min-eval-anomalies", type=int, default=0,
        help="Drop datasets whose n_eval_anomalies < this threshold "
             "(uses per_dataset_metrics.csv column). Default 0 = include all.",
    )
    parser.add_argument(
        "--metric", choices=list(METRIC_COLS.keys()), default="auroc",
        help="Detection metric to correlate with Normal MSE. Default: auroc.",
    )
    args = parser.parse_args()
    metric_prod, metric_base, metric_label = METRIC_COLS[args.metric]

    metrics_path = METRICS_DIR / f"per_dataset_metrics__{args.backbone}.csv"
    out_dir = OUT_DIR_BASE / args.backbone

    if not metrics_path.exists():
        sys.exit(f"Missing {metrics_path}. Run `05_metrics.py --backbone {args.backbone}` first.")
    df = pd.read_csv(metrics_path)
    df = df[df["status"] == "ok"].copy()

    if metric_prod not in df.columns or metric_base not in df.columns:
        sys.exit(
            f"Metric '{args.metric}' not found in {metrics_path}. "
            f"Re-run 05_metrics.py without --skip-vus."
        )

    if args.min_eval_anomalies > 0 and "n_eval_anomalies" in df.columns:
        before = len(df)
        df["n_eval_anomalies"] = pd.to_numeric(df["n_eval_anomalies"], errors="coerce").fillna(0)
        df = df[df["n_eval_anomalies"] >= args.min_eval_anomalies].copy()
        print(f"  applied --min-eval-anomalies={args.min_eval_anomalies}: "
              f"{before} → {len(df)} dataset(s)")

    for col in ("normal_mse", metric_prod, metric_base):
        df[col] = df[col].apply(_to_float)

    main_mask = (
        np.isfinite(df["normal_mse"])
        & np.isfinite(df[metric_prod])
        & (df["normal_mse"] > 0)
    )
    main_df = df[main_mask]
    n_main = len(main_df)

    main_corr = {}
    if n_main >= 3:
        rho, p_rho = spearmanr(main_df["normal_mse"], main_df[metric_prod])
        log_mse = np.log10(main_df["normal_mse"].to_numpy())
        r, p_r = pearsonr(log_mse, main_df[metric_prod].to_numpy())
        main_corr = {
            "metric": args.metric,
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r_log_mse": float(r),
            "pearson_p_log_mse": float(p_r),
            "n_datasets": int(n_main),
        }
    else:
        main_corr = {
            "metric": args.metric,
            "error": f"too few datasets for correlation: n={n_main}",
            "n_datasets": int(n_main),
        }

    pair_mask = np.isfinite(df[metric_prod]) & np.isfinite(df[metric_base])
    pair_df = df[pair_mask]
    n_pair = len(pair_df)
    baseline_cmp: dict = {"metric": args.metric, "n_datasets": int(n_pair)}
    if n_pair >= 1:
        diff = pair_df[metric_prod].to_numpy() - pair_df[metric_base].to_numpy()
        n_prod_wins = int(np.sum(diff > 0))
        baseline_cmp.update({
            "datasets_D_w_z_wins": f"{n_prod_wins}/{n_pair}",
            "n_dwz_wins": n_prod_wins,
            "mean_delta": float(np.mean(diff)),
            "median_delta": float(np.median(diff)),
        })
        if n_pair >= 2 and not np.allclose(diff, 0):
            try:
                W, p = wilcoxon(pair_df[metric_prod], pair_df[metric_base],
                                zero_method="wilcox", alternative="two-sided")
                baseline_cmp.update({"wilcoxon_W": float(W), "wilcoxon_p": float(p)})
            except ValueError as exc:
                baseline_cmp["wilcoxon_error"] = str(exc)

    by_family: list[dict] = []
    if "family" in df.columns:
        # Coerce err columns once for within-family Spearman
        for c in ("normal_mse", "normal_mae", "normal_mse_strict", "normal_mae_strict"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Choose the same X column convention as the main analysis (MSE column)
        for fam, g in df.groupby("family"):
            valid = g[np.isfinite(g[metric_prod])]
            if valid.empty:
                continue
            entry = {
                "family": fam,
                "n": int(valid.shape[0]),
                f"mean_{metric_prod}": float(valid[metric_prod].mean()),
                f"median_{metric_prod}": float(valid[metric_prod].median()),
                f"mean_{metric_base}": float(valid[metric_base].mean()),
                "mean_delta": float((valid[metric_prod] - valid[metric_base]).mean()),
            }
            # Within-family Spearman ρ vs Normal MSE / MAE (target + strict)
            for err_col in ("normal_mse", "normal_mae",
                            "normal_mse_strict", "normal_mae_strict"):
                if err_col not in valid.columns:
                    continue
                v = valid[[err_col, metric_prod]].dropna()
                if len(v) >= 3:
                    rho_f, p_f = spearmanr(v[err_col], v[metric_prod])
                    entry[f"spearman_{err_col}"] = float(rho_f)
                    entry[f"p_{err_col}"] = float(p_f)
                    entry[f"n_{err_col}"] = int(len(v))
            by_family.append(entry)
        # Aggregate within-family ρ via Fisher z-transform (mean of z, weighted by n−3)
        for err_col in ("normal_mse", "normal_mae",
                        "normal_mse_strict", "normal_mae_strict"):
            rhos, weights = [], []
            for e in by_family:
                key = f"spearman_{err_col}"
                if key in e and e.get(f"n_{err_col}", 0) >= 4:
                    r = float(e[key])
                    if abs(r) < 1.0:
                        rhos.append(r)
                        weights.append(max(e[f"n_{err_col}"] - 3, 1))
            if rhos:
                z = np.arctanh(np.array(rhos))
                w = np.array(weights, dtype=float)
                z_avg = float(np.average(z, weights=w))
                main_corr[f"within_family_spearman_{err_col}"] = float(np.tanh(z_avg))
                main_corr[f"within_family_n_families_{err_col}"] = int(len(rhos))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "results.json" if args.metric == "auroc" else f"results_{args.metric}.json"
    out_path = out_dir / out_name
    payload = {
        "backbone": args.backbone,
        "metric": args.metric,
        "metric_label": metric_label,
        "min_eval_anomalies_cutoff": int(args.min_eval_anomalies),
        "main_correlation": main_corr,
        "baseline_comparison": baseline_cmp,
        "by_family": by_family,
        "n_total_ok_datasets": int(len(df)),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"06_cross_dataset (metric={args.metric}):")
    print(f"  ✓ Wrote {out_path}")
    if "spearman_rho" in main_corr:
        print(f"  Spearman ρ = {main_corr['spearman_rho']:+.3f} "
              f"(p={main_corr['spearman_p']:.3g}, n={main_corr['n_datasets']})")
        print(f"  Pearson r (log MSE) = {main_corr['pearson_r_log_mse']:+.3f} "
              f"(p={main_corr['pearson_p_log_mse']:.3g})")
    else:
        print(f"  Main correlation: {main_corr}")
    if "n_dwz_wins" in baseline_cmp:
        print(f"  D_w_z wins (vs D_w): {baseline_cmp['datasets_D_w_z_wins']}, "
              f"mean Δ{metric_label} = {baseline_cmp['mean_delta']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
