"""Aggregate the horizon ablation results.

Reads:
  - results/04_metrics/iTransformer/per_dataset_metrics.csv   (production H=96)
  - ablations/results/horizon/per_dataset_metrics.csv         (ablation H=192, 336)

Writes:
  - ablations/results/horizon/summary.csv
        long-format: (dataset_key × H) × 6 TSB-AD-M metrics + family + status
  - ablations/results/horizon/summary_aggregate.csv
        per-H mean / median + paired Wilcoxon signed-rank vs H=96 (where both
        runs succeeded). Headline numbers for the ablation paragraph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

PROD_METRICS = _REPO_ROOT / "results" / "04_metrics" / "iTransformer" / "per_dataset_metrics.csv"
ABL_METRICS = _REPO_ROOT / "ablations" / "results" / "horizon" / "per_dataset_metrics.csv"
OUT_DIR = _REPO_ROOT / "ablations" / "results" / "horizon"

METRIC_KEYS = ["vus_pr", "vus_roc", "auc_pr", "auc_roc", "standard_f1", "pa_f1"]


def _load_h96() -> pd.DataFrame:
    if not PROD_METRICS.exists():
        return pd.DataFrame(columns=["dataset_key", "pred_len", "family",
                                     *METRIC_KEYS, "status"])
    df = pd.read_csv(PROD_METRICS)
    out = pd.DataFrame({
        "dataset_key":  df["dataset_id"],
        "pred_len":     96,
        "family":       df["family"],
        "vus_pr":       df["vus_pr_D_w_z"],
        "vus_roc":      df["vus_roc_D_w_z"],
        "auc_pr":       df["auc_pr_D_w_z"],
        "auc_roc":      df["auroc_D_w_z"],
        "standard_f1":  df["standard_f1_D_w_z"],
        "pa_f1":        df["pa_f1_D_w_z"],
        "status":       df["status"],
    })
    return out


def _load_ablation() -> pd.DataFrame:
    if not ABL_METRICS.exists():
        return pd.DataFrame(columns=["dataset_key", "pred_len", "family",
                                     *METRIC_KEYS, "status"])
    df = pd.read_csv(ABL_METRICS)
    keep = ["pred_len", "dataset_key", "family",
            "vus_pr", "vus_roc", "auc_pr", "auc_roc",
            "standard_f1", "pa_f1", "status"]
    return df[keep]


def build_summary() -> pd.DataFrame:
    parts = [_load_h96(), _load_ablation()]
    summary = pd.concat(parts, ignore_index=True)
    summary = summary.sort_values(["dataset_key", "pred_len"]).reset_index(drop=True)
    return summary


def build_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-H mean/median + paired Wilcoxon vs H=96.

    Paired test uses only datasets where BOTH H and H=96 produced status='ok'.
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows: list[dict] = []
    horizons = sorted(summary["pred_len"].dropna().unique().astype(int).tolist())
    h96 = summary[summary["pred_len"] == 96].set_index("dataset_key")

    for H in horizons:
        sub = summary[(summary["pred_len"] == H) & (summary["status"] == "ok")].copy()
        n_ok = len(sub)
        row: dict = {"pred_len": int(H), "n_ok": int(n_ok)}
        for m in METRIC_KEYS:
            arr = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy()
            row[f"{m}_mean"] = float(np.mean(arr)) if arr.size else np.nan
            row[f"{m}_median"] = float(np.median(arr)) if arr.size else np.nan
        if H != 96 and not h96.empty:
            for m in METRIC_KEYS:
                paired = sub.set_index("dataset_key").join(
                    h96[[m]], how="inner", lsuffix=f"_{H}", rsuffix="_96"
                )
                a = pd.to_numeric(paired[f"{m}_{H}"], errors="coerce")
                b = pd.to_numeric(paired[f"{m}_96"], errors="coerce")
                mask = a.notna() & b.notna()
                a, b = a[mask].to_numpy(), b[mask].to_numpy()
                row[f"{m}_delta_vs96_mean"] = (
                    float(np.mean(a - b)) if a.size else np.nan
                )
                if wilcoxon is not None and a.size > 0 and not np.allclose(a, b):
                    try:
                        _, p = wilcoxon(a, b, zero_method="wilcox")
                        row[f"{m}_wilcoxon_p"] = float(p)
                    except Exception:
                        row[f"{m}_wilcoxon_p"] = np.nan
                else:
                    row[f"{m}_wilcoxon_p"] = np.nan
                row[f"{m}_n_paired"] = int(a.size)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'summary.csv'}  ({len(summary)} rows)")
    agg = build_aggregate(summary)
    agg.to_csv(OUT_DIR / "summary_aggregate.csv", index=False)
    print(f"Wrote {OUT_DIR / 'summary_aggregate.csv'}  ({len(agg)} rows)")
    print("\nPreview:")
    cols_preview = ["pred_len", "n_ok",
                    "vus_pr_mean", "vus_roc_mean", "vus_pr_delta_vs96_mean",
                    "vus_pr_wilcoxon_p"]
    avail = [c for c in cols_preview if c in agg.columns]
    print(agg[avail].to_string(index=False))


if __name__ == "__main__":
    main()
