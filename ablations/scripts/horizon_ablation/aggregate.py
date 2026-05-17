"""Aggregate the horizon ablation results (multi-backbone).

Reads:
  - ablations/results/horizon/per_dataset_metrics.csv  (all backbones, all H ablation rows)
  - results/04_metrics/<backbone>/per_dataset_metrics.csv  (production H=96, per backbone)

Writes:
  - ablations/results/horizon/summary.csv
        long-format: (backbone × dataset_key × H) × 6 TSB-AD-M metrics + family + status
  - ablations/results/horizon/summary_aggregate.csv
        per-(backbone, H) mean/median + paired Wilcoxon vs H=96 (within-backbone)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

PROD_METRICS_ROOT = _REPO_ROOT / "results" / "04_metrics"
ABL_METRICS = _REPO_ROOT / "ablations" / "results" / "horizon" / "per_dataset_metrics.csv"
OUT_DIR = _REPO_ROOT / "ablations" / "results" / "horizon"

METRIC_KEYS = ["vus_pr", "vus_roc", "auc_pr", "auc_roc", "standard_f1", "pa_f1"]


def _load_production_h96(backbone: str) -> pd.DataFrame:
    """Pull production H=96 for a given backbone (if available)."""
    path = PROD_METRICS_ROOT / backbone / "per_dataset_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "pred_len":     96,
        "backbone":     backbone,
        "dataset_key":  df.get("dataset_id", df.get("dataset_key", pd.Series())),
        "family":       df.get("family", ""),
        "vus_pr":       df.get("vus_pr_D_w_z", np.nan),
        "vus_roc":      df.get("vus_roc_D_w_z", np.nan),
        "auc_pr":       df.get("auc_pr_D_w_z", np.nan),
        "auc_roc":      df.get("auroc_D_w_z", np.nan),
        "standard_f1":  df.get("standard_f1_D_w_z", np.nan),
        "pa_f1":        df.get("pa_f1_D_w_z", np.nan),
        "status":       df.get("status", "ok"),
    })
    return out


def _load_ablation() -> pd.DataFrame:
    if not ABL_METRICS.exists():
        return pd.DataFrame(columns=["pred_len", "backbone", "dataset_key", "family",
                                     *METRIC_KEYS, "status"])
    df = pd.read_csv(ABL_METRICS)
    if "backbone" not in df.columns:
        df["backbone"] = "iTransformer"
    df["backbone"] = df["backbone"].fillna("iTransformer")
    keep = ["pred_len", "backbone", "dataset_key", "family",
            "vus_pr", "vus_roc", "auc_pr", "auc_roc",
            "standard_f1", "pa_f1", "status"]
    return df[keep]


def build_summary() -> pd.DataFrame:
    parts = [_load_ablation()]
    # For every backbone seen in ablation OR present in production, pull H=96
    abl = parts[0]
    backbones_in_abl = set(abl["backbone"].dropna().unique().tolist()) if not abl.empty else set()
    backbones_in_prod = {p.parent.name for p in PROD_METRICS_ROOT.glob("*/per_dataset_metrics.csv")}
    for bb in sorted(backbones_in_abl | backbones_in_prod):
        # Only pull production H=96 for backbones that don't already have H=96 in ablation
        if not abl.empty and ((abl["backbone"] == bb) & (abl["pred_len"] == 96)).any():
            continue
        prod = _load_production_h96(bb)
        if not prod.empty:
            parts.append(prod)

    summary = pd.concat(parts, ignore_index=True)
    summary = summary.sort_values(
        ["backbone", "dataset_key", "pred_len"]
    ).reset_index(drop=True)
    return summary


def build_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-(backbone, H) mean/median + paired Wilcoxon vs H=96 within-backbone."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows: list[dict] = []
    for bb in sorted(summary["backbone"].dropna().unique().tolist()):
        sub_bb = summary[summary["backbone"] == bb].copy()
        horizons = sorted(sub_bb["pred_len"].dropna().unique().astype(int).tolist())
        h96 = sub_bb[sub_bb["pred_len"] == 96].set_index("dataset_key")
        for H in horizons:
            sub = sub_bb[(sub_bb["pred_len"] == H) & (sub_bb["status"] == "ok")].copy()
            n_ok = len(sub)
            row: dict = {"backbone": bb, "pred_len": int(H), "n_ok": int(n_ok)}
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
    cols_preview = ["backbone", "pred_len", "n_ok",
                    "vus_pr_mean", "vus_roc_mean", "vus_pr_delta_vs96_mean",
                    "vus_pr_wilcoxon_p"]
    avail = [c for c in cols_preview if c in agg.columns]
    print(agg[avail].to_string(index=False))


if __name__ == "__main__":
    main()
