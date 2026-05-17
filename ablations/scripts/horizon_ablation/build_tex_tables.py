"""Generate LaTeX tables for the horizon ablation, per backbone.

Mirrors the style of results/00_result_table/table_{1,2}/vuspr_table_*.tex
(standalone document, booktabs, micro-averaged across ok datasets).

Default backbone is iTransformer. For TTM (or any other backbone) the
production H=96 row is pulled from results/04_metrics/<backbone>/per_dataset_metrics.csv
when present, so the table can compare against the paper-grade headline.

Output filename pattern:
  iTransformer  → horizon_table_1.tex / horizon_table_2.tex     (legacy, unchanged)
  Other         → horizon_table_1__<backbone>.tex / horizon_table_2__<backbone>.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
SUMMARY = REPO / "ablations" / "results" / "horizon" / "summary.csv"
PROD_METRICS_ROOT = REPO / "results" / "04_metrics"
OUT_DIR = REPO / "ablations" / "results" / "table"

METRICS = ["auc_pr", "auc_roc", "vus_pr", "vus_roc", "standard_f1", "pa_f1"]
METRIC_LABELS = {
    "auc_pr": "AUC-PR",
    "auc_roc": "AUC-ROC",
    "vus_pr": "VUS-PR",
    "vus_roc": "VUS-ROC",
    "standard_f1": "Standard-F1",
    "pa_f1": "PA-F1",
}
ALL_HORIZONS = [12, 24, 48, 96, 192, 336]
# Display names for the tex table titles. Map BACKBONES key → paper-name.
BACKBONE_DISPLAY = {
    "iTransformer": "iTransformer",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "TimeMixer": "TimeMixer",
    "TimesNet": "TimesNet",
    "TimeXer": "TimeXer",
    "TimesFM": "TimesFM",
    "TTM": "TTM-r2",
    "Moirai": "Moirai-1.1-R",
}
FAMILY_LABELS = {
    "CATSv2": "CATSv2", "CreditCard": "CreditCard", "Daphnet": "Daphnet",
    "Exathlon": "Exathlon", "GECCO": "GECCO", "GHL": "GHL", "Genesis": "Genesis",
    "LTDB": "LTDB", "MITDB": "MITDB", "MSL": "MSL", "OPPORTUNITY": "OPP",
    "PSM": "PSM", "SMAP": "SMAP", "SMD": "SMD", "SVDB": "SVDB",
    "SWaT": "SWaT", "TAO": "TAO",
}


def _load_production_h96(backbone: str) -> pd.DataFrame:
    """Pull production H=96 rows from results/04_metrics/<backbone>/per_dataset_metrics.csv
    and reshape to the summary.csv schema. Returns empty DF if file missing."""
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
        "standard_f1": df.get("standard_f1_D_w_z", np.nan),
        "pa_f1":        df.get("pa_f1_D_w_z", np.nan),
        "status":       df.get("status", "ok"),
    })
    return out


def _load_df_for_backbone(backbone: str) -> tuple[pd.DataFrame, list[int]]:
    """Combine ablation summary.csv (per backbone) with production H=96 (per backbone).
    Returns (combined_df, horizons_present)."""
    summary = pd.read_csv(SUMMARY)
    # Some legacy summary rows have empty backbone — assume iTransformer.
    if "backbone" not in summary.columns:
        summary["backbone"] = "iTransformer"
    summary["backbone"] = summary["backbone"].fillna("iTransformer")
    sub = summary[summary["backbone"] == backbone].copy()

    # Pull production H=96 if not already in summary.csv for this backbone.
    if 96 not in sub["pred_len"].astype(int).tolist():
        prod = _load_production_h96(backbone)
        if not prod.empty:
            sub = pd.concat([sub, prod], ignore_index=True)

    horizons_present = sorted(
        [int(h) for h in sub["pred_len"].dropna().unique() if int(h) in ALL_HORIZONS]
    )
    return sub, horizons_present


def _p_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "$^{\\ast\\ast\\ast}$"
    if p < 1e-3:
        return "$^{\\ast\\ast}$"
    if p < 5e-2:
        return "$^{\\ast}$"
    return ""


def _compute_h_summary(df: pd.DataFrame, horizons: list[int]) -> dict:
    out: dict = {}
    if 96 in horizons:
        h96 = df[df["pred_len"] == 96].set_index("dataset_key")
    else:
        h96 = pd.DataFrame()
    for H in horizons:
        sub = df[(df["pred_len"] == H) & (df["status"] == "ok")].copy()
        row: dict = {"n_ok": int(len(sub))}
        for m in METRICS:
            v = pd.to_numeric(sub[m], errors="coerce").dropna()
            row[m] = float(v.mean()) if v.size else float("nan")
            if H == 96 or h96.empty:
                row[m + "_delta"] = None
                row[m + "_p"] = None
            else:
                a = pd.to_numeric(sub.set_index("dataset_key")[m], errors="coerce")
                b = pd.to_numeric(h96[m], errors="coerce")
                common = a.index.intersection(b.index)
                a = a.loc[common].to_numpy()
                b = b.loc[common].to_numpy()
                mask = ~(np.isnan(a) | np.isnan(b))
                a, b = a[mask], b[mask]
                if a.size == 0 or not (a - b).any():
                    row[m + "_delta"] = float("nan")
                    row[m + "_p"] = float("nan")
                else:
                    _, p = wilcoxon(a, b, zero_method="wilcox")
                    row[m + "_delta"] = float((a - b).mean())
                    row[m + "_p"] = float(p)
        out[H] = row
    return out


def build_table_1(df: pd.DataFrame, horizons: list[int], backbone: str) -> str:
    summary = _compute_h_summary(df, horizons)
    display = BACKBONE_DISPLAY.get(backbone, backbone)

    best_col: dict = {}
    for m in METRICS:
        vals = [summary[H][m] for H in horizons if not np.isnan(summary[H][m])]
        best_col[m] = max(vals) if vals else float("nan")

    lines: list[str] = []
    lines.append(r"\documentclass[border=2pt,varwidth=\maxdimen]{standalone}")
    lines.append("")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{colortbl}")
    lines.append(r"\arrayrulecolor{black}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(
        r"\textbf{\large Horizon Ablation \textemdash\ "
        + display + r"} \quad (overall metrics, $L=192$)\par\vspace{6pt}"
    )
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\small")
    lines.append("")
    lines.append(r"\begin{tabular}{l|c|cccccc}")
    lines.append(r"\toprule")
    header = (
        r"\textbf{Horizon $H$} & \textbf{$n_\textrm{ok}$} & "
        + " & ".join(f"\\textbf{{{METRIC_LABELS[m]}}}" for m in METRICS)
        + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    for H in horizons:
        r = summary[H]
        label = "$H=96$ (production)" if H == 96 else f"$H={H}$"
        cells = []
        for m in METRICS:
            v = r[m]
            s = f"{v:.4f}"
            if not np.isnan(v) and v >= best_col[m] - 1e-12:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(
            f"\\textbf{{{label}}}  & {r['n_ok']:>3} & " + " & ".join(cells) + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def build_table_2(df: pd.DataFrame, horizons: list[int], backbone: str) -> str:
    display = BACKBONE_DISPLAY.get(backbone, backbone)
    fam = df[df["status"] == "ok"][["family", "pred_len", "vus_pr"]].copy()
    fam["vus_pr"] = pd.to_numeric(fam["vus_pr"], errors="coerce")
    piv = fam.pivot_table(
        index="family", columns="pred_len", values="vus_pr", aggfunc="mean"
    )
    fam_order = sorted(piv.index.tolist())

    best_in_col = piv.max(axis=1).to_dict()

    lines: list[str] = []
    lines.append(r"\documentclass[border=2pt,varwidth=\maxdimen]{standalone}")
    lines.append("")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{colortbl}")
    lines.append(r"\arrayrulecolor{black}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(
        r"\textbf{\large Horizon Ablation \textemdash\ "
        + display + r"} \quad (per-family VUS-PR, $L=192$)\par\vspace{6pt}"
    )
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\footnotesize")
    lines.append("")
    n_fam = len(fam_order)
    col_spec = "l|" + "c" * n_fam
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    head = (
        r"\textbf{Horizon $H$} & "
        + " & ".join(f"\\textbf{{{FAMILY_LABELS[f]}}}" for f in fam_order)
        + r" \\"
    )
    lines.append(head)
    lines.append(r"\midrule")

    for H in horizons:
        if H not in piv.columns:
            continue
        cells = []
        for f in fam_order:
            v = piv.at[f, H] if H in piv.columns else float("nan")
            if pd.isna(v):
                cells.append("--")
            else:
                s = f"{v:.3f}"
                if v >= best_in_col[f] - 1e-12:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
        lines.append(f"\\textbf{{$H={H}$}} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="iTransformer",
                        help="Backbone whose tables to build. Default: iTransformer.")
    args = parser.parse_args()
    backbone = args.backbone

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, horizons = _load_df_for_backbone(backbone)
    if not horizons:
        print(f"[warn] no rows found for backbone={backbone}")
        return

    # Every backbone gets a __<backbone> suffix for unambiguous naming.
    suffix = f"__{backbone}"
    t1 = OUT_DIR / f"horizon_table_1{suffix}.tex"
    t2 = OUT_DIR / f"horizon_table_2{suffix}.tex"
    t1.write_text(build_table_1(df, horizons, backbone))
    t2.write_text(build_table_2(df, horizons, backbone))
    print(f"Wrote {t1}  (H={horizons})")
    print(f"Wrote {t2}")


if __name__ == "__main__":
    main()
