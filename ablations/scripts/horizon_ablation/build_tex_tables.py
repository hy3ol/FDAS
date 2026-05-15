"""Generate LaTeX tables for the horizon ablation.

Mirrors the style of results/00_result_table/table_{1,2}/vuspr_table_*.tex
(standalone document, booktabs, micro-averaged across ok datasets).

Outputs into ablations/results/table/:
  - horizon_table_1.tex   — H × 6 TSB-AD-M metrics + Δ vs H=96 + Wilcoxon p
  - horizon_table_2.tex   — H × per-family VUS-PR breakdown (17 families)

After generation, pdflatex + pdftoppm produces .pdf and .png alongside.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
SUMMARY = REPO / "ablations" / "results" / "horizon" / "summary.csv"
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
HORIZONS = [12, 24, 48, 96, 192, 336]
# TSB-AD-M family display names; "OPP" abbreviation matches production table_2.
FAMILY_LABELS = {
    "CATSv2": "CATSv2", "CreditCard": "CreditCard", "Daphnet": "Daphnet",
    "Exathlon": "Exathlon", "GECCO": "GECCO", "GHL": "GHL", "Genesis": "Genesis",
    "LTDB": "LTDB", "MITDB": "MITDB", "MSL": "MSL", "OPPORTUNITY": "OPP",
    "PSM": "PSM", "SMAP": "SMAP", "SMD": "SMD", "SVDB": "SVDB",
    "SWaT": "SWaT", "TAO": "TAO",
}


def _p_stars(p: float) -> str:
    """Wilcoxon p → significance stars (matches common ML paper convention)."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "$^{\\ast\\ast\\ast}$"
    if p < 1e-3:
        return "$^{\\ast\\ast}$"
    if p < 5e-2:
        return "$^{\\ast}$"
    return ""


def _compute_h_summary(df: pd.DataFrame) -> dict:
    """Returns {H: {metric: mean, metric_delta: float, metric_p: float, n_ok: int}}."""
    out: dict = {}
    h96 = df[df["pred_len"] == 96].set_index("dataset_key")
    for H in HORIZONS:
        sub = df[(df["pred_len"] == H) & (df["status"] == "ok")].copy()
        row: dict = {"n_ok": int(len(sub))}
        for m in METRICS:
            v = pd.to_numeric(sub[m], errors="coerce").dropna()
            row[m] = float(v.mean()) if v.size else float("nan")
            if H == 96:
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


def build_table_1(df: pd.DataFrame) -> str:
    summary = _compute_h_summary(df)

    # Find best (max) per column among the three H rows for bold marking.
    best_col: dict = {}
    for m in METRICS:
        vals = [summary[H][m] for H in HORIZONS]
        best_col[m] = max(vals)

    lines: list[str] = []
    lines.append(r"\documentclass[border=2pt]{standalone}")
    lines.append("")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")
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

    # One row per H — production H=96 first, then H=192, H=336.
    for H in HORIZONS:
        r = summary[H]
        label = "$H=96$ (production)" if H == 96 else f"$H={H}$"
        cells = []
        for m in METRICS:
            v = r[m]
            s = f"{v:.4f}"
            if v >= best_col[m] - 1e-12:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(
            f"\\textbf{{{label}}}  & {r['n_ok']:>3} & " + " & ".join(cells) + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def build_table_2(df: pd.DataFrame) -> str:
    fam = df[df["status"] == "ok"][["family", "pred_len", "vus_pr"]].copy()
    fam["vus_pr"] = pd.to_numeric(fam["vus_pr"], errors="coerce")
    piv = fam.pivot_table(
        index="family", columns="pred_len", values="vus_pr", aggfunc="mean"
    )
    # Order families to match production table_2 (alphabetical)
    fam_order = sorted(piv.index.tolist())

    # n per (family, H) for footer / decision making
    n_by_fam_H = (
        fam.groupby(["family", "pred_len"]).size().unstack(fill_value=0)
    )

    # Per-column (family) best across H for bold marking
    best_in_col = piv.max(axis=1).to_dict()

    lines: list[str] = []
    lines.append(r"\documentclass[border=2pt]{standalone}")
    lines.append("")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")
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

    for H in HORIZONS:
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
    lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    (OUT_DIR / "horizon_table_1.tex").write_text(build_table_1(df))
    (OUT_DIR / "horizon_table_2.tex").write_text(build_table_2(df))
    print(f"Wrote {OUT_DIR / 'horizon_table_1.tex'}")
    print(f"Wrote {OUT_DIR / 'horizon_table_2.tex'}")


if __name__ == "__main__":
    main()
