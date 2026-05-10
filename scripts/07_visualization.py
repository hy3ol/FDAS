"""
07_visualization.py — V12 v7.0 figure generation (3 IEEE-style figures).

Inputs:
  results/04_metrics/per_dataset_metrics.csv
  results/{dataset_id}/scores.parquet (or .csv)

Outputs:
  results/figures/figure_main_mse_auroc_scatter.{png,pdf}
  results/figures/figure_supporting_score_timeseries.{png,pdf}
  results/figures/figure_supporting_auroc_distribution.{png,pdf}
  results/figures/captions.md
  results/statistics_table.md
"""
from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator
from scipy.stats import linregress, pearsonr, spearmanr

sys.path.append(str(Path(__file__).resolve().parent))

from artifact_paths import RESULTS_ROOT, get_dataset_results_dir
from score_utils import find_anomaly_regions, load_score_frame


METRICS_DIR = RESULTS_ROOT / "04_metrics"
# FIG_DIR and STATS_PATH are now per-backbone (resolved in main()):
#   results/figures/<backbone>/...
#   results/<backbone>/statistics_table.md
FIG_DIR_BASE = RESULTS_ROOT / "figures"
# Module-level mutable refs that save_figure() and other helpers read.
# main() rebinds these to the per-backbone variants before any plotting.
FIG_DIR = FIG_DIR_BASE
STATS_PATH = RESULTS_ROOT / "statistics_table.md"
# Set by main() so figure helpers (figure_score_timeseries via _load_scores)
# resolve per-dataset score paths under the right backbone.
CURRENT_BACKBONE = "iTransformer"

# V14: 비교 대상이 GT-using `base` → production `D_w_z` (= z_train_max)
# 로 변경. 각 metric 의 두번째 컬럼은 컨벤션 상 `metric_base` 라는 이름의
# 변수에 들어가지만 실제 의미는 D_w 의 z-normalized 비교 대상.
METRIC_COLS = {
    "auroc": ("auroc_D_w", "auroc_D_w_z", "AUROC", "auroc"),
    "vus_pr": ("vus_pr_D_w", "vus_pr_D_w_z", "VUS-PR", "vus_pr"),
    "vus_roc": ("vus_roc_D_w", "vus_roc_D_w_z", "VUS-ROC", "vus_roc"),
    "auc_pr": ("auc_pr_D_w", "auc_pr_D_w_z", "AUC-PR", "auc_pr"),
}


# ──────────────────────────────────────────────────────────────
# Conference-grade style (NeurIPS / ICML / ICLR aesthetic)
# ──────────────────────────────────────────────────────────────
def apply_paper_style():
    """Modern paper-grade typography: larger fonts, cleaner grids, tighter
    spines. Calibrated for two-column conference figures viewed at 100% zoom
    in a PDF viewer."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "savefig.facecolor": "white",
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#2a2a2a",
        "axes.labelcolor": "#1a1a1a",
        "axes.titlecolor": "#0e0e0e",
        "axes.titleweight": "normal",
        "axes.titlepad": 10,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "grid.color": "#888888",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#bdbdbd",
        "legend.fancybox": False,
        "legend.borderpad": 0.6,
        "mathtext.fontset": "stix",
    })


# Backwards-compat alias — keeps any external callers working.
apply_ieee_rcparams = apply_paper_style


COLORS = {
    "D_w": "#1f4e79",         # deep teal blue (more saturated than 1f77b4)
    "base": "#c1292e",        # paper-grade crimson
    "anomaly_shade": "#9a9a9a",
    "reference": "#555555",
    "fit_line": "#b5179e",    # magenta — distinct from family colors
    "marker_edge": "#1a1a1a",
    "panel_bg": "#fafafa",    # very subtle scatter background
    "highlight": "#d62728",   # for per-channel max / annotated points
}

EPS = 1e-12

FAMILY_MARKERS = [
    "o", "s", "^", "D", "P", "X", "v", "<", ">",
    "*", "h", "p", "8", "d", "H",
]


def _make_family_color_map(families: list[str]) -> dict[str, tuple]:
    """V8-style maximin HSV color allocation — produces visually distinct
    colors for any number of families by greedy farthest-point selection
    in HSV space."""
    n = len(families)
    if n <= 0:
        return {}

    hues = np.linspace(0.0, 1.0, 72, endpoint=False)
    sats = [0.78, 0.90, 0.98]
    vals = [0.86, 0.94]

    cand = []
    for h in hues:
        for s in sats:
            for v in vals:
                cand.append(colorsys.hsv_to_rgb(float(h), float(s), float(v)))
    cand_arr = np.asarray(cand, dtype=np.float64)

    seed = np.asarray(colorsys.hsv_to_rgb(0.30, 0.98, 0.95), dtype=np.float64)
    d0 = np.sum((cand_arr - seed[None, :]) ** 2, axis=1)
    first_idx = int(np.argmax(d0))

    selected = [first_idx]
    min_d = np.sum((cand_arr - cand_arr[first_idx:first_idx + 1]) ** 2, axis=1)
    while len(selected) < n:
        idx = int(np.argmax(min_d))
        selected.append(idx)
        d_new = np.sum((cand_arr - cand_arr[idx:idx + 1]) ** 2, axis=1)
        min_d = np.minimum(min_d, d_new)

    out = {}
    for fam, idx in zip(families, selected):
        r, g, b = cand_arr[idx]
        out[fam] = (float(r), float(g), float(b), 1.0)
    return out


def get_family_style(family_list):
    """Return {family: {'color': RGBA, 'marker': mpl-marker}} using V8 theme."""
    families = sorted(set(family_list))
    color_map = _make_family_color_map(families)
    out = {}
    for i, fam in enumerate(families):
        out[fam] = {
            "color": color_map[fam],
            "marker": FAMILY_MARKERS[i % len(FAMILY_MARKERS)],
        }
    return out


def get_family_palette(family_list):
    """Backwards-compat: color-only mapping."""
    return {f: s["color"] for f, s in get_family_style(family_list).items()}


# ── Rank helpers (V8 theme) ────────────────────────────────────
def rank_equal(vals: np.ndarray) -> np.ndarray:
    """Rank values to [0, 1] with average-rank tie handling.
    Different from rankdata: scaled to unit interval so axes have a fixed range."""
    x = np.asarray(vals, dtype=np.float64)
    n = x.size
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks / max(1.0, float(n - 1))


def set_rank_ticks(ax, axis: str, raw_vals: np.ndarray) -> None:
    """V8: show 5 quantile labels (raw values) at evenly-spaced positions."""
    q = np.linspace(0.0, 1.0, 5)
    raw_q = np.quantile(raw_vals, q) if raw_vals.size else np.zeros_like(q)
    labels = [f"{v:.3g}" for v in raw_q]
    if axis == "x":
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(q)
        ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(q)
        ax.set_yticklabels(labels, fontsize=8)


def _format_stat(name: str, value: float, p_value: float, symbol: str) -> str:
    if not np.isfinite(value):
        return f"{name} {symbol}=NA"
    return f"{name} {symbol}={value:+.3f} ($p$={p_value:.2g})"


# Module-level switch — flipped by CLI flag --no-pdf in main().
SAVE_PDF: bool = True


def save_figure(fig, name, *, pdf: bool | None = None):
    """Write PNG (always) and PDF (default) for paper-grade outputs.
    When `pdf` is None, falls back to module-level SAVE_PDF (settable via
    --no-pdf CLI flag)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=300,
                bbox_inches="tight", facecolor="white")
    write_pdf = SAVE_PDF if pdf is None else pdf
    if write_pdf:
        fig.savefig(FIG_DIR / f"{name}.pdf", format="pdf",
                    bbox_inches="tight", facecolor="white")


def _style_log_xaxis(ax, numticks: int = 6, data: np.ndarray | None = None) -> None:
    """Unified log x-axis style for every scatter in this script.

    Adaptive: if the data range spans ≥ 1.5 decades, use integer-power-of-10
    major ticks (10^x labels). If narrower (e.g., a family clustered between
    0.3 and 0.7), include sub-decade ticks at {1, 2, 5} × 10^k so the axis
    still has labeled ticks.

    - capped at `numticks` so labels never overlap,
    - minor tick labels suppressed,
    - no rotation (labels are short enough to read horizontally).
    """
    ax.set_xscale("log")

    decades = float("inf")
    if data is not None and len(data) > 0:
        a = np.asarray(data, dtype=np.float64)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size >= 2 and a.min() > 0:
            decades = float(np.log10(a.max() / a.min()))

    if decades >= 1.5:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=numticks))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    else:
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=numticks + 4)
        )
        ax.xaxis.set_major_formatter(
            LogFormatterMathtext(base=10.0, labelOnlyBase=False)
        )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", rotation=0)


# ──────────────────────────────────────────────────────────────
# Figure 1 — Forecast-error vs AUROC scatter (parameterized)
# ──────────────────────────────────────────────────────────────
def _figure_main_variant(
    df: pd.DataFrame,
    err_col: str,
    err_label: str,
    err_short: str,
    out_name: str,
    metric_dw: str = "auroc_D_w",
    metric_label: str = "AUROC",
    family_fit_lines: list[str] | None = None,
) -> dict:
    """One scatter: X = err_col on linear (raw) scale, Y = raw metric value.

    Conference-grade layout — large markers with bilateral edges, light panel
    background, in-axes title (no suptitle), stats box upper-right, external
    family legend on the right.
    """
    fam_style = get_family_style(df["family"])
    fig, ax = plt.subplots(figsize=(11.0, 9.5))
    ax.set_facecolor(COLORS["panel_bg"])
    # Per user: each plot uses 1:1 box aspect
    ax.set_box_aspect(1.0)

    yvals = df[metric_dw].to_numpy()
    err = df[err_col].to_numpy()
    n = len(df)

    ax.set_xlabel(f"Normal {err_label}")
    ax.set_ylabel(f"{metric_label} ($D_w$)")

    if (metric_label.startswith("AUROC") or metric_label.startswith("VUS-ROC")):
        ax.axhline(0.5, color=COLORS["reference"], linestyle="--",
                   linewidth=0.9, alpha=0.55, zorder=1)
        # Inline label so the random baseline is obvious without a legend entry
        ax.text(
            err.max(), 0.5 + 0.012, "random baseline (0.5)",
            ha="right", va="bottom", fontsize=10, color=COLORS["reference"],
            fontstyle="italic", zorder=2,
        )

    # Scatter — large markers with thick dark edge for paper print clarity.
    for family, group in df.groupby("family"):
        st = fam_style[family]
        ax.scatter(group[err_col], group[metric_dw],
                   label=family, s=110, alpha=0.92,
                   marker=st["marker"], c=[st["color"]],
                   edgecolors=COLORS["marker_edge"], linewidths=0.7,
                   zorder=3)

    # Linear fit + correlation
    slope, intercept, _, _, _ = linregress(err, yvals)
    xfit_x = np.linspace(err.min(), err.max(), 200)
    ax.plot(xfit_x, slope * xfit_x + intercept,
            color=COLORS["fit_line"], linewidth=2.2, alpha=0.92, zorder=4)
    pearson_r, pearson_p = pearsonr(err, yvals)

    # Per-family fit lines (drawn by default for all families with N≥3).
    # `family_fit_lines is None` → all eligible families.
    # `family_fit_lines == []` → none (caller explicitly disabled).
    # `family_fit_lines == [names...]` → restrict to those families.
    if family_fit_lines is None:
        fit_targets = sorted(df["family"].unique())
    else:
        fit_targets = family_fit_lines
    for fam in fit_targets:
        sub = df[df["family"] == fam]
        if len(sub) < 3:
            continue
        st = fam_style.get(fam)
        if st is None:
            continue
        fe = sub[err_col].to_numpy()
        fy = sub[metric_dw].to_numpy()
        if np.std(fe) < EPS or np.std(fy) < EPS:
            continue
        fs, fi, _, _, _ = linregress(fe, fy)
        fxx = np.linspace(fe.min(), fe.max(), 50)
        ax.plot(fxx, fs * fxx + fi,
                color=st["color"], linewidth=1.6, alpha=0.85, zorder=3)

    ymin = max(0.0, df[metric_dw].min() - 0.05) if n else 0.0
    ax.set_ylim(ymin, 1.02)

    rho, p_rho = spearmanr(err, yvals)

    # Stats box — upper-right inside axes
    trend_handle = Line2D([0], [0], color=COLORS["fit_line"], linewidth=2.4,
                          label="Linear fit")
    stats_handles = [
        trend_handle,
        Line2D([0], [0], linestyle="None", marker=None, color="none",
               label=_format_stat("Pearson", float(pearson_r),
                                  float(pearson_p), "r")),
        Line2D([0], [0], linestyle="None", marker=None, color="none",
               label=f"$N$ = {n} datasets"),
    ]
    stats_legend = ax.legend(
        handles=stats_handles, loc="upper right",
        frameon=True, fancybox=False, framealpha=0.96,
        edgecolor="#5a5a5a", facecolor="white",
        fontsize=11, borderpad=0.55, handlelength=1.6, labelspacing=0.35,
    )
    ax.add_artist(stats_legend)

    # External family legend (right of the axes)
    family_handles = [
        Line2D([0], [0], linestyle="None",
               marker=fam_style[fam]["marker"],
               markerfacecolor=fam_style[fam]["color"],
               markeredgecolor=COLORS["marker_edge"],
               markeredgewidth=0.6,
               markersize=9, label=fam)
        for fam in sorted(df["family"].unique())
    ]
    fig.legend(handles=family_handles, title="Family",
               loc="center left", bbox_to_anchor=(0.86, 0.5),
               frameon=False, fontsize=11, title_fontsize=12, ncol=1,
               borderaxespad=0.0)

    # In-axes title (no suptitle — keep it clean)
    ax.set_title(
        f"Forecast {err_label} vs anomaly-detection {metric_label}  "
        f"($N$ = {n} datasets)",
        loc="left", pad=12,
    )

    plt.tight_layout(rect=(0.0, 0.02, 0.86, 0.98))
    save_figure(fig, out_name)
    plt.close(fig)
    return {
        "err_col": err_col,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "n": int(n),
    }


def figure_main(
    df: pd.DataFrame,
    metric_dw: str = "auroc_D_w",
    metric_label: str = "AUROC",
    metric_slug: str = "auroc",
    mse_col: str = "normal_mse",
    mae_col: str = "normal_mae",
    family_fit_lines: list[str] | None = None,
) -> dict:
    """Emit main scatter variants for the chosen metric (raw-X linear scale):
       - figure_main_mse_{slug}_scatter   (MSE × metric)
       - figure_main_mae_{slug}_scatter   (MAE × metric)
    """
    variants = {}
    has_mae = mae_col in df.columns and df[mae_col].notna().any()

    variants["mse_raw"] = _figure_main_variant(
        df, mse_col, "MSE", "MSE",
        f"figure_main_mse_{metric_slug}_scatter",
        metric_dw=metric_dw, metric_label=metric_label,
        family_fit_lines=family_fit_lines,
    )
    if has_mae:
        df_mae = df[df[mae_col] > 0].copy()
        if len(df_mae) >= 3:
            variants["mae_raw"] = _figure_main_variant(
                df_mae, mae_col, "MAE", "MAE",
                f"figure_main_mae_{metric_slug}_scatter",
                metric_dw=metric_dw, metric_label=metric_label,
                family_fit_lines=family_fit_lines,
            )

    primary = variants["mse_raw"]
    primary["all_variants"] = variants
    primary["metric"] = metric_slug
    primary["metric_label"] = metric_label
    return primary


# ──────────────────────────────────────────────────────────────
# Figure 1b — Per-family small-multiples scatter
# ──────────────────────────────────────────────────────────────
MIN_FAMILY_SIZE = 3  # exclude singletons/pairs (Spearman ρ unreliable below N=3)


def _figure_per_family_variant(
    df: pd.DataFrame,
    err_col: str,
    err_label: str,
    out_name: str,
    metric_dw: str,
    metric_label: str,
    min_n: int = MIN_FAMILY_SIZE,
) -> dict:
    """Family small-multiples (raw linear-X): one subplot per family with
    ≥ min_n datasets. Conference-grade: larger panels, clean per-panel
    headers, consistent reference baseline."""
    fam_style = get_family_style(df["family"])
    fam_counts = df["family"].value_counts()
    eligible = sorted([f for f, n in fam_counts.items() if n >= min_n])
    if not eligible:
        return {"families": [], "skipped": "no_family_meets_min_n"}

    import math
    n = len(eligible)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    # Each panel target ≈ 4.6 × 4.6 (1:1 aspect via set_box_aspect below)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols + 1.4, 4.8 * nrows + 0.6),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    rho_per_family: dict[str, dict] = {}

    for ax, fam in zip(axes_flat, eligible):
        sub = df[df["family"] == fam].copy()
        st = fam_style[fam]
        ax.set_facecolor(COLORS["panel_bg"])
        err = sub[err_col].to_numpy()
        yvals = sub[metric_dw].to_numpy()
        n_sub = len(sub)
        if n_sub >= 2:
            rho, p_rho = spearmanr(err, yvals)
            rho = float(rho) if rho is not None else float("nan")
            p_rho = float(p_rho) if p_rho is not None else float("nan")
        else:
            rho, p_rho = float("nan"), float("nan")
        pearson_r_fam = float("nan")
        pearson_p_fam = float("nan")
        rho_per_family[fam] = {
            "n": int(n_sub),
            "spearman_rho": rho,
            "spearman_p": p_rho,
        }

        ax.scatter(err, yvals, s=78, alpha=0.92,
                   marker=st["marker"], c=[st["color"]],
                   edgecolors=COLORS["marker_edge"], linewidths=0.6,
                   zorder=3)
        try:
            slope, intercept, _, _, _ = linregress(err, yvals)
            xfit = np.linspace(err.min(), err.max(), 100)
            ax.plot(xfit, slope * xfit + intercept,
                    color=COLORS["fit_line"], linewidth=1.6, alpha=0.9,
                    zorder=2)
            p_r, p_p = pearsonr(err, yvals)
            pearson_r_fam = float(p_r)
            pearson_p_fam = float(p_p)
            rho_per_family[fam]["pearson_r"] = float(p_r)
            rho_per_family[fam]["pearson_p"] = float(p_p)
        except (ValueError, FloatingPointError):
            pass
        ax.set_xlabel(f"Normal {err_label}", fontsize=11)
        ax.set_ylabel(f"{metric_label} ($D_w$)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        # Two-line title: family + N on the first line, Pearson r on the second
        ax.set_title(
            f"{fam}   ($N$ = {len(sub)})\n"
            rf"Pearson $r$ = ${pearson_r_fam:+.3f}$  ($p$ = {pearson_p_fam:.2g})",
            fontsize=11, pad=8,
        )
        if (metric_label.startswith("AUROC") or metric_label.startswith("VUS-ROC")):
            ax.axhline(0.5, color=COLORS["reference"], linestyle="--",
                       linewidth=0.7, alpha=0.55, zorder=1)
        # Each plot 1:1 aspect
        ax.set_box_aspect(1.0)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.axis("off")

    # External family legend (right side)
    family_handles = [
        Line2D([0], [0], linestyle="None",
               marker=fam_style[fam]["marker"],
               markerfacecolor=fam_style[fam]["color"],
               markeredgecolor=COLORS["marker_edge"],
               markeredgewidth=0.6,
               markersize=9, label=fam)
        for fam in eligible
    ]
    legend_x = (4.6 * ncols) / (4.6 * ncols + 1.4)
    fig.legend(
        handles=family_handles, title="Family",
        loc="center left", bbox_to_anchor=(legend_x + 0.005, 0.5),
        frameon=False, fontsize=11, title_fontsize=12, ncol=1,
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=(0.0, 0.0, legend_x, 0.99))
    save_figure(fig, out_name)
    plt.close(fig)
    return {
        "families": eligible,
        "min_n": min_n,
        "rho_per_family": rho_per_family,
    }


def figure_main_per_family(
    df: pd.DataFrame,
    metric_dw: str,
    metric_label: str,
    metric_slug: str,
    mse_col: str,
    mae_col: str,
) -> dict:
    """Emit per-family small-multiples for MSE and MAE (raw linear-X)."""
    out = {}
    out["mse_raw"] = _figure_per_family_variant(
        df, mse_col, "MSE",
        f"figure_main_mse_{metric_slug}_byfamily",
        metric_dw=metric_dw, metric_label=metric_label,
    )
    if mae_col in df.columns and df[mae_col].notna().any():
        df_mae = df[df[mae_col] > 0].copy()
        if len(df_mae) >= 3:
            out["mae_raw"] = _figure_per_family_variant(
                df_mae, mae_col, "MAE",
                f"figure_main_mae_{metric_slug}_byfamily",
                metric_dw=metric_dw, metric_label=metric_label,
            )
    return out


# ──────────────────────────────────────────────────────────────
# Figure 2 — Score time series (3 cases)
# ──────────────────────────────────────────────────────────────
def _load_scores(dataset_id: str) -> pd.DataFrame:
    return load_score_frame(
        get_dataset_results_dir(dataset_id, backbone=CURRENT_BACKBONE) / "scores"
    )


def figure_score_timeseries(
    df: pd.DataFrame,
    metric_dw: str = "auroc_D_w",
    metric_base: str = "auroc_base",
    metric_label: str = "AUROC",
    metric_slug: str = "auroc",
) -> list[str]:
    sorted_df = df.sort_values(metric_dw, ascending=False).reset_index(drop=True)
    n = len(sorted_df)
    cases = []
    if n >= 1:
        cases.append(("Best", sorted_df.iloc[0]))
    if n >= 3:
        cases.append(("Median", sorted_df.iloc[n // 2]))
    if n >= 2:
        cases.append(("Worst", sorted_df.iloc[-1]))

    fig, axes = plt.subplots(
        1, max(len(cases), 1), figsize=(16.5, 5.8), squeeze=False,
    )
    axes = axes[0]

    panel_letters = ["a", "b", "c"]
    used_ids = []
    for ax_idx, (ax, (tag, case)) in enumerate(zip(axes, cases)):
        ds_id = case["dataset_id"]
        used_ids.append(ds_id)
        try:
            scores = _load_scores(ds_id)
        except FileNotFoundError:
            ax.set_title(f"({panel_letters[ax_idx]}) {ds_id}  (scores missing)",
                         loc="left")
            continue

        t = scores["t"].to_numpy()
        D_w = scores["D_w"].to_numpy()
        # V14: secondary curve = D_w_z (production), was `base` (GT-using).
        base = scores["D_w_z"].to_numpy()
        labels = scores["label"].to_numpy()

        # Anomaly shading — softer & with subtle edge
        regions = find_anomaly_regions(labels)
        for s, e in regions:
            if e > s:
                ax.axvspan(t[s], t[min(e, len(t)) - 1],
                           color=COLORS["anomaly_shade"], alpha=0.18,
                           edgecolor="none", zorder=1)

        line1 = ax.plot(t, D_w, color=COLORS["D_w"], linewidth=1.1,
                        label=r"$D_w(t)$", zorder=3)
        ax.set_xlabel(r"time step $t$")
        ax.set_ylabel(r"$D_w(t)$", color=COLORS["D_w"])
        ax.tick_params(axis="y", labelcolor=COLORS["D_w"])
        try:
            if np.nanmin(D_w[D_w > 0]) > 0:
                ax.set_yscale("log")
                ymin, ymax = np.nanpercentile(D_w[D_w > 0], [1, 99])
                if ymin > 0 and ymax > ymin:
                    ax.set_ylim(ymin * 0.5, ymax * 2)
        except (ValueError, IndexError):
            pass

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        line2 = ax2.plot(t, base, color=COLORS["base"], linewidth=0.9,
                         linestyle="--", alpha=0.75,
                         label=r"$D_{w,z}(t)$", zorder=2)
        ax2.set_ylabel(r"$D_{w,z}(t)$", color=COLORS["base"])
        ax2.tick_params(axis="y", labelcolor=COLORS["base"])
        try:
            if np.nanmin(base[base > 0]) > 0:
                ax2.set_yscale("log")
        except (ValueError, IndexError):
            pass

        title = (
            f"({panel_letters[ax_idx]}) {tag}: {ds_id}   "
            rf"{metric_label}$_{{D_w}}$ = {case[metric_dw]:.3f},   "
            rf"{metric_label}$_{{D_{{w,z}}}}$ = {case[metric_base]:.3f}"
        )
        ax.set_title(title, fontsize=12, loc="left", pad=8)

        if ax is axes[0]:
            lines = line1 + line2
            # Anomaly shading proxy in legend
            lines.append(Line2D([0], [0], color=COLORS["anomaly_shade"],
                                lw=8, alpha=0.4, label="anomaly"))
            ax.legend(lines, [l.get_label() for l in lines],
                      loc="upper left", fontsize=10, frameon=True,
                      framealpha=0.95, edgecolor="#bdbdbd")

        # 1:1 box aspect on the primary axes (twin axis follows automatically)
        ax.set_box_aspect(1.0)

    plt.tight_layout()
    save_figure(fig, f"figure_supporting_score_timeseries_{metric_slug}")
    plt.close(fig)
    return used_ids


# ──────────────────────────────────────────────────────────────
# Figure 3 — Family-wise distribution + paired scatter
# ──────────────────────────────────────────────────────────────
def figure_auroc_distribution(
    df: pd.DataFrame,
    metric_dw: str = "auroc_D_w",
    metric_base: str = "auroc_base",
    metric_label: str = "AUROC",
    metric_slug: str = "auroc",
) -> dict:
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    BOXPLOT_FAMILIES = [
        "CATSv2", "Exathlon", "GHL", "LTDB",
        "MITDB", "OPPORTUNITY", "SMD", "SVDB",
    ]
    show_random_line = (metric_label.startswith("AUROC") or metric_label.startswith("VUS-ROC"))

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.set_facecolor(COLORS["panel_bg"])

    # Fixed alphabetical order — CATSv2, Exathlon, GHL, LTDB, MITDB,
    # OPPORTUNITY, SMD, SVDB. Drop families that have no data in this run.
    eligible = [f for f in BOXPLOT_FAMILIES if (df["family"] == f).any()]
    bp_data = [df.loc[df["family"] == f, metric_dw].to_numpy() for f in eligible]

    # Viridis colormap fixed to AUROC range [0, 1] — each box's fill encodes
    # the family's median AUROC (high → yellow, low → dark purple).
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    medians = [
        float(np.median(d)) if d.size else float("nan") for d in bp_data
    ]

    if eligible:
        bp = ax.boxplot(
            bp_data, labels=eligible, patch_artist=True,
            widths=0.55, showmeans=True,
            boxprops=dict(linewidth=0.9),
            medianprops=dict(color="black", linewidth=1.6),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=6,
                           markeredgewidth=0.9),
            whiskerprops=dict(color="#5a5a5a", linewidth=0.9),
            capprops=dict(color="#5a5a5a", linewidth=0.9),
            flierprops=dict(marker="o", markersize=4.5, alpha=0.55,
                            markerfacecolor="#888888",
                            markeredgecolor="none"),
            zorder=2,
        )
        # Color each box by its family's median AUROC via magma [0, 1]
        for patch, med in zip(bp["boxes"], medians):
            patch.set_facecolor(cmap(norm(med if np.isfinite(med) else 0.0)))
            patch.set_edgecolor(COLORS["marker_edge"])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    if show_random_line:
        # Draw on top of every box so the reference is unmistakable.
        ax.axhline(0.5, color=COLORS["highlight"], linestyle="--",
                   linewidth=1.6, alpha=0.95, zorder=10)
        ax.text(
            len(eligible) + 0.45 if eligible else 1.0, 0.51,
            "random (0.5)",
            ha="right", va="bottom", fontsize=10,
            color=COLORS["highlight"], fontstyle="italic", zorder=11,
        )

    # Legend (mean / median markers + colorbar info)
    legend_handles = [
        Line2D([0], [0], marker="D", color="none",
               markerfacecolor="white", markeredgecolor="black",
               markersize=7, markeredgewidth=0.9, label="mean"),
        Line2D([0], [0], color="black", linewidth=1.6, label="median"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=10, frameon=True, framealpha=0.95,
              edgecolor="#bdbdbd")

    # Magma colorbar — communicates the box-fill encoding
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.025, aspect=24)
    cbar.set_label(f"{metric_label}", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#888888")

    ax.set_ylabel(f"{metric_label} ($D_w$)")
    ax.set_xlabel("Dataset family")
    ax.set_ylim(0.0, 1.02)
    # Each plot 1:1 visual aspect (per user request)
    ax.set_box_aspect(1.0)
    ax.set_title("Family-wise distribution of detection performance",
                 loc="left", pad=10)

    n_dw_wins = int((df[metric_dw] > df[metric_base]).sum())
    n_total = int(len(df))
    mean_diff = float((df[metric_dw] - df[metric_base]).mean())

    families = eligible  # for return — preserves original key
    plt.tight_layout()
    save_figure(fig, f"figure_supporting_{metric_slug}_distribution")
    plt.close(fig)
    return {
        "n_dw_wins": n_dw_wins, "n_total": n_total,
        "mean_delta": mean_diff, "metric_label": metric_label,
        "boxplot_families": families,
    }


# ──────────────────────────────────────────────────────────────
# Captions + statistics table
# ──────────────────────────────────────────────────────────────
def write_captions(
    main_stats: dict, dist_stats: dict,
    metric_label: str = "AUROC", metric_slug: str = "auroc",
) -> None:
    captions = []
    pearson = main_stats.get("pearson_r", float("nan"))
    pearson_p = main_stats.get("pearson_p", float("nan"))
    n = main_stats.get("n", 0)
    sig = (
        "supports" if (np.isfinite(pearson) and pearson < -0.2)
        else "does not strongly support"
    )
    random_clause = (
        f" Dashed line indicates random baseline ({metric_label} = 0.5)."
        if (metric_label.startswith("AUROC") or metric_label.startswith("VUS-ROC")) else ""
    )
    captions.append(
        f"**Figure 1.** Cross-dataset relationship between forecasting accuracy "
        f"and anomaly detection performance. Each point represents one dataset; "
        f"X-axis shows the 1-step Normal MSE (raw value) and Y-axis shows the "
        f"{metric_label} of forecast divergence $D_w$. The Pearson correlation "
        f"$r = {pearson:+.3f}$ ($p = {pearson_p:.3g}$, $N = {n}$ datasets) {sig} "
        f"the hypothesis that forecasting accuracy and detection performance "
        f"are systematically related.{random_clause}"
    )
    captions.append(
        f"**Figure 2.** Per-timestep score trajectories for representative "
        f"datasets (best / median / worst by {metric_label}). For each dataset the "
        f"forecast divergence $D_w(t)$ (blue, left axis) and baseline 1-step "
        f"prediction error (red, right axis) are shown over time. Gray-shaded "
        f"regions indicate ground-truth anomaly intervals."
    )
    captions.append(
        f"**Figure 3.** Cross-dataset {metric_label} analysis. (a) Distribution of "
        f"$D_w$ {metric_label} across dataset families. (b) Paired comparison of $D_w$ "
        f"versus baseline {metric_label}. $D_w$ outperforms the baseline in "
        f"{dist_stats['n_dw_wins']} of {dist_stats['n_total']} datasets "
        f"(mean $\\Delta${metric_label} = {dist_stats['mean_delta']:+.3f})."
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cap_name = "captions.md" if metric_slug == "auroc" else f"captions_{metric_slug}.md"
    with open(FIG_DIR / cap_name, "w") as fh:
        fh.write("\n\n".join(captions))


def write_statistics_table(
    df: pd.DataFrame, main_stats: dict,
    metric_dw: str = "auroc_D_w",
    metric_base: str = "auroc_base",
    metric_label: str = "AUROC",
    metric_slug: str = "auroc",
    mse_col: str = "normal_mse",
    mae_col: str = "normal_mae",
) -> None:
    pearson = main_stats.get("pearson_r", float("nan"))
    pearson_p = main_stats.get("pearson_p", float("nan"))
    cutoff = main_stats.get("min_eval_anomalies_cutoff", 0)

    n_dw_wins = int((df[metric_dw] > df[metric_base]).sum())
    has_eval_col = "n_eval_anomalies" in df.columns

    mse_label = "Normal MSE"
    if "strict" in mse_col:
        mse_label += " (strict"
        if "predictable" in mse_col:
            mse_label += ", predictable channels"
        mse_label += ")"
    elif "predictable" in mse_col:
        mse_label += " (predictable channels)"
    header = (
        f"| Dataset | Family | {mse_label} | {metric_label} ($D_w$) | "
        f"{metric_label} (base) | $\\Delta${metric_label} | $N$ anomaly events"
        + (" | $N$ eval anomalies |" if has_eval_col else " |")
    )
    sep = "|---|---|---|---|---|---|---|" + ("---|" if has_eval_col else "")

    window_mode = main_stats.get("normal_window", "target")
    lines = [
        f"# Per-dataset Metrics ({metric_label}, normal_window={window_mode})",
        "",
        f"min_eval_anomalies cutoff applied: **{cutoff}**",
        "",
        header,
        sep,
    ]
    for _, row in df.sort_values(metric_dw, ascending=False).iterrows():
        delta = row[metric_dw] - row[metric_base]
        line = (
            f"| {row['dataset_id']} | {row['family']} | "
            f"{row[mse_col]:.4f} | {row[metric_dw]:.3f} | "
            f"{row[metric_base]:.3f} | {delta:+.3f} | "
            f"{int(row['n_anomaly_events'])} "
        )
        if has_eval_col:
            line += f"| {int(row['n_eval_anomalies'])} |"
        else:
            line += "|"
        lines.append(line)
    lines += [
        "",
        "## Summary",
        f"- min_eval_anomalies cutoff: {cutoff}",
        f"- normal_window mode: {window_mode}  (column: {mse_col})",
        f"- N datasets: {len(df)}",
        f"- Mean {metric_label} ($D_w$): {df[metric_dw].mean():.3f}",
        f"- Mean {metric_label} (base): {df[metric_base].mean():.3f}",
        f"- $D_w$ wins: {n_dw_wins}/{len(df)} datasets",
        f"- Pearson $r$ ({mse_label} vs {metric_label}): {pearson:+.3f} (p={pearson_p:.3g})",
    ]
    out_path = STATS_PATH if metric_slug == "auroc" else \
        STATS_PATH.with_name(f"statistics_table_{metric_slug}.md")
    out_path.write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", type=str, default="iTransformer",
        help="Backbone whose per_dataset_metrics CSV to plot. Default: iTransformer.",
    )
    parser.add_argument(
        "--min-eval-anomalies", type=int, default=0,
        help="Drop datasets whose n_eval_anomalies < this threshold before plotting. "
             "Default 0 = include all that survived 05_metrics.",
    )
    parser.add_argument(
        "--metric", choices=list(METRIC_COLS.keys()), default="auroc",
        help="Detection metric to plot on the Y-axis. Default: auroc.",
    )
    parser.add_argument(
        "--normal-window", choices=["target", "strict"], default="strict",
        help="Which Normal MSE/MAE definition to plot on the X-axis. "
             "'target' = label[t]==0 only (legacy); "
             "'strict' = full [t-L_w, t+H-1] window all label==0 (recommended). "
             "Default: strict.",
    )
    parser.add_argument(
        "--channel-agg", choices=["mean", "median"], default="median",
        help="How to aggregate per-channel MSE/MAE into a per-dataset scalar. "
             "'mean' = flat mean over (t, c) — sensitive to outlier channels "
             "(one degenerate channel can dominate). "
             "'median' = per-channel mean over time, then median across "
             "channels — robust to outlier channels (no need for explicit "
             "channel mask). Default: median.",
    )
    parser.add_argument(
        "--exclude-family", nargs="*", default=["SMAP", "MSL"],
        help="Family names to drop before plotting. Default: SMAP MSL "
             "(use --exclude-family with no values to include everything).",
    )
    parser.add_argument(
        "--family-fit-lines", nargs="*", default=None,
        help="Per-family linear fit lines on the main scatter. "
             "Default (no flag): draw fits for ALL families with N≥3. "
             "Pass specific names to restrict (e.g. --family-fit-lines "
             "Exathlon SMD). Use --no-family-fits to disable entirely.",
    )
    parser.add_argument(
        "--no-family-fits", action="store_true",
        help="Disable per-family linear fit lines on the main scatter "
             "(only the overall trend line is drawn).",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF export (PNG only). Faster for iterative review.",
    )
    args = parser.parse_args()
    # Wire --no-pdf into the module-level switch used by save_figure().
    global SAVE_PDF, FIG_DIR, STATS_PATH, CURRENT_BACKBONE
    SAVE_PDF = not args.no_pdf
    # Per-backbone output paths — rebind the module-level FIG_DIR/STATS_PATH
    # used by save_figure() and other helpers.
    FIG_DIR = FIG_DIR_BASE / args.backbone
    STATS_PATH = RESULTS_ROOT / args.backbone / "statistics_table.md"
    CURRENT_BACKBONE = args.backbone
    metrics_path = METRICS_DIR / f"per_dataset_metrics__{args.backbone}.csv"

    metric_dw, metric_base, metric_label, metric_slug = METRIC_COLS[args.metric]
    _suffix_window = "" if args.normal_window == "target" else "_strict"
    _suffix_agg = "_median" if args.channel_agg == "median" else ""
    mse_col = f"normal_mse{_suffix_window}{_suffix_agg}"
    mae_col = f"normal_mae{_suffix_window}{_suffix_agg}"

    if not metrics_path.exists():
        sys.exit(f"Missing {metrics_path}. Run `05_metrics.py --backbone {args.backbone}` first.")
    apply_paper_style()

    df = pd.read_csv(metrics_path)
    df = df[df["status"] == "ok"].copy()
    if metric_dw not in df.columns or metric_base not in df.columns:
        sys.exit(
            f"Metric '{args.metric}' not found in {metrics_path}. "
            f"Re-run 05_metrics.py without --skip-vus."
        )
    if mse_col not in df.columns:
        sys.exit(
            f"Column '{mse_col}' not found. Either re-run 05_metrics.py with the "
            f"updated code, or run scripts/05b_supplement_normal_strict.py to "
            f"backfill strict-window MSE/MAE."
        )
    for c in ("normal_mse", "normal_mae",
              "normal_mse_median", "normal_mae_median",
              "normal_mse_strict", "normal_mae_strict",
              "normal_mse_strict_median", "normal_mae_strict_median",
              metric_dw, metric_base):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[mse_col, metric_dw, metric_base])
    df = df[df[mse_col] > 0]
    if args.exclude_family:
        excl = set(args.exclude_family)
        before = len(df)
        df = df[~df["family"].isin(excl)].copy()
        print(f"  applied --exclude-family={sorted(excl)}: "
              f"{before} → {len(df)} dataset(s)")
    if args.min_eval_anomalies > 0 and "n_eval_anomalies" in df.columns:
        before = len(df)
        df["n_eval_anomalies"] = pd.to_numeric(df["n_eval_anomalies"], errors="coerce").fillna(0)
        df = df[df["n_eval_anomalies"] >= args.min_eval_anomalies].copy()
        print(f"  applied --min-eval-anomalies={args.min_eval_anomalies}: "
              f"{before} → {len(df)} dataset(s)")
    if len(df) == 0:
        sys.exit("No datasets with valid metrics for plotting.")

    print(f"07_visualization — metric={args.metric} "
          f"(cols={metric_dw}/{metric_base}), "
          f"normal_window={args.normal_window} ({mse_col}/{mae_col}), "
          f"{len(df)} dataset(s) available")

    # Resolve --family-fit-lines / --no-family-fits into a single value:
    #   None → draw all eligible (default)
    #   []   → draw none (--no-family-fits)
    #   list → draw only listed
    if args.no_family_fits:
        fit_lines_arg = []
    else:
        fit_lines_arg = args.family_fit_lines  # None (default) or explicit list

    main_stats = figure_main(df, metric_dw=metric_dw,
                             metric_label=metric_label, metric_slug=metric_slug,
                             mse_col=mse_col, mae_col=mae_col,
                             family_fit_lines=fit_lines_arg)
    main_stats["normal_window"] = args.normal_window
    main_stats["family_fit_lines"] = (
        list(fit_lines_arg) if fit_lines_arg is not None else "all"
    )
    family_stats = figure_main_per_family(
        df, metric_dw=metric_dw, metric_label=metric_label,
        metric_slug=metric_slug, mse_col=mse_col, mae_col=mae_col,
    )
    main_stats["per_family"] = family_stats
    main_stats["min_eval_anomalies_cutoff"] = int(args.min_eval_anomalies)
    used_ids = figure_score_timeseries(
        df, metric_dw=metric_dw, metric_base=metric_base,
        metric_label=metric_label, metric_slug=metric_slug,
    )
    dist_stats = figure_auroc_distribution(
        df, metric_dw=metric_dw, metric_base=metric_base,
        metric_label=metric_label, metric_slug=metric_slug,
    )
    write_captions(main_stats, dist_stats,
                   metric_label=metric_label, metric_slug=metric_slug)
    write_statistics_table(
        df, main_stats,
        metric_dw=metric_dw, metric_base=metric_base,
        metric_label=metric_label, metric_slug=metric_slug,
        mse_col=mse_col, mae_col=mae_col,
    )

    print(f"  ✓ Wrote figures + captions in {FIG_DIR}")
    print(f"  ✓ Wrote {STATS_PATH}")
    print(f"  Score-timeseries cases: {used_ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
