"""
05_metrics.py — V12 v7.0 per-dataset metrics.

Reads results/{dataset_id}/scores.parquet (produced by 04_score_compute.py)
and computes:
  - Normal MSE (1-step prediction error on label==0 timesteps in evaluable range)
  - AUROC of D_w (point-based; sklearn roc_auc_score)

Output: results/04_metrics/per_dataset_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

# TSB-AD's threshold sweep evaluates extreme thresholds where the prediction
# is all-zeros or all-ones; sklearn raises UndefinedMetricWarning in those
# cases. Silenced for clean per-dataset logs — the metric values themselves
# are unaffected (these thresholds simply contribute F1=0 to the max-sweep).
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(str(Path(__file__).resolve().parent))

from artifact_paths import (
    RESULTS_ROOT,
    get_dataset_results_dir,
)
from score_utils import (
    compute_normal_errors,
    compute_tsb_metrics,
    compute_tsb_sliding_window,
    has_required_prediction_artifacts,
    load_per_channel_scores,
    load_prediction_artifacts,
    load_score_frame,
    prepare_dataset_bundle,
)
from artifact_paths import resolve_dataset


_AGG_FUNCS = {
    "max":    lambda x: np.nanmax(x, axis=1),
    "median": lambda x: np.nanmedian(x, axis=1),
    "mean":   lambda x: np.nanmean(x, axis=1),
}


METRICS_DIR = RESULTS_ROOT / "04_metrics"

# 6-variant comparison (raw / z_train × max / mean / median).
#   D_w   = raw_max         (the V13 production GT-free score)
#   D_w_z = z_train_max     (production with channel z-score normalization)
# We keep the legacy `*_D_w` and `*_D_w_z` columns for 06/07
# backward-compatibility, and add 4 more variants (raw_mean, raw_median,
# z_train_mean, z_train_median) so this single CSV reproduces the
# compare_agg_normalize 6-variant view inside the main 05 pipeline.
# (The GT-using `*_base` columns were removed in V14.)
VARIANTS_6 = [
    "raw_max", "raw_mean", "raw_median",
    "z_train_max", "z_train_mean", "z_train_median",
]
VARIANT_METRIC_KEYS = ["AUROC", "VUS-PR", "VUS-ROC", "AUC-PR", "Standard-F1", "PA-F1"]


def _variant_col(metric: str, variant: str) -> str:
    """e.g. ('VUS-PR', 'z_train_max') -> 'vus_pr_z_train_max'."""
    return f"{metric.lower().replace('-', '_')}_{variant}"


METRICS_COLUMNS = [
    "dataset_id", "family", "test_length",
    "n_eval_rows", "n_anomaly_events", "n_eval_anomalies",
    "n_label_pos", "n_label_neg",
    "normal_mse", "normal_mae",
    "normal_mse_median", "normal_mae_median",
    "normal_mse_strict", "normal_mae_strict",
    "normal_mse_strict_median", "normal_mae_strict_median",
    "n_normal_target", "n_normal_strict",
    # Legacy columns (raw_max == D_w, z_train_max == D_w_z) — kept for 06/07.
    "auroc_D_w",
    "vus_pr_D_w",
    "vus_roc_D_w",
    "auc_pr_D_w",
    "standard_f1_D_w",
    "pa_f1_D_w",
    "sliding_window",
    "vus_error_D_w",
    "auroc_D_w_z",
    "vus_pr_D_w_z",
    "vus_roc_D_w_z",
    "auc_pr_D_w_z",
    "standard_f1_D_w_z",
    "pa_f1_D_w_z",
    "vus_error_D_w_z",
    # 6-variant comparison columns (vus_pr_raw_max, vus_pr_raw_mean, ...).
    *[_variant_col(m, v) for m in VARIANT_METRIC_KEYS for v in VARIANTS_6],
    "n_kept_train_baseline",
    "status", "error",
]


def _list_all_datasets_with_scores() -> list[dict]:
    """Enumerate every dataset under RESULTS_ROOT that has scores.parquet/.csv.

    V13 evaluates every dataset (no filter step) — TSB-AD-M-aligned
    convention.
    """
    rows: list[dict] = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not ((d / "scores.parquet").exists() or (d / "scores.csv").exists()):
            continue
        rows.append({
            "dataset_id": d.name,
            "family": d.name.split("_")[0],
            "n_anomaly_events": "0",
            "n_eval_anomalies": "0",
        })
    return rows


def _safe_auroc(label: np.ndarray, score: np.ndarray) -> float:
    fin = np.isfinite(score)
    label, score = label[fin], score[fin]
    if label.size == 0:
        return float("nan")
    classes = np.unique(label)
    if classes.size < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def _edge_fill_score(score: np.ndarray) -> np.ndarray:
    """Return a clean score vector using TSB-AD-style padding.

    TSB-AD model wrappers return one score per input timestep before calling
    `get_metrics`. Forecasting wrappers pad undefined leading regions with the
    first valid score, not NaN. V12 evaluates TEST artifacts here, so this
    helper edge-fills undefined non-evaluable TEST rows and forward-fills any
    internal unscored gaps.
    """
    out = np.asarray(score, dtype=np.float64).copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float64)

    valid_idx = np.flatnonzero(valid)
    first = int(valid_idx[0])
    last = int(valid_idx[-1])
    out[:first] = out[first]
    out[last + 1:] = out[last]

    # Forward-fill internal gaps; the leading region is already initialized.
    last_val = out[first]
    for i in range(first, last + 1):
        if np.isfinite(out[i]):
            last_val = out[i]
        else:
            out[i] = last_val
    return out


def process_one(
    dataset_id: str, family: str,
    n_anomaly_events: int, n_eval_anomalies: int,
    skip_vus: bool = False,
    agg_mode: str = "max",
) -> dict:
    rec = {col: "" for col in METRICS_COLUMNS}
    rec.update({
        "dataset_id": dataset_id, "family": family,
        "n_anomaly_events": n_anomaly_events,
        "n_eval_anomalies": n_eval_anomalies,
        "status": "ok", "error": "",
    })

    score_path = get_dataset_results_dir(dataset_id) / "scores"
    try:
        score_frame = load_score_frame(score_path)
    except FileNotFoundError as exc:
        rec["status"] = "skip"
        rec["error"] = str(exc)
        return rec

    if not has_required_prediction_artifacts(dataset_id):
        rec["status"] = "skip"
        rec["error"] = "no_prediction_artifacts"
        return rec

    try:
        bundle = prepare_dataset_bundle(dataset_id)
        preds, _, _ = load_prediction_artifacts(dataset_id)

        # V13 full-series metrics — TSB-AD-M-aligned.
        # 04_score_compute.py produces scores.parquet with global timestep
        # indices spanning [0, T_full-1]. We allocate a length-T_full score
        # vector, fill in the evaluable rows, and edge-fill non-evaluable
        # leading/trailing/boundary regions for get_metrics.
        full_len = int(bundle.total_timesteps)
        label = bundle.full_labels.astype(np.int64)
        D_w = np.full(full_len, np.nan, dtype=np.float64)

        # Channel aggregation: read per-channel D_w_c and collapse to 1D using
        # the requested mode (max | median | mean). For agg_mode="max" the
        # result is identical to score_frame["D_w"] (which 04_score_compute
        # produced via channel max), so we use whichever data path is available.
        ts_global = score_frame["t"].to_numpy(dtype=np.int64)
        if agg_mode == "max":
            D_w[ts_global] = score_frame["D_w"].to_numpy(dtype=np.float64)
        else:
            try:
                per_ch = load_per_channel_scores(get_dataset_results_dir(dataset_id))
                D_w_c = per_ch["D_w_c"]
                pc_t = per_ch["t"].astype(np.int64)
                D_w[pc_t] = _AGG_FUNCS[agg_mode](D_w_c)
            except FileNotFoundError as exc:
                rec["status"] = "skip"
                rec["error"] = f"no_per_channel_scores: {exc}"
                return rec
        D_w = _edge_fill_score(D_w)

        auroc_dw = _safe_auroc(label, D_w)

        # D_w_z: train-baseline z-score per channel + median across channels.
        # Computed in 04 and persisted as score_frame["D_w_z"]. Edge-fill the
        # non-evaluable region same way as D_w.
        D_w_z = np.full(full_len, np.nan, dtype=np.float64)
        if "D_w_z" in score_frame.columns:
            D_w_z[ts_global] = score_frame["D_w_z"].to_numpy(dtype=np.float64)
        D_w_z = _edge_fill_score(D_w_z)
        auroc_dwz = _safe_auroc(label, D_w_z)

        normal_err = compute_normal_errors(
            predictions=preds,
            test_values_norm=bundle.test_values_norm,
            test_labels=bundle.test_labels,
        )

        rec.update({
            "test_length": full_len,
            "n_eval_rows": int(score_frame.shape[0]),
            "n_label_pos": int(np.sum(label == 1)),
            "n_label_neg": int(np.sum(label == 0)),
            "normal_mse": float(normal_err["normal_mse"]),
            "normal_mae": float(normal_err["normal_mae"]),
            "normal_mse_median": float(normal_err["normal_mse_median"]),
            "normal_mae_median": float(normal_err["normal_mae_median"]),
            "normal_mse_strict": float(normal_err["normal_mse_strict"]),
            "normal_mae_strict": float(normal_err["normal_mae_strict"]),
            "normal_mse_strict_median": float(normal_err["normal_mse_strict_median"]),
            "normal_mae_strict_median": float(normal_err["normal_mae_strict_median"]),
            "n_normal_target": int(normal_err["n_normal_target"]),
            "n_normal_strict": int(normal_err["n_normal_strict"]),
            "auroc_D_w": float(auroc_dw),
            "auroc_D_w_z": float(auroc_dwz),
        })

        if skip_vus:
            return rec

        # TSB-AD-M aligned: slidingWindow from RAW full-series first channel,
        # AND scoring on full-series score/label (D_w, D_w_z already extended
        # to length T_full above). Reproduces TSB-AD-M's
        # `get_metrics(output, full_label, slidingWindow=sw)` call shape.
        sw_int = compute_tsb_sliding_window(bundle.full_values_raw)

        # All anomaly metrics (VUS-PR/ROC, AUC-PR/ROC, Standard-F1, PA-F1,
        # Event-based-F1, R-based-F1, Affiliation-F) come from a SINGLE
        # call into TSB_AD.evaluation.metrics.get_metrics — the same
        # function used by TSB-AD-M's published benchmark.
        m = compute_tsb_metrics(D_w, label, sw_int)
        m_dwz = compute_tsb_metrics(D_w_z, label, sw_int)

        rec.update({
            "vus_pr_D_w":      float(m["VUS-PR"]),
            "vus_roc_D_w":     float(m["VUS-ROC"]),
            "auc_pr_D_w":      float(m["AUC-PR"]),
            "standard_f1_D_w": float(m["Standard-F1"]),
            "pa_f1_D_w":       float(m["PA-F1"]),
            "sliding_window":  int(m.get("sliding_window_used", sw_int)),
            "vus_error_D_w":   m.get("error", ""),
            "vus_pr_D_w_z":      float(m_dwz["VUS-PR"]),
            "vus_roc_D_w_z":     float(m_dwz["VUS-ROC"]),
            "auc_pr_D_w_z":      float(m_dwz["AUC-PR"]),
            "standard_f1_D_w_z": float(m_dwz["Standard-F1"]),
            "pa_f1_D_w_z":       float(m_dwz["PA-F1"]),
            "vus_error_D_w_z":   m_dwz.get("error", ""),
        })

        # AUROC: use TSB-AD's value (sklearn.roc_auc_score on the same
        # padded series — same input we just fed get_metrics) so that all
        # rank-based metrics in this row come from one consistent pipeline.
        if np.isfinite(m["AUC-ROC"]):
            rec["auroc_D_w"] = float(m["AUC-ROC"])
        if np.isfinite(m_dwz["AUC-ROC"]):
            rec["auroc_D_w_z"] = float(m_dwz["AUC-ROC"])

        # ── 6-variant comparison (raw / z_train × max / mean / median) ──
        # Reads scores_per_ch.npz (D_w_c) and re-derives the train baseline
        # from predictions_train.npy, then evaluates the 6 channel-aggregation
        # variants on the same full-series score/label vectors. Output schema
        # matches compare_agg_normalize.py for cross-tool consistency.
        try:
            from score_utils import (
                compute_train_baseline_stats as _compute_baseline,
            )
            per_ch_path = get_dataset_results_dir(dataset_id) / "scores_per_ch.npz"
            train_pred_path = get_dataset_results_dir(dataset_id) / "predictions_train.npy"
            if per_ch_path.exists() and train_pred_path.exists():
                pc = load_per_channel_scores(get_dataset_results_dir(dataset_id))
                Dwc = pc["D_w_c"]                                # (T_eval, C)
                t_idx = pc["t"].astype(np.int64)
                base_train = _compute_baseline(
                    np.load(train_pred_path), bundle.train_values_norm,
                    lookback=192, pred_len=96,
                )
                mu, sigma = base_train["mean"], base_train["std"]
                eps = 1e-8
                valid_c = np.isfinite(sigma) & (sigma > eps)
                rec["n_kept_train_baseline"] = int(valid_c.sum())
                Zt = np.full_like(Dwc, np.nan)
                if valid_c.any():
                    Zt[:, valid_c] = (
                        (Dwc[:, valid_c] - mu[valid_c][None, :])
                        / sigma[valid_c][None, :]
                    )
                # Channel aggregations across 6 variants
                with np.errstate(invalid="ignore"):
                    aggs = {
                        "raw_max":        np.nanmax(Dwc, axis=1),
                        "raw_mean":       np.nanmean(Dwc, axis=1),
                        "raw_median":     np.nanmedian(Dwc, axis=1),
                        "z_train_max":    np.nanmax(Zt, axis=1),
                        "z_train_mean":   np.nanmean(Zt, axis=1),
                        "z_train_median": np.nanmedian(Zt, axis=1),
                    }
                # Edge-fill on full series + TSB-AD metrics per variant
                for v_name, v_vals in aggs.items():
                    v_full = np.full(full_len, np.nan, dtype=np.float64)
                    v_full[t_idx] = v_vals
                    v_full = _edge_fill_score(v_full)
                    if skip_vus:
                        # Only AUROC available without VUS calculation
                        rec[_variant_col("AUROC", v_name)] = _safe_auroc(label, v_full)
                        continue
                    m_v = compute_tsb_metrics(v_full, label, sw_int)
                    auc_roc = m_v.get("AUC-ROC", float("nan"))
                    rec[_variant_col("AUROC", v_name)] = (
                        float(auc_roc) if np.isfinite(auc_roc)
                        else _safe_auroc(label, v_full)
                    )
                    for k in ("VUS-PR", "VUS-ROC", "AUC-PR",
                              "Standard-F1", "PA-F1"):
                        rec[_variant_col(k, v_name)] = float(m_v.get(k, np.nan))
        except Exception as v_exc:
            # 6-variant block is best-effort; legacy columns above are
            # the canonical source for downstream 06/07.
            rec["error"] = (rec.get("error") or "") + (
                f" | 6-variant: {type(v_exc).__name__}: {v_exc}"
            )

        return rec
    except Exception as exc:
        rec["status"] = "fail"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


TSB_COLUMNS = [
    "dataset",
    "VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
    "Standard-F1", "PA-F1",
    "MSE", "MAE",
]


def _tsb_row_from_rec(rec: dict) -> dict:
    """Convert an internal `rec` dict to a TSB-AD-format row dict."""
    ds_key = str(rec.get("dataset_id", ""))
    try:
        info = resolve_dataset(ds_key)
        name = str(info["filename"]).replace(".csv", "")
    except Exception:
        name = ds_key

    def _f(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.6f}"
        return v if v is not None else ""

    return {
        "dataset":     name,
        "MSE":         _f(rec.get("normal_mse", "")),
        "MAE":         _f(rec.get("normal_mae", "")),
        "VUS-PR":      _f(rec.get("vus_pr_D_w", "")),
        "VUS-ROC":     _f(rec.get("vus_roc_D_w", "")),
        "AUC-PR":      _f(rec.get("auc_pr_D_w", "")),
        "AUC-ROC":     _f(rec.get("auroc_D_w", "")),
        "Standard-F1": _f(rec.get("standard_f1_D_w", "")),
        "PA-F1":       _f(rec.get("pa_f1_D_w", "")),
    }


def _tsb_row_from_csv_row(r: dict) -> dict:
    """Convert a row from per_dataset_metrics.csv to TSB-AD format
    (used to seed the incremental file with already-ok rows under
    --only-missing mode)."""
    ds_key = str(r.get("dataset_id", ""))
    try:
        info = resolve_dataset(ds_key)
        name = str(info["filename"]).replace(".csv", "")
    except Exception:
        name = ds_key
    return {
        "dataset":     name,
        "MSE":         r.get("normal_mse", ""),
        "MAE":         r.get("normal_mae", ""),
        "VUS-PR":      r.get("vus_pr_D_w", ""),
        "VUS-ROC":     r.get("vus_roc_D_w", ""),
        "AUC-PR":      r.get("auc_pr_D_w", ""),
        "AUC-ROC":     r.get("auroc_D_w", ""),
        "Standard-F1": r.get("standard_f1_D_w", ""),
        "PA-F1":       r.get("pa_f1_D_w", ""),
    }


def _process_one_worker(args_tuple) -> tuple[str, dict]:
    """Worker entry point — wraps process_one with the key in result tuple.

    Top-level (not closure) so it can be pickled across ProcessPoolExecutor.
    """
    dataset_id, family, n_evt, n_eval, skip_vus, agg_mode = args_tuple
    rec = process_one(
        dataset_id, family, n_evt, n_eval,
        skip_vus=skip_vus,
        agg_mode=agg_mode,
    )
    return (dataset_id, rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip datasets already present (status=ok) in per_dataset_metrics.csv "
             "and merge new results into the existing CSV.",
    )
    parser.add_argument(
        "--min-eval-anomalies", type=int, default=0,
        help="Skip datasets whose n_eval_anomalies (from filter CSV) is below this. "
             "Default 0 = include all that passed the base filter.",
    )
    parser.add_argument(
        "--skip-vus", action="store_true",
        help="Skip VUS-PR / VUS-ROC / AUC-PR computation (faster; AUROC only).",
    )
    parser.add_argument(
        "--agg", choices=list(_AGG_FUNCS.keys()), default="max",
        help="Channel aggregation for D_w(t). 'max' (default) reproduces the "
             "score saved in scores.parquet by 04_score_compute.py. 'median' "
             "and 'mean' read scores_per_ch.npz and re-aggregate. Output "
             "files are suffixed with _{agg} when agg != max so each variant "
             "lands in its own CSV (kept side-by-side with the max baseline).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel processes. Default: cpu_count() // 2 "
             "(set 1 to disable multiprocessing).",
    )
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.agg == "max" else f"_{args.agg}"
    out_path = METRICS_DIR / f"per_dataset_metrics{suffix}.csv"

    rows = _list_all_datasets_with_scores()
    print(f"  enumerating {len(rows)} dataset(s) with score artifacts")
    if args.only:
        only = set(args.only)
        rows = [r for r in rows if r["dataset_id"] in only]
    if args.min_eval_anomalies > 0:
        rows = [r for r in rows
                if int(r.get("n_eval_anomalies") or 0) >= args.min_eval_anomalies]
        print(f"  applied --min-eval-anomalies={args.min_eval_anomalies} "
              f"→ {len(rows)} dataset(s) remain")

    tsb_path = out_path.parent / f"metrics_tsb_format{suffix}.csv"

    # --only-missing: seed both files with already-ok rows so the on-disk
    # state is consistent before we start appending new ones.
    existing_ok_ids: set = set()
    seed_records: list[dict] = []
    if args.only_missing and out_path.exists():
        import pandas as _pd
        existing_df = _pd.read_csv(out_path)
        ok_df = existing_df[existing_df["status"] == "ok"]
        existing_ok_ids = set(ok_df["dataset_id"].astype(str).tolist())
        seed_records = ok_df.to_dict(orient="records")
        before = len(rows)
        rows = [r for r in rows if r["dataset_id"] not in existing_ok_ids]
        print(f"  --only-missing: {before} → {len(rows)} dataset(s) "
              f"(seeding {len(seed_records)} already-ok rows into both CSVs)")

    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) // 2)
    workers = max(1, min(workers, len(rows))) if rows else 1
    print(f"05_metrics — processing {len(rows)} dataset(s), workers={workers}")

    out_records: list[dict] = []
    n_ok, n_skip, n_fail = 0, 0, 0

    def _emit(i: int, total: int, ds_id: str, rec: dict) -> str:
        """Format the per-dataset progress line."""
        nonlocal n_ok, n_skip, n_fail
        if rec["status"] == "ok":
            n_ok += 1
            line = (
                f"  [{i:>3}/{total}] {ds_id} ✓ "
                f"MSE={rec['normal_mse']:.4f} MAE={rec['normal_mae']:.4f} "
                f"AUROC(D_w)={rec['auroc_D_w']:.3f}"
            )
            if not args.skip_vus and isinstance(rec.get("vus_pr_D_w"), float):
                line += f" VUS_PR(D_w)={rec['vus_pr_D_w']:.3f}"
            return line
        elif rec["status"] == "skip":
            n_skip += 1
            return f"  [{i:>3}/{total}] {ds_id} SKIP ({rec['error']})"
        else:
            n_fail += 1
            return f"  [{i:>3}/{total}] {ds_id} FAIL ({rec['error']})"

    # Open both CSVs and write headers + any seed rows. We flush after every
    # completed result so an external `tail -f` (or a re-opened spreadsheet
    # view) sees progress in real time.
    with open(out_path, "w", newline="") as full_fh, \
         open(tsb_path, "w", newline="") as tsb_fh:
        full_w = csv.DictWriter(full_fh, fieldnames=METRICS_COLUMNS)
        full_w.writeheader()
        tsb_w = csv.DictWriter(tsb_fh, fieldnames=TSB_COLUMNS)
        tsb_w.writeheader()

        for sr in seed_records:
            full_w.writerow({k: sr.get(k, "") for k in METRICS_COLUMNS})
            tsb_w.writerow(_tsb_row_from_csv_row(sr))
        if seed_records:
            full_fh.flush()
            tsb_fh.flush()

        def _write(rec: dict) -> None:
            out_records.append(rec)
            full_w.writerow({k: rec.get(k, "") for k in METRICS_COLUMNS})
            full_fh.flush()
            if rec["status"] == "ok":
                tsb_w.writerow(_tsb_row_from_rec(rec))
                tsb_fh.flush()

        if workers == 1:
            for i, r in enumerate(rows, 1):
                ds_id = r["dataset_id"]
                rec = process_one(
                    ds_id, r["family"],
                    int(r.get("n_anomaly_events") or 0),
                    int(r.get("n_eval_anomalies") or 0),
                    skip_vus=args.skip_vus,
                    agg_mode=args.agg,
                )
                _write(rec)
                print(_emit(i, len(rows), ds_id, rec), flush=True)
        else:
            tasks = [
                (
                    r["dataset_id"], r["family"],
                    int(r.get("n_anomaly_events") or 0),
                    int(r.get("n_eval_anomalies") or 0),
                    args.skip_vus, args.agg,
                )
                for r in rows
            ]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process_one_worker, t): t[0] for t in tasks}
                for done_count, fut in enumerate(as_completed(futs), 1):
                    ds_id, rec = fut.result()
                    _write(rec)
                    print(_emit(done_count, len(rows), ds_id, rec), flush=True)

    print(f"\n  ✓ Wrote {out_path}")
    print(f"  ✓ Wrote {tsb_path}")
    print(f"  ok={n_ok}, skip={n_skip}, fail={n_fail}")

    # Sanity-check distribution
    ok = [r for r in out_records if r["status"] == "ok"]
    if ok:
        aurocs = np.array([r["auroc_D_w"] for r in ok], dtype=np.float64)
        mses = np.array([r["normal_mse"] for r in ok], dtype=np.float64)
        maes = np.array([r["normal_mae"] for r in ok], dtype=np.float64)
        print(f"  [SANITY] AUROC(D_w): min={np.nanmin(aurocs):.3f} "
              f"median={np.nanmedian(aurocs):.3f} max={np.nanmax(aurocs):.3f}")
        print(f"  [SANITY] Normal MSE: min={np.nanmin(mses):.4f} "
              f"median={np.nanmedian(mses):.4f} max={np.nanmax(mses):.4f}")
        print(f"  [SANITY] Normal MAE: min={np.nanmin(maes):.4f} "
              f"median={np.nanmedian(maes):.4f} max={np.nanmax(maes):.4f}")
        if not args.skip_vus:
            vus = np.array([r.get("vus_pr_D_w", float("nan")) for r in ok], dtype=np.float64)
            if np.any(np.isfinite(vus)):
                print(f"  [SANITY] VUS-PR(D_w): min={np.nanmin(vus):.3f} "
                      f"median={np.nanmedian(vus):.3f} max={np.nanmax(vus):.3f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
