"""
04_score_compute.py — V13 backward sliding evaluation (full-series).

For every dataset with prediction artifacts (train + test), compute the
per-timestep backward forecast divergence D_w(t) on the FULL concatenated
series (train + test, global indexing) and emit:

  results/{dataset_id}/scores.parquet   columns: t, D_w, D_w_z, label

The score range covers both regions:
  - train: t ∈ [L+H-1, T_train_data - 1]
  - test:  t ∈ [T_test_start + L+H-1, T_full - 1]   (global idx)
  - boundary gap [T_train_data, T_test_start + L+H-2] is unscored (NaN)
    and edge-filled by 05_metrics, matching TSB-AD-M's full-series eval.

A cross-dataset summary log is written to:
  results/04_score_compute_log.csv

Score is NOT inverted: higher D_w ⇒ more anomalous.
Channel aggregation = max (NOT mean).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from artifact_paths import (
    get_dataset_results_dir,
    RESULTS_ROOT,
)
from score_utils import (
    apply_channel_zscore_aggregation,
    compute_backward_score_per_channel,
    compute_space_check,
    compute_train_baseline_stats,
    has_required_prediction_artifacts,
    load_prediction_artifacts,
    prepare_dataset_bundle,
    save_per_channel_scores,
    save_table_with_fallback,
    set_all_seeds,
)
from artifact_paths import RESULTS_ROOT as _RESULTS_ROOT_FOR_TRAIN


LOG_COLUMNS = [
    "dataset_id", "family",
    "test_length", "n_pred", "n_eval_rows",
    "n_skipped_nan", "elapsed_sec",
    "D_w_min", "D_w_median", "D_w_max",
    "D_w_z_min", "D_w_z_median", "D_w_z_max",
    "n_baseline_train_rows", "n_baseline_channels_used", "baseline_source",
    "label_pos_count", "label_neg_count",
    "space_consistent", "pred_normalized", "gt_normalized",
    "score_path", "status", "error",
]


def _percentile_safe(arr: np.ndarray, q: float) -> float:
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return float("nan")
    return float(np.percentile(fin, q))


def process_one(dataset_id: str, family: str) -> dict:
    record = {col: "" for col in LOG_COLUMNS}
    record.update({"dataset_id": dataset_id, "family": family, "status": "ok", "error": ""})

    if not has_required_prediction_artifacts(dataset_id):
        record["status"] = "skip"
        record["error"] = "no_prediction_artifacts"
        return record

    try:
        bundle = prepare_dataset_bundle(dataset_id)
        preds, labels_saved, meta = load_prediction_artifacts(dataset_id)

        # Sanity: labels saved by inference must equal bundle labels (post-split)
        if labels_saved.shape[0] != bundle.test_labels.shape[0]:
            raise ValueError(
                f"label length mismatch: saved={labels_saved.shape[0]} "
                f"bundle={bundle.test_labels.shape[0]}"
            )

        # Score-space consistency check (test predictions only — train preds
        # are in-sample so don't inform space-consistency much)
        space = compute_space_check(preds, bundle.test_values_norm)
        record["space_consistent"] = bool(space["space_consistent"])
        record["pred_normalized"] = bool(space["pred_normalized"])
        record["gt_normalized"] = bool(space["gt_normalized"])

        if not space["space_consistent"]:
            # NOTE: is_normalized_space() is a heuristic (mean<0.5, std<1.5).
            # Test-set distribution shift can push pred or gt out of tolerance
            # even though both are technically StandardScaler-normalized.
            # We warn and continue rather than skip, since AUROC of D_w is
            # invariant to monotonic transforms of the score.
            record["error"] = (
                f"WARN_space_check (pred_norm={space['pred_normalized']}, "
                f"gt_norm={space['gt_normalized']}); proceeding"
            )

        # ── Full-series scoring (TSB-AD-M aligned) ─────────────────────
        # Compute D_w on TRAIN region (predictions_train + train_values_norm)
        # and TEST region (predictions_test + test_values_norm) separately,
        # then concatenate with global timestep indexing. The boundary gap
        # of L+H-1 timesteps at train→test (uncovered by either inference)
        # is left NaN and edge-filled in 05_metrics, matching the TSB-AD-M
        # convention of evaluating on the full series.
        t0 = time.time()
        T_train = int(bundle.train_values_norm.shape[0])    # length of train_data array
        T_test_start = int(bundle.test_start)               # global offset for test region

        # Test region (existing pathway)
        score_test, per_ch_test, skipped_test = compute_backward_score_per_channel(
            predictions=preds,
            test_values_norm=bundle.test_values_norm,
            test_labels=bundle.test_labels,
            block_size=4096,
        )

        # Train region — predictions_train.npy is required for full-series eval.
        val_pred_path = _RESULTS_ROOT_FOR_TRAIN / dataset_id / "predictions_val.npy"
        train_pred_path = _RESULTS_ROOT_FOR_TRAIN / dataset_id / "predictions_train.npy"
        preds_train = np.load(train_pred_path) if train_pred_path.exists() else None

        if preds_train is not None and preds_train.shape[0] > 0:
            score_train, per_ch_train, skipped_train = compute_backward_score_per_channel(
                predictions=preds_train,
                test_values_norm=bundle.train_values_norm,
                test_labels=bundle.full_labels[:T_train].astype(np.int64),
                block_size=4096,
            )
        else:
            channels = preds.shape[2]
            score_train = pd.DataFrame(columns=["t", "D_w", "label"])
            per_ch_train = {
                "D_w_c":  np.zeros((0, channels), dtype=np.float64),
                "t":      np.zeros((0,), dtype=np.int64),
                "label":  np.zeros((0,), dtype=np.int64),
            }
            skipped_train = []

        # Shift test region's local t to global indexing (test_start = official_train_end).
        score_test_g = score_test.copy()
        score_test_g["t"] = score_test_g["t"].astype(np.int64) + T_test_start
        per_ch_test_g = dict(per_ch_test)
        per_ch_test_g["t"] = per_ch_test["t"].astype(np.int64) + T_test_start

        # Concatenate train + test
        score_frame = pd.concat([score_train, score_test_g], ignore_index=True)
        per_ch = {
            "D_w_c":  np.concatenate([per_ch_train["D_w_c"],  per_ch_test_g["D_w_c"]],  axis=0),
            "t":      np.concatenate([per_ch_train["t"],      per_ch_test_g["t"]]),
            "label":  np.concatenate([per_ch_train["label"],  per_ch_test_g["label"]]),
        }
        skipped = (
            list(skipped_train)
            + [{"t": int(s["t"]) + T_test_start, "reason": s["reason"]}
               for s in skipped_test]
        )

        # ── Train-baseline channel z-score + max aggregation (D_w_z) ──
        # Compute per-channel D_w_c distribution stats on the train predictions,
        # then z-score the full-series D_w_c using train (μ, σ) and aggregate
        # across channels via max. Train baseline is V13's production choice
        # (matches V13_RESULTS_REPORT z_train_max winner): it has wider channel
        # coverage than val baseline because the model overfits val on ~41/200
        # datasets, leaving them with all-channel σ ≤ ε and a degenerate score.
        # Fully GT-free either way.
        baseline = None
        baseline_source = "none"
        if preds_train is not None and preds_train.shape[0] > 0:
            baseline = compute_train_baseline_stats(
                predictions_train=preds_train,
                train_values_norm=bundle.train_values_norm,
                lookback=192,
                pred_len=96,
            )
            baseline_source = "train"

        if baseline is None and val_pred_path.exists():
            # Fallback: train predictions missing — use val instead.
            preds_val = np.load(val_pred_path)
            if preds_val.shape[0] > 0:
                baseline = compute_train_baseline_stats(
                    predictions_train=preds_val,
                    train_values_norm=bundle.val_values_norm,
                    lookback=192,
                    pred_len=96,
                )
                baseline_source = "val_fallback"

        baseline_n = 0
        baseline_n_channels_used = 0
        if baseline is not None:
            baseline_n = int(baseline.get("n", 0))
            D_w_z = apply_channel_zscore_aggregation(
                D_w_c=per_ch["D_w_c"],
                baseline=baseline,
                centering="mean",
                scaling="std",
                agg="max",
            )
            sigma = baseline["std"]
            baseline_n_channels_used = int(np.sum(np.isfinite(sigma) & (sigma > 1e-8)))
        else:
            D_w_z = np.full(score_frame.shape[0], np.nan, dtype=np.float64)

        score_frame["D_w_z"] = D_w_z
        per_ch["baseline_mean"] = (baseline["mean"]
                                   if baseline is not None
                                   else np.full(per_ch["D_w_c"].shape[1], np.nan))
        per_ch["baseline_std"] = (baseline["std"]
                                  if baseline is not None
                                  else np.full(per_ch["D_w_c"].shape[1], np.nan))
        record["baseline_source"] = baseline_source

        elapsed = time.time() - t0

        out_dir = get_dataset_results_dir(dataset_id)
        score_path = save_table_with_fallback(
            score_frame, out_dir / "scores.parquet"
        )
        save_per_channel_scores(per_ch, out_dir)

        D_w = score_frame["D_w"].to_numpy(dtype=np.float64)
        # label_*_count is over the FULL series (train + test) — matches
        # TSB-AD-M convention where train labels (typically all 0) are
        # included in the metric calculation.
        full_label = bundle.full_labels.astype(np.int64)

        record.update({
            "test_length": int(bundle.total_timesteps),
            "n_pred": int(preds.shape[0]
                          + (preds_train.shape[0] if preds_train is not None else 0)),
            "n_eval_rows": int(score_frame.shape[0]),
            "n_skipped_nan": int(len(skipped)),
            "elapsed_sec": float(round(elapsed, 3)),
            "D_w_min": _percentile_safe(D_w, 0),
            "D_w_median": _percentile_safe(D_w, 50),
            "D_w_max": _percentile_safe(D_w, 100),
            "D_w_z_min": _percentile_safe(D_w_z, 0),
            "D_w_z_median": _percentile_safe(D_w_z, 50),
            "D_w_z_max": _percentile_safe(D_w_z, 100),
            "n_baseline_train_rows": baseline_n,
            "n_baseline_channels_used": baseline_n_channels_used,
            "label_pos_count": int(np.sum(full_label == 1)),
            "label_neg_count": int(np.sum(full_label == 0)),
            "score_path": str(score_path),
        })
        return record

    except Exception as exc:
        record["status"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record


def _list_all_datasets_with_predictions() -> list[dict]:
    """Enumerate every dataset under RESULTS_ROOT that has predictions_test.npy.

    V13 evaluates every dataset (no filter step) — TSB-AD-M-aligned eval
    convention.
    """
    rows: list[dict] = []
    if not RESULTS_ROOT.exists():
        return rows
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "predictions_test.npy").exists():
            continue
        # family is best-effort (resolved later by prepare_dataset_bundle)
        rows.append({"dataset_id": d.name, "family": d.name.split("_")[0]})
    return rows


def _process_one_with_key(args_tuple) -> tuple[str, dict]:
    """Worker entry point — wraps process_one with the key in result tuple.

    Top-level (not closure) so it can be pickled across ProcessPoolExecutor.
    """
    dataset_id, family = args_tuple
    rec = process_one(dataset_id, family)
    return (dataset_id, rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Limit to specific dataset keys.",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip datasets whose scores.parquet/.csv already exists. "
             "Useful for incremental re-runs after adding new datasets.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel processes. Default: 8 (chosen as a safe "
             "balance — 16+ workers can OOM on big multi-GB prediction "
             "arrays). Set 1 to disable multiprocessing.",
    )
    args = parser.parse_args()

    set_all_seeds(42)

    rows = _list_all_datasets_with_predictions()
    print(f"  enumerating {len(rows)} dataset(s) with prediction artifacts")

    if args.only:
        only = set(args.only)
        rows = [r for r in rows if r["dataset_id"] in only]
    if args.only_missing:
        from artifact_paths import RESULTS_ROOT as _RR
        before = len(rows)
        rows = [r for r in rows
                if not ((_RR / r["dataset_id"] / "scores.parquet").exists()
                        or (_RR / r["dataset_id"] / "scores.csv").exists())]
        print(f"  --only-missing: {before} → {len(rows)} dataset(s) (skipping already-done)")

    workers = max(1, min(int(args.workers), len(rows))) if rows else 1
    print(f"04_score_compute — processing {len(rows)} dataset(s), workers={workers}")
    summary: list[dict] = []
    n_ok, n_skip, n_fail = 0, 0, 0

    def _report(i: int, total: int, dataset_id: str, rec: dict) -> None:
        nonlocal n_ok, n_skip, n_fail
        if rec["status"] == "ok":
            n_ok += 1
            print(f"  [{i:>3}/{total}] {dataset_id} ✓ "
                  f"rows={rec['n_eval_rows']} elapsed={rec['elapsed_sec']:.1f}s",
                  flush=True)
        elif rec["status"] == "skip":
            n_skip += 1
            print(f"  [{i:>3}/{total}] {dataset_id} SKIP ({rec['error']})", flush=True)
        else:
            n_fail += 1
            print(f"  [{i:>3}/{total}] {dataset_id} FAIL ({rec['error']})", flush=True)

    if workers == 1:
        for i, r in enumerate(rows, 1):
            rec = process_one(r["dataset_id"], r["family"])
            summary.append(rec)
            _report(i, len(rows), r["dataset_id"], rec)
    else:
        # Preserve input order in the summary by indexing futures.
        order = {r["dataset_id"]: idx for idx, r in enumerate(rows)}
        results: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_process_one_with_key, (r["dataset_id"], r["family"])): r["dataset_id"]
                for r in rows
            }
            for done_count, fut in enumerate(as_completed(futs), 1):
                dataset_id, rec = fut.result()
                results[order[dataset_id]] = rec
                _report(done_count, len(rows), dataset_id, rec)
        summary = [results[i] for i in range(len(rows))]

    log_path = RESULTS_ROOT / "04_score_compute_log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for s in summary:
            writer.writerow({k: s.get(k, "") for k in LOG_COLUMNS})

    print(f"\n  ✓ Wrote {log_path}")
    print(f"  ok={n_ok}, skip={n_skip}, fail={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
