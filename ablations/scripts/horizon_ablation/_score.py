"""Score + metric computation for horizon ablation.

Wraps production score_utils + 05_metrics.process_one with two adaptations:
  1. Plumbs `pred_len` correctly into score_utils.compute_backward_score_per_channel
     and compute_train_baseline_stats. Production 04_score_compute.py hardcodes
     pred_len=96 at the call site — we bypass that script and call score_utils
     directly with the right H.
  2. Temporarily monkey-patches artifact_paths.RESULTS_ROOT (and score_utils
     mirror) so prepare_dataset_bundle / load_prediction_artifacts read from
     the ablation tree (ablations/results/horizon/H<H>/<key>/...) instead of
     the production results/<key>/ tree.

We do NOT modify any production .py file. The patch is in-process and reverted
in a try/finally so concurrent (or subsequent) production runs see the original
RESULTS_ROOT.

PRODUCTION_REF for score logic: scripts/04_score_compute.py::process_one
PRODUCTION_REF for metric logic: scripts/05_metrics.py::process_one
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import artifact_paths  # type: ignore
import score_utils  # type: ignore
# 05_metrics.py starts with a digit so it can't be imported the usual way.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_metrics_mod", _REPO_ROOT / "scripts" / "05_metrics.py")
_metrics_mod = _ilu.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_metrics_mod)  # type: ignore


@contextlib.contextmanager
def _patch_results_root(ablation_root: Path):
    """Redirect RESULTS_ROOT so production helpers (prepare_dataset_bundle,
    load_prediction_artifacts, get_dataset_results_dir) read+write under the
    ablation tree for the duration of the block."""
    orig_ap = artifact_paths.RESULTS_ROOT
    orig_su = score_utils.RESULTS_ROOT
    orig_m = _metrics_mod.RESULTS_ROOT
    try:
        artifact_paths.RESULTS_ROOT = ablation_root
        score_utils.RESULTS_ROOT = ablation_root
        _metrics_mod.RESULTS_ROOT = ablation_root
        yield
    finally:
        artifact_paths.RESULTS_ROOT = orig_ap
        score_utils.RESULTS_ROOT = orig_su
        _metrics_mod.RESULTS_ROOT = orig_m


def compute_scores(
    dataset_key: str,
    lookback: int,
    pred_len: int,
    ablation_root: Path,
    backbone: str = "iTransformer",
) -> dict:
    """Mirror of 04_score_compute.py::process_one with explicit pred_len.

    Reads predictions + bundle_meta from ablation_root (= the H-specific
    ablation tree the orchestrator already populated), writes
    scores.parquet + scores_per_ch.npz back into the same tree.

    Returns a small status dict for the run log.
    """
    out: dict = {
        "dataset_id": dataset_key,
        "backbone": backbone,
        "pred_len": pred_len,
        "status": "ok",
        "error": "",
    }
    with _patch_results_root(ablation_root):
        try:
            if not score_utils.has_required_prediction_artifacts(
                dataset_key, backbone=backbone
            ):
                out["status"] = "skip"
                out["error"] = "no_prediction_artifacts"
                return out

            bundle = score_utils.prepare_dataset_bundle(dataset_key)
            preds, labels_saved, _ = score_utils.load_prediction_artifacts(
                dataset_key, backbone=backbone
            )

            if labels_saved.shape[0] != bundle.test_labels.shape[0]:
                raise ValueError(
                    f"label length mismatch: saved={labels_saved.shape[0]} "
                    f"bundle={bundle.test_labels.shape[0]}"
                )

            T_train = int(bundle.train_values_norm.shape[0])
            T_test_start = int(bundle.test_start)

            score_test, per_ch_test, _ = score_utils.compute_backward_score_per_channel(
                predictions=preds,
                test_values_norm=bundle.test_values_norm,
                test_labels=bundle.test_labels,
                block_size=4096,
                lookback=lookback,
                pred_len=pred_len,
            )

            train_pred_path = ablation_root / dataset_key / backbone / "predictions_train.npy"
            preds_train = np.load(train_pred_path) if train_pred_path.exists() else None

            channels = preds.shape[2]
            if preds_train is not None and preds_train.shape[0] > 0:
                score_train, per_ch_train, _ = score_utils.compute_backward_score_per_channel(
                    predictions=preds_train,
                    test_values_norm=bundle.train_values_norm,
                    test_labels=bundle.full_labels[:T_train].astype(np.int64),
                    block_size=4096,
                    lookback=lookback,
                    pred_len=pred_len,
                )
            else:
                score_train = pd.DataFrame(columns=["t", "D_w", "label"])
                per_ch_train = {
                    "D_w_c": np.zeros((0, channels), dtype=np.float64),
                    "t":     np.zeros((0,), dtype=np.int64),
                    "label": np.zeros((0,), dtype=np.int64),
                }

            score_test_g = score_test.copy()
            score_test_g["t"] = score_test_g["t"].astype(np.int64) + T_test_start
            per_ch_test_g = dict(per_ch_test)
            per_ch_test_g["t"] = per_ch_test["t"].astype(np.int64) + T_test_start

            score_frame = pd.concat([score_train, score_test_g], ignore_index=True)
            per_ch = {
                "D_w_c": np.concatenate(
                    [per_ch_train["D_w_c"], per_ch_test_g["D_w_c"]], axis=0
                ),
                "t":     np.concatenate([per_ch_train["t"], per_ch_test_g["t"]]),
                "label": np.concatenate([per_ch_train["label"], per_ch_test_g["label"]]),
            }

            # Train baseline (default — falls back to val if train missing)
            baseline = None
            if preds_train is not None and preds_train.shape[0] > 0:
                baseline = score_utils.compute_train_baseline_stats(
                    predictions_train=preds_train,
                    train_values_norm=bundle.train_values_norm,
                    lookback=lookback,
                    pred_len=pred_len,
                )
            else:
                val_pred_path = ablation_root / dataset_key / backbone / "predictions_val.npy"
                if val_pred_path.exists():
                    preds_val = np.load(val_pred_path)
                    if preds_val.shape[0] > 0:
                        baseline = score_utils.compute_train_baseline_stats(
                            predictions_train=preds_val,
                            train_values_norm=bundle.val_values_norm,
                            lookback=lookback,
                            pred_len=pred_len,
                        )

            if baseline is not None:
                D_w_z = score_utils.apply_channel_zscore_aggregation(
                    D_w_c=per_ch["D_w_c"],
                    baseline=baseline,
                    centering="mean",
                    scaling="std",
                    agg="max",
                )
                per_ch["baseline_mean"] = baseline["mean"]
                per_ch["baseline_std"] = baseline["std"]
            else:
                D_w_z = np.full(score_frame.shape[0], np.nan, dtype=np.float64)
                per_ch["baseline_mean"] = np.full(channels, np.nan)
                per_ch["baseline_std"] = np.full(channels, np.nan)

            score_frame["D_w_z"] = D_w_z
            score_frame_out = score_frame.drop(columns=["D_w"])

            out_dir = ablation_root / dataset_key / backbone
            out_dir.mkdir(parents=True, exist_ok=True)
            score_utils.save_table_with_fallback(
                score_frame_out, out_dir / "scores.parquet"
            )
            score_utils.save_per_channel_scores(per_ch, out_dir)
            out["n_eval_rows"] = int(score_frame.shape[0])
            return out

        except Exception as exc:
            out["status"] = "fail"
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out


def compute_metrics(
    dataset_key: str,
    pred_len: int,
    ablation_root: Path,
    backbone: str = "iTransformer",
) -> dict:
    """Reuse 05_metrics.process_one verbatim with the RESULTS_ROOT patch.

    05_metrics is backbone-agnostic and pred_len-agnostic — it just reads
    scores.parquet (which we wrote under the ablation tree) and runs the
    TSB-AD-M metric stack. No reimplementation needed.
    """
    with _patch_results_root(ablation_root):
        try:
            bundle = score_utils.prepare_dataset_bundle(dataset_key)
            family = bundle.family
            # n_anomaly_events = number of 0→1 transitions in full label stream
            n_evt = int(np.sum(np.diff(bundle.full_labels.astype(np.int32)) == 1))
            n_eval = int(np.sum(bundle.test_labels))
            rec = _metrics_mod.process_one(
                dataset_key, family, n_evt, n_eval, skip_vus=False, backbone=backbone,
            )
            rec["pred_len"] = pred_len
            return rec
        except Exception as exc:
            return {
                "dataset_id": dataset_key, "backbone": backbone, "pred_len": pred_len,
                "status": "fail", "error": f"{type(exc).__name__}: {exc}",
            }
