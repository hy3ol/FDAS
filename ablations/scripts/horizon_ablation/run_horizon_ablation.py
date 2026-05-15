"""Horizon ablation orchestrator.

Sweeps pred_len ∈ {192, 336} for iTransformer, L=192 fixed, on the TSB-AD-M
200-dataset benchmark. Runs the full FDAS pipeline (01 split → 02 train →
03 infer → 04 score → 05 metrics) per (H, dataset_key), with all artifacts
isolated under ablations/results/horizon/H<H>/ so the production H=96
results/ and models/ trees stay frozen.

Design (recap):
- Production code is NOT modified. All horizon-specific behavior lives in
  ablations/scripts/horizon_ablation/.
- _split.prepare_split() reimplements 01's split + scaler with explicit
  (lookback, pred_len). Writes data/ for 02/03 to consume.
- 02_train.py / 03_inference.py are invoked as subprocesses (unchanged) and
  produce models/<key>/iTransformer/ + results/<key>/iTransformer/ as usual.
  The orchestrator then moves these into the ablation tree.
- _score.compute_scores() reimplements 04's process_one with correct pred_len.
- _score.compute_metrics() delegates to 05_metrics.process_one via in-process
  call, with RESULTS_ROOT monkey-patched to the ablation tree.
- Serial per-(H, key) iteration. Production training/inference must not be
  running concurrently (we hold the singleton data/ + models/<key>/iTransformer/).

Usage:
  python ablations/scripts/horizon_ablation/run_horizon_ablation.py
  python ablations/scripts/horizon_ablation/run_horizon_ablation.py --pred-lens 192
  python ablations/scripts/horizon_ablation/run_horizon_ablation.py --only SMD_id_1 Genesis
  python ablations/scripts/horizon_ablation/run_horizon_ablation.py --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from _split import prepare_split, write_data_dir, write_bundle_meta, SegmentTooShortError  # noqa: E402
from _score import compute_scores, compute_metrics  # noqa: E402

from artifact_paths import (  # noqa: E402
    list_available_dataset_keys,
    MODELS_ROOT as PROD_MODELS_ROOT,
    RESULTS_ROOT as PROD_RESULTS_ROOT,
)

ABLATION_ROOT = _REPO_ROOT / "ablations" / "results" / "horizon"
BACKBONE = "iTransformer"
LOOKBACK = 192

RUN_LOG_COLUMNS = [
    "pred_len", "dataset_key", "status", "phase",
    "train_sec", "infer_sec", "score_sec", "metric_sec",
    "elapsed_sec", "error",
]
METRIC_COLUMNS = [
    "pred_len", "dataset_key", "family",
    "vus_pr", "vus_roc", "auc_pr", "auc_roc", "standard_f1", "pa_f1",
    "normal_mse", "normal_mae",
    "n_eval_rows", "status", "error",
]


def _h_root(pred_len: int) -> Path:
    return ABLATION_ROOT / f"H{pred_len}"


def _move(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


# ── Production artifact backup/restore ───────────────────────────────────
# 02_train.py / 03_inference.py write into production models/<key>/iTransformer/
# and results/<key>/iTransformer/. Running the ablation with a different pred_len
# overwrites these singletons. We rename the existing (H=96) artifacts to
# `<name>.h96.bak` before training, then rename them back AFTER the H≠96
# artifacts have been moved into the ablation tree. The production H=96
# checkpoints + scores + per_dataset CSV stay bit-for-bit identical end-to-end.
#
# `data/` (the data prep singleton) is intentionally NOT backed up — it's
# transient and overwritten by every 01 run, never persisted state.
_BAK_SUFFIX = ".h96.bak"


def _bak_path(p: Path) -> Path:
    return p.with_name(p.name + _BAK_SUFFIX)


def _backup_prod_iTransformer(dataset_key: str) -> None:
    """Rename production iTransformer artifacts out of the way so 02/03
    can write fresh ones without destroying the H=96 paper-grade outputs."""
    for prod in (
        PROD_MODELS_ROOT / dataset_key / BACKBONE,
        PROD_RESULTS_ROOT / dataset_key / BACKBONE,
    ):
        bak = _bak_path(prod)
        if prod.exists() and not bak.exists():
            prod.rename(bak)


def _restore_prod_iTransformer(dataset_key: str) -> None:
    """Inverse of _backup_prod_iTransformer. Called AFTER the orchestrator
    has moved the newly trained H≠96 artifacts to the ablation tree."""
    for prod in (
        PROD_MODELS_ROOT / dataset_key / BACKBONE,
        PROD_RESULTS_ROOT / dataset_key / BACKBONE,
    ):
        bak = _bak_path(prod)
        if bak.exists():
            if prod.exists():
                # Defensive: 02/03 left a stale dir despite our move. Drop it,
                # the .bak is the source of truth.
                shutil.rmtree(prod)
            bak.rename(prod)


def _run_subprocess(cmd: list[str], log_path: Path) -> tuple[int, float]:
    """Run a production script, stream stdout to log file. Returns (returncode, elapsed_sec)."""
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n# === {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd, cwd=_REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT
        )
    return proc.returncode, time.time() - t0


def _checkpoint_exists(pred_len: int, dataset_key: str) -> bool:
    return (
        _h_root(pred_len) / "models" / dataset_key / BACKBONE / "best_model.pth"
    ).exists()


def _scores_exist(pred_len: int, dataset_key: str) -> bool:
    out_dir = _h_root(pred_len) / dataset_key / BACKBONE
    return (out_dir / "scores.parquet").exists() or (out_dir / "scores.csv").exists()


def run_one(
    pred_len: int,
    dataset_key: str,
    batch_size: int | None,
    skip_existing: bool,
) -> tuple[dict, dict]:
    """Returns (log_row, metric_row). metric_row may be empty on early-skip."""
    h_root = _h_root(pred_len)
    h_root.mkdir(parents=True, exist_ok=True)
    log_path = h_root / f"_logs/{dataset_key}.log"
    t_all = time.time()

    log_row = {col: "" for col in RUN_LOG_COLUMNS}
    log_row.update({"pred_len": pred_len, "dataset_key": dataset_key,
                    "status": "ok", "phase": "init"})
    metric_row = {col: "" for col in METRIC_COLUMNS}
    metric_row.update({"pred_len": pred_len, "dataset_key": dataset_key})

    # ── Split ──
    log_row["phase"] = "split"
    try:
        split = prepare_split(dataset_key, lookback=LOOKBACK, pred_len=pred_len)
    except SegmentTooShortError as exc:
        log_row.update({"status": "skip_too_short", "error": str(exc)})
        return log_row, metric_row
    except Exception as exc:
        log_row.update({"status": "fail",
                        "error": f"split: {type(exc).__name__}: {exc}"})
        return log_row, metric_row

    # Decide if we can skip training/inference (artifacts already present).
    artifacts_dir = h_root / dataset_key / BACKBONE
    have_inference = (artifacts_dir / "predictions_test.npy").exists() \
        and (artifacts_dir / "predictions_train.npy").exists()
    have_checkpoint = _checkpoint_exists(pred_len, dataset_key)

    if not (skip_existing and have_inference):
        # ── data/ + bundle_meta materialization ──
        write_data_dir(split)
        write_bundle_meta(split, h_root / dataset_key / "bundle_meta.json")

        # ── Back up production H=96 iTransformer artifacts so 02/03 don't
        # destroy them. The try/finally guarantees the .h96.bak directories
        # are renamed back even if 02 or 03 fails mid-run. ──
        _backup_prod_iTransformer(dataset_key)
        try:
            # ── 02_train ──
            log_row["phase"] = "train"
            if skip_existing and have_checkpoint:
                log_row["train_sec"] = 0.0
                # Stage saved checkpoint into prod location so 03 finds it.
                saved = h_root / "models" / dataset_key / BACKBONE
                prod_target = PROD_MODELS_ROOT / dataset_key / BACKBONE
                if prod_target.exists():
                    shutil.rmtree(prod_target)
                prod_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(saved, prod_target)
            else:
                cmd = [sys.executable, "scripts/02_train.py", "--backbone", BACKBONE]
                if batch_size is not None:
                    cmd += ["--batch-size", str(batch_size)]
                rc, elapsed = _run_subprocess(cmd, log_path)
                log_row["train_sec"] = round(elapsed, 1)
                if rc != 0:
                    log_row.update({"status": "fail", "error": f"02_train rc={rc}"})
                    return log_row, metric_row

            # ── 03_inference ──
            log_row["phase"] = "infer"
            cmd = [sys.executable, "scripts/03_inference.py", "--backbone", BACKBONE]
            if batch_size is not None:
                cmd += ["--batch-size", str(batch_size)]
            rc, elapsed = _run_subprocess(cmd, log_path)
            log_row["infer_sec"] = round(elapsed, 1)
            if rc != 0:
                log_row.update({"status": "fail", "error": f"03_inference rc={rc}"})
                return log_row, metric_row

            # ── Move newly trained H≠96 artifacts into ablation tree.
            # H=96 originals are still safe under <name>.h96.bak. ──
            prod_models = PROD_MODELS_ROOT / dataset_key / BACKBONE
            prod_results = PROD_RESULTS_ROOT / dataset_key / BACKBONE
            ab_models = h_root / "models" / dataset_key / BACKBONE
            ab_results = h_root / dataset_key / BACKBONE
            if prod_models.exists():
                _move(prod_models, ab_models)
            if prod_results.exists():
                _move(prod_results, ab_results)
        finally:
            _restore_prod_iTransformer(dataset_key)

    # ── 04_score ──
    log_row["phase"] = "score"
    if skip_existing and _scores_exist(pred_len, dataset_key):
        log_row["score_sec"] = 0.0
    else:
        t0 = time.time()
        s_rec = compute_scores(dataset_key, LOOKBACK, pred_len, h_root, BACKBONE)
        log_row["score_sec"] = round(time.time() - t0, 1)
        if s_rec["status"] != "ok":
            log_row.update({"status": "fail", "error": f"04: {s_rec.get('error', '')}"})
            return log_row, metric_row

    # ── 05_metrics ──
    log_row["phase"] = "metric"
    t0 = time.time()
    m_rec = compute_metrics(dataset_key, pred_len, h_root, BACKBONE)
    log_row["metric_sec"] = round(time.time() - t0, 1)
    if m_rec.get("status") not in ("ok", ""):
        log_row.update({"status": "fail",
                        "error": f"05: {m_rec.get('error', '')}"})
        return log_row, metric_row

    log_row.update({"status": "ok", "phase": "done",
                    "elapsed_sec": round(time.time() - t_all, 1)})
    metric_row.update({
        "family": m_rec.get("family", ""),
        "vus_pr": m_rec.get("vus_pr_D_w_z", ""),
        "vus_roc": m_rec.get("vus_roc_D_w_z", ""),
        "auc_pr": m_rec.get("auc_pr_D_w_z", ""),
        "auc_roc": m_rec.get("auroc_D_w_z", ""),
        "standard_f1": m_rec.get("standard_f1_D_w_z", ""),
        "pa_f1": m_rec.get("pa_f1_D_w_z", ""),
        "normal_mse": m_rec.get("normal_mse", ""),
        "normal_mae": m_rec.get("normal_mae", ""),
        "n_eval_rows": m_rec.get("n_eval_rows", ""),
        "status": m_rec.get("status", "ok"),
        "error": m_rec.get("error", ""),
    })
    return log_row, metric_row


def main():
    parser = argparse.ArgumentParser(description="Horizon ablation orchestrator")
    parser.add_argument("--pred-lens", type=int, nargs="+", default=[192, 336],
                        help="Prediction horizons to sweep. Default: 192 336.")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Subset of dataset keys. Default: all 200.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (H, key) pairs whose predictions+scores already exist.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for 02/03.")
    args = parser.parse_args()

    all_keys = list_available_dataset_keys()
    if args.only:
        keys = [k for k in args.only if k in set(all_keys)]
        missing = sorted(set(args.only) - set(all_keys))
        if missing:
            print(f"[warn] unknown keys: {missing}")
    else:
        keys = all_keys

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict] = []
    metric_rows: list[dict] = []

    for H in args.pred_lens:
        print(f"\n========== H={H}  ({len(keys)} candidate datasets) ==========")
        h_root = _h_root(H)
        h_root.mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(keys, 1):
            print(f"[H={H}] ({i}/{len(keys)}) {k}", flush=True)
            try:
                log_row, metric_row = run_one(H, k, args.batch_size, args.skip_existing)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                log_row = {col: "" for col in RUN_LOG_COLUMNS}
                log_row.update({"pred_len": H, "dataset_key": k,
                                "status": "fail",
                                "error": f"orchestrator: {type(exc).__name__}: {exc}"})
                metric_row = {col: "" for col in METRIC_COLUMNS}
                metric_row.update({"pred_len": H, "dataset_key": k,
                                   "status": "fail", "error": log_row["error"]})
            log_rows.append(log_row)
            metric_rows.append(metric_row)
            print(f"    -> {log_row['status']} ({log_row.get('elapsed_sec', '')}s)",
                  flush=True)

            # Incremental write — merge with any prior invocation's rows so
            # an interrupted run can be resumed with --skip-existing without
            # losing already-completed (H, key) pairs.
            log_merged = _merge_existing(ABLATION_ROOT / "run_log.csv",
                                         log_rows, RUN_LOG_COLUMNS)
            metric_merged = _merge_existing(ABLATION_ROOT / "per_dataset_metrics.csv",
                                            metric_rows, METRIC_COLUMNS)
            _write_csv(ABLATION_ROOT / "run_log.csv", log_merged, RUN_LOG_COLUMNS)
            _write_csv(ABLATION_ROOT / "per_dataset_metrics.csv",
                       metric_merged, METRIC_COLUMNS)
            _write_csv(h_root / "per_dataset_metrics.csv",
                       [r for r in metric_merged if str(r.get("pred_len")) == str(H)],
                       METRIC_COLUMNS)

    print(f"\nDone. Logs: {ABLATION_ROOT / 'run_log.csv'}")
    print(f"      Metrics: {ABLATION_ROOT / 'per_dataset_metrics.csv'}")


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _merge_existing(path: Path, new_rows: list[dict], cols: list[str]) -> list[dict]:
    """Merge new rows into any existing CSV, keyed by (pred_len, dataset_key).

    Lets a killed/resumed run preserve previously-completed (H, key) results
    rather than truncating the CSV to the current invocation's iterations.
    The newest row for a given (H, key) wins, so successful re-runs overwrite
    stale failures.
    """
    if not path.exists():
        return new_rows
    keep: list[dict] = []
    new_keys = {(str(r.get("pred_len", "")), str(r.get("dataset_key", "")))
                for r in new_rows}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (str(row.get("pred_len", "")), str(row.get("dataset_key", "")))
            if key in new_keys:
                continue
            keep.append({c: row.get(c, "") for c in cols})
    return keep + new_rows


if __name__ == "__main__":
    main()
