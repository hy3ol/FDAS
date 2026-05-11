"""
run_all.py — V13 end-to-end driver (backbone-pluggable).

Modes:
  --datasets KEY [KEY ...]  Train + infer for the listed dataset keys
                            (calls 01 → 02 → 03 per key, with --backbone).
  --all-keys                Train + infer for every dataset under V13/datasets/.
  --analyze                 Run analysis stage only (04 → 05 → 06 → 07),
                            scoped to the chosen backbone.
  --backbone NAME           Backbone to use across all stages. Default: iTransformer.

Typical workflows:
  # Train + infer iTransformer across every dataset (V13's original flow)
  python scripts/run_all.py --all-keys --skip-existing
  # Then run analysis for that backbone
  python scripts/run_all.py --analyze

  # Add another backbone (after registering it in model/__init__.py)
  python scripts/run_all.py --all-keys --backbone DLinear --skip-existing
  python scripts/run_all.py --analyze --backbone DLinear
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR.parent / "run_logs"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.run(cmd, env=env)
    return proc.returncode


def train_one_dataset(dataset_key: str, backbone: str) -> int:
    log = LOGS_DIR / backbone / f"{dataset_key}_train_infer.log"
    rc = _run([sys.executable, str(SCRIPT_DIR / "01_data_preparation.py"),
               "--dataset-key", dataset_key], log_path=log)
    if rc != 0:
        return rc
    rc = _run([sys.executable, str(SCRIPT_DIR / "02_train.py"),
               "--backbone", backbone], log_path=log)
    if rc != 0:
        return rc
    return _run([sys.executable, str(SCRIPT_DIR / "03_inference.py"),
                 "--backbone", backbone], log_path=log)


def already_done(key: str, backbone: str) -> bool:
    return (RESULTS_DIR / key / backbone / "predictions_test.npy").exists()


def run_analysis_pipeline(backbone: str) -> int:
    # Main analysis path: 04 (score: D_w_z) → 05 (per-dataset metrics CSV).
    # 06_cross_dataset.py (D_w_z vs D_w paired comparison) and
    # 07_visualization.py are no longer part of the production pipeline:
    # we standardized on D_w_z as the single production score, so the
    # baseline comparison and per-paper figures are deprecated. 06 still
    # lives in scripts/ for ad-hoc use; 07 was removed in this vintage.
    for script in [
        "04_score_compute.py",
        "05_metrics.py",
    ]:
        print(f"\n=== {script} (backbone={backbone}) ===")
        rc = _run([sys.executable, str(SCRIPT_DIR / script),
                   "--backbone", backbone])
        if rc != 0:
            return rc
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all-keys", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip dataset keys whose predictions_test.npy "
                             "already exists for the chosen backbone.")
    parser.add_argument("--backbone", type=str, default="iTransformer",
                        help="Backbone name (default: iTransformer). Must be "
                             "registered in V13/model/__init__.py.")
    args = parser.parse_args()

    if not (args.datasets or args.all_keys or args.analyze):
        parser.error("Specify --datasets, --all-keys, or --analyze.")

    keys: list[str] = []
    if args.all_keys:
        sys.path.append(str(SCRIPT_DIR))
        from artifact_paths import list_available_dataset_keys
        keys = list_available_dataset_keys()
    elif args.datasets:
        keys = args.datasets

    failures: list[str] = []
    skipped: list[str] = []
    for i, k in enumerate(keys, 1):
        print(f"\n{'='*70}\n[{i}/{len(keys)}] [{args.backbone}] Train + infer: {k}\n{'='*70}")
        if args.skip_existing and already_done(k, args.backbone):
            print(f"  SKIP (predictions_test.npy already exists for backbone={args.backbone})")
            skipped.append(k)
            continue
        rc = train_one_dataset(k, args.backbone)
        if rc != 0:
            print(f"  [{k}] FAILED (rc={rc})")
            failures.append(k)

    if keys:
        print(f"\nTrain+infer summary [{args.backbone}]: "
              f"trained={len(keys) - len(failures) - len(skipped)}, "
              f"skipped={len(skipped)}, failed={len(failures)}")
    if failures:
        print(f"Failed: {failures}")

    if args.analyze:
        print(f"\n{'='*70}\nAnalysis pipeline [{args.backbone}] (04 → 05 → 06 → 07)\n{'='*70}")
        rc = run_analysis_pipeline(args.backbone)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
