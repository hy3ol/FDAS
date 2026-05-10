"""migrate_to_backbone_layout.py — one-time mv for the V13 refactor.

Pre-refactor layout:
  V13/models/<dataset_key>/{best_model.pth, checkpoint.pth, train_config.json,
                            training_history.json}
  V13/results/<dataset_key>/{predictions_*.npy, inference_metadata.json,
                             scores.parquet, scores.csv, scores_per_ch.npz,
                             test_labels.npy}

Post-refactor layout:
  V13/models/<dataset_key>/iTransformer/{best_model.pth, ...}
  V13/results/<dataset_key>/iTransformer/{predictions_*.npy, scores.parquet, ...}
  V13/results/<dataset_key>/bundle_meta.json   ← stays at dataset level

Usage:
  python scripts/migrate_to_backbone_layout.py            # DRY RUN (default)
  python scripts/migrate_to_backbone_layout.py --execute  # actually move

Idempotent: skips files already moved (e.g. partial prior run, or new training
that already wrote into the backbone subdir). Always reports what it would do
before doing it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
MODELS_ROOT = BASE / "models"
RESULTS_ROOT = BASE / "results"
BACKBONE = "iTransformer"

# Files that move from <dataset>/ to <dataset>/<backbone>/
MODEL_FILES = ["best_model.pth", "checkpoint.pth",
               "train_config.json", "training_history.json"]
RESULT_FILES = [
    "predictions_train.npy", "predictions_val.npy", "predictions_test.npy",
    "inference_metadata.json", "test_labels.npy",
    "scores.parquet", "scores.csv", "scores_per_ch.npz",
]
# Files that STAY at the dataset level (never moved):
RESULT_DATASET_LEVEL = ["bundle_meta.json"]
# Cross-dataset analysis dirs that are never per-dataset → not touched:
SKIP_RESULT_DIRS = {"04_metrics", "05_cross_dataset", "00_dataset_filter",
                    "06_lead_time", "figures", "iTransformer"}


def _plan_moves(root: Path, files: list[str]) -> list[tuple[Path, Path]]:
    """For each <dataset>/ subdir, plan moves of legacy files into
    <dataset>/<BACKBONE>/."""
    moves: list[tuple[Path, Path]] = []
    if not root.exists():
        return moves
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in SKIP_RESULT_DIRS:
            continue
        bb_dir = d / BACKBONE
        for fname in files:
            src = d / fname
            if not src.exists():
                continue
            dst = bb_dir / fname
            if dst.exists():
                # Already migrated (or new file) — skip silently to keep runs idempotent
                continue
            moves.append((src, dst))
    return moves


def _summarize(label: str, moves: list[tuple[Path, Path]]) -> None:
    if not moves:
        print(f"  {label}: nothing to move")
        return
    print(f"  {label}: {len(moves)} file(s)")
    # Count distinct source dirs
    src_dirs = sorted({m[0].parent.name for m in moves})
    print(f"    across {len(src_dirs)} dataset(s); first 5: {src_dirs[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform the moves. Default is dry-run (prints plan only).",
    )
    args = parser.parse_args()

    print(f"V13 backbone-layout migration  (target backbone = '{BACKBONE}')")
    print(f"  MODELS_ROOT  = {MODELS_ROOT}")
    print(f"  RESULTS_ROOT = {RESULTS_ROOT}")
    print()

    model_moves = _plan_moves(MODELS_ROOT, MODEL_FILES)
    result_moves = _plan_moves(RESULTS_ROOT, RESULT_FILES)

    print("[plan]")
    _summarize("models/  ", model_moves)
    _summarize("results/ ", result_moves)

    total = len(model_moves) + len(result_moves)
    if total == 0:
        print("\nNothing to migrate. (Already in new layout, or no artifacts present.)")
        return 0

    if not args.execute:
        print("\nDry run only — re-run with --execute to perform the moves.")
        return 0

    print("\n[execute]")
    n_done = 0
    for src, dst in model_moves + result_moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n_done += 1
    print(f"  ✓ moved {n_done} file(s) into {BACKBONE}/ subdirectories")

    # Verify
    print("\n[verify] sample paths after migration:")
    for d_root, files in [(MODELS_ROOT, MODEL_FILES), (RESULTS_ROOT, RESULT_FILES)]:
        if not d_root.exists():
            continue
        sample = next((d for d in sorted(d_root.iterdir())
                       if d.is_dir() and d.name not in SKIP_RESULT_DIRS), None)
        if sample is None:
            continue
        bb = sample / BACKBONE
        present = [f for f in files if (bb / f).exists()]
        leftover = [f for f in files if (sample / f).exists()]
        print(f"  {sample}: {len(present)} in {BACKBONE}/, {len(leftover)} legacy "
              f"(should be 0)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
