"""
V11 artifact path utilities.

Layout:
  results/{dataset_key}/                — per-dataset products
      predictions_test.npy
      predictions_train.npy
      test_labels.npy
      inference_metadata.json
      scores.parquet                    — D_w_fwd + base time series
      per_horizon.parquet               — per-(t, k) D_w(t+k; t)
      traj.parquet                      — pre-anomaly trajectories
      heatmap.parquet                   — pre-anomaly heatmap data (long form)
      figures/                          — per-dataset PNGs (V8-style)

  results/00_dataset_filter/            — cross-dataset filter
  results/06_lead_time/                 — cross-dataset lead-time tables
  results/figure_supporting_*.png       — cross-dataset summary figures
  results/statistics_table.md           — cross-dataset summary
  results/selected_cases.csv            — pointer index for poster cases

The single-active-dataset pattern (data/metadata.json) is preserved so that
01_data_preparation / 02_train / 03_inference can run dataset-by-dataset.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
DATASETS_DIR = BASE_DIR / 'datasets'
RESULTS_ROOT = BASE_DIR / 'results'
MODELS_ROOT = BASE_DIR / 'models'

# Cross-dataset (analysis) result subdirectories
DATASET_FILTER_DIR = RESULTS_ROOT / '00_dataset_filter'
LEAD_TIME_DIR = RESULTS_ROOT / '06_lead_time'
FIGURES_DIR = RESULTS_ROOT

# Per-dataset filenames (used inside results/{dataset_key}/)
SCORES_FILENAME = 'scores'                 # .parquet (or .csv fallback)
PER_HORIZON_FILENAME = 'per_horizon'
TRAJ_FILENAME = 'traj'
HEATMAP_FILENAME = 'heatmap'


def _parse_dataset_filename(path):
    name = Path(path).name
    stem = Path(path).stem
    parts = stem.split('_')

    dataset = parts[1] if len(parts) > 1 else stem

    record = {
        'filename': name,
        'dataset': dataset,
        'file_id': None,
        'entity_id': None,
    }

    if parts and parts[0].isdigit():
        record['file_id'] = int(parts[0])

    if 'id' in parts:
        idx = parts.index('id')
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            record['entity_id'] = int(parts[idx + 1])

    if 'tr' in parts:
        idx = parts.index('tr')
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            record['train_size_from_name'] = int(parts[idx + 1])

    if '1st' in parts:
        idx = parts.index('1st')
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            record['first_anomaly_from_name'] = int(parts[idx + 1])

    return record


def _build_dataset_registry():
    csv_files = sorted(DATASETS_DIR.rglob('*.csv')) if DATASETS_DIR.exists() else []

    records = []
    for csv_file in csv_files:
        rec = _parse_dataset_filename(csv_file)
        rec['path'] = csv_file
        rec['relative_path'] = csv_file.relative_to(DATASETS_DIR).as_posix()
        records.append(rec)

    grouped = defaultdict(list)
    for rec in records:
        grouped[rec['dataset']].append(rec)

    registry = {}
    for dataset, items in grouped.items():
        dataset_has_multiple_files = len(items) > 1

        for idx, rec in enumerate(items, start=1):
            if dataset_has_multiple_files:
                if rec.get('entity_id') is not None:
                    key = f"{dataset}_id_{rec['entity_id']}"
                elif rec.get('file_id') is not None:
                    key = f"{dataset}_{rec['file_id']}"
                else:
                    key = f"{dataset}_{idx}"
            else:
                key = dataset

            if key in registry:
                suffix = 2
                while f'{key}_{suffix}' in registry:
                    suffix += 1
                key = f'{key}_{suffix}'

            rec = dict(rec)
            rec['dataset_key'] = key
            registry[key] = rec

    return registry


def list_available_dataset_keys():
    return sorted(_build_dataset_registry().keys())


def resolve_dataset(dataset_key):
    registry = _build_dataset_registry()
    if dataset_key not in registry:
        available = ', '.join(sorted(registry.keys()))
        raise KeyError(
            f"Unknown dataset key `{dataset_key}`. Available keys: {available}"
        )
    return registry[dataset_key]


def load_data_metadata():
    with open(DATA_DIR / 'metadata.json', 'r') as f:
        return json.load(f)


def save_data_metadata(metadata):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)


def save_dataset_bundle_meta(dataset_key, bundle_meta):
    """Persist per-dataset split + scaler metadata under results/{dataset_key}/.

    Produced by 01_data_preparation.py and read by score_utils.prepare_
    dataset_bundle so 04/05 use the same split + scaler that 01 used,
    rather than re-deriving them from filename heuristics. Survives
    DATA_DIR being overwritten by a subsequent dataset's preparation.
    """
    out_dir = RESULTS_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'bundle_meta.json', 'w') as f:
        json.dump(bundle_meta, f, indent=4)


def _resolve_dataset_key_from_metadata(metadata):
    if 'dataset_key' in metadata:
        return metadata['dataset_key']

    dataset_name = metadata.get('dataset')
    if dataset_name is None:
        raise KeyError("`dataset` is missing in data metadata.")

    registry = _build_dataset_registry()
    if dataset_name in registry:
        return dataset_name

    candidates = [k for k, v in registry.items() if v.get('dataset') == dataset_name]
    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        "Could not infer dataset key from metadata. "
        "Please regenerate metadata with 01_data_preparation.py --dataset-key ..."
    )


def get_current_dataset_key():
    metadata = load_data_metadata()
    return _resolve_dataset_key_from_metadata(metadata)


def get_models_dir(dataset_key=None, create=True):
    key = dataset_key or get_current_dataset_key()
    models_dir = MODELS_ROOT / key
    if create:
        models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_dataset_results_dir(dataset_key=None, create=True):
    """Per-dataset folder. Holds predictions, scores, trajectories, heatmaps, figures."""
    key = dataset_key or get_current_dataset_key()
    out_dir = RESULTS_ROOT / key
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# Backwards-compatible alias used by 03_inference.py
def get_outputs_dir(dataset_key=None, create=True):
    return get_dataset_results_dir(dataset_key=dataset_key, create=create)


def get_dataset_figures_dir(dataset_key=None, create=True):
    d = get_dataset_results_dir(dataset_key=dataset_key, create=create) / 'figures'
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def get_dataset_score_path(dataset_key=None) -> Path:
    return get_dataset_results_dir(dataset_key=dataset_key) / SCORES_FILENAME


def get_dataset_per_horizon_path(dataset_key=None) -> Path:
    return get_dataset_results_dir(dataset_key=dataset_key) / PER_HORIZON_FILENAME


def get_dataset_traj_path(dataset_key=None) -> Path:
    return get_dataset_results_dir(dataset_key=dataset_key) / TRAJ_FILENAME


def get_dataset_heatmap_path(dataset_key=None) -> Path:
    return get_dataset_results_dir(dataset_key=dataset_key) / HEATMAP_FILENAME


def get_results_root(create=True):
    if create:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


def ensure_analysis_subdirs():
    """Create cross-dataset analysis subdirectories."""
    for d in [
        DATASET_FILTER_DIR,
        LEAD_TIME_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
