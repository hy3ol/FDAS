"""
Data Preparation Script for V11 (forward-looking).

Identical to V10 in mechanics — V11 reuses the TSB-AD-M aligned split logic
verbatim because the iTransformer backbone is unchanged. Differences land in
the analysis stage (00, 04-07).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from artifact_paths import (
    DATA_DIR,
    list_available_dataset_keys,
    resolve_dataset,
    save_data_metadata,
    save_dataset_bundle_meta,
)

# Configuration
LOOKBACK = 192
PRED_LEN = 96
TRAIN_RATIO_WITHIN_TRAINVAL = 0.70


def parse_filename(filename):
    """
    Parse TSB-AD-M filename to extract metadata.

    Format: {id}_{dataset}_{entity_id}_{entity_name}_tr_{train_size}_1st_{first_anomaly_index}.csv
    Example: 057_SMD_id_1_Facility_tr_4529_1st_4629.csv
    """
    parts = filename.replace('.csv', '').split('_')

    # Find positions of key markers
    tr_idx = parts.index('tr')
    first_idx = parts.index('1st')

    metadata = {
        'id': int(parts[0]),
        'dataset': parts[1],
        'train_size': int(parts[tr_idx + 1]),
        'first_anomaly_idx': int(parts[first_idx + 1]),
        'filename': filename
    }

    return metadata


def load_and_prepare_data(dataset_key):
    """Load the configured dataset and create train/val/test splits"""

    dataset_info = resolve_dataset(dataset_key)
    data_path = dataset_info['path']
    dataset_name = dataset_info['filename']

    print("="*60)
    metadata = parse_filename(dataset_name)
    print(f"V11 Data Preparation - {metadata['dataset']} Dataset")
    print(f"\n[Dataset Info]")
    print(f"  Key: {dataset_key}")
    print(f"  Name: {dataset_name}")
    print(f"  Path: {data_path}")
    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Train size from filename: {metadata['train_size']}")
    print(f"  First anomaly index from filename: {metadata['first_anomaly_idx']}")

    # Load data
    print(f"\n[1] Loading data from TSB-AD-M")
    df = pd.read_csv(data_path)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Total rows: {len(df)}")

    # Extract values and labels
    # TSB-AD-M format: last column is 'Label' (or 'is_anomaly'), rest are sensor values
    label_col = 'Label' if 'Label' in df.columns else 'is_anomaly'
    value_cols = [col for col in df.columns if col != label_col]

    values = df[value_cols].values  # (T, C)
    labels = df[label_col].values    # (T,)

    num_channels = len(value_cols)
    total_timesteps = len(values)

    print(f"  Shape: {values.shape}")
    print(f"  Channels: {num_channels}")
    print(f"  Anomaly ratio: {labels.mean():.2%}")

    # Split data
    print(f"\n[2] Creating train/val/test splits")
    anomaly_indices = np.where(labels == 1)[0]
    first_anomaly_idx = int(anomaly_indices[0]) if len(anomaly_indices) > 0 else len(labels)

    required_span = LOOKBACK + PRED_LEN

    # Official split boundary from filename metadata.
    official_train_end = min(metadata['train_size'], total_timesteps)
    if official_train_end < required_span:
        raise ValueError(
            "Official train segment is shorter than the minimum required window span "
            f"(train={official_train_end}, required={required_span})."
        )

    # Internal train/val split inside official train.
    # Prefer disjoint 80/20 split; fallback to overlapping val tail if too short.
    train_end_nominal = int(official_train_end * TRAIN_RATIO_WITHIN_TRAINVAL)
    train_end_nominal = max(train_end_nominal, required_span)
    disjoint_val_len = official_train_end - train_end_nominal
    uses_overlap_val = disjoint_val_len < required_span

    if uses_overlap_val:
        # Keep official train/test boundary fixed; create val from train tail.
        train_end = official_train_end
        val_start = official_train_end - required_span
    else:
        train_end = train_end_nominal
        val_start = train_end

    train_val_end = official_train_end
    test_start = official_train_end
    train_data = values[:train_end]
    train_labels = labels[:train_end]
    val_data = values[val_start:train_val_end]
    val_labels = labels[val_start:train_val_end]
    test_data = values[test_start:]
    test_labels = labels[test_start:]

    if len(val_data) < required_span:
        raise ValueError(
            "Validation segment is shorter than required window span after split. "
            f"(val={len(val_data)}, required={required_span})"
        )
    if len(test_data) < required_span:
        raise ValueError(
            "Test segment is shorter than required window span after split. "
            f"(test={len(test_data)}, required={required_span})"
        )

    if official_train_end > first_anomaly_idx:
        print("  [Warn] Official train boundary is after first anomaly index.")
        print("         This dataset appears to include anomaly points in train.")

    print(f"  Official train end (from filename tr): {official_train_end}")
    print(f"  Verified first anomaly index: {first_anomaly_idx}")
    print(f"  Required minimum span per split: {required_span}")
    print(f"  Train+Val end: {train_val_end}")
    print(f"  Train end: {train_end}")
    print(f"  Val start: {val_start}")
    print(f"  Val mode: {'overlap-tail' if uses_overlap_val else 'disjoint'}")
    print(f"  Train: {train_data.shape} (anomaly: {train_labels.mean():.2%})")
    print(f"  Val:   {val_data.shape} (anomaly: {val_labels.mean():.2%})")
    print(f"  Test:  {test_data.shape} (anomaly: {test_labels.mean():.2%})")
    print(f"  Test normal prefix before anomaly: {max(0, first_anomaly_idx - test_start)}")

    # Normalize using train statistics
    print(f"\n[3] Normalizing data")
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_norm = scaler.transform(train_data)
    val_data_norm = scaler.transform(val_data)
    test_data_norm = scaler.transform(test_data)

    print(f"  Train mean: {train_data_norm.mean():.6f}, std: {train_data_norm.std():.6f}")
    print(f"  Val mean:   {val_data_norm.mean():.6f}, std: {val_data_norm.std():.6f}")
    print(f"  Test mean:  {test_data_norm.mean():.6f}, std: {test_data_norm.std():.6f}")

    # Save data
    print(f"\n[4] Saving processed data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(DATA_DIR / "train_data.npy", train_data_norm)
    np.save(DATA_DIR / "train_labels.npy", train_labels)
    np.save(DATA_DIR / "val_data.npy", val_data_norm)
    np.save(DATA_DIR / "val_labels.npy", val_labels)
    np.save(DATA_DIR / "test_data.npy", test_data_norm)
    np.save(DATA_DIR / "test_labels.npy", test_labels)

    # Save metadata
    metadata_out = {
        'dataset_name': dataset_name,
        'dataset': metadata['dataset'],
        'dataset_key': dataset_key,
        'dataset_relative_path': dataset_info['relative_path'],
        'dataset_source_path': str(data_path),
        'num_channels': num_channels,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'total_timesteps': total_timesteps,
        'anomaly_ratio': float(test_labels.mean()),
        'first_anomaly_idx': first_anomaly_idx,
        'official_train_end': official_train_end,
        'train_end': train_end,
        'train_val_end': train_val_end,
        'val_start': val_start,
        'test_start': test_start,
        'val_mode': 'overlap-tail' if uses_overlap_val else 'disjoint',
        # numpy .sum() returns np.int64 — keep int() for JSON serialization
        'train_anomalies': int(train_labels.sum()),
        'val_anomalies': int(val_labels.sum()),
        'test_anomalies': int(test_labels.sum()),

        # Model hyperparameters used by the V11 iTransformer pipeline
        'lookback': LOOKBACK,
        'pred_len': PRED_LEN,

        # Scaler parameters
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    save_data_metadata(metadata_out)

    # Also persist per-dataset bundle metadata under results/{dataset_key}/
    # so 04_score_compute.py / 05_metrics.py see the exact split + scaler
    # 01 used (instead of re-deriving them). Subset of metadata_out, with
    # scaler_train_end aliased to the canonical name used in score_utils.
    bundle_meta = {
        'dataset_key': dataset_key,
        'dataset_name': dataset_name,
        'official_train_end': official_train_end,
        'scaler_train_end': train_end,
        'train_end': train_end,
        'val_start': val_start,
        'test_start': test_start,
        'val_mode': 'overlap-tail' if uses_overlap_val else 'disjoint',
        'total_timesteps': total_timesteps,
        'num_channels': num_channels,
        'lookback': LOOKBACK,
        'pred_len': PRED_LEN,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }
    save_dataset_bundle_meta(dataset_key, bundle_meta)

    print(f"  ✓ Saved to {DATA_DIR}/")
    print(f"    - train_data.npy: {train_data_norm.shape}")
    print(f"    - train_labels.npy: {train_labels.shape}")
    print(f"    - val_data.npy: {val_data_norm.shape}")
    print(f"    - val_labels.npy: {val_labels.shape}")
    print(f"    - test_data.npy: {test_data_norm.shape}")
    print(f"    - test_labels.npy: {test_labels.shape}")
    print(f"    - metadata.json")
    print(f"    - results/{dataset_key}/bundle_meta.json")

    print("\n" + "="*60)
    print("Data Preparation Complete!")
    print("="*60)

    return metadata_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-key',
        type=str,
        default=None,
        help='Dataset key under V11/datasets, e.g., SWaT, PSM, SMD_id_1'
    )
    args = parser.parse_args()

    dataset_key = args.dataset_key
    if dataset_key is None:
        keys = list_available_dataset_keys()
        if len(keys) == 1:
            dataset_key = keys[0]
            print(f"[Info] Single dataset found. Using --dataset-key {dataset_key}")
        else:
            raise ValueError(
                "Multiple datasets are available. "
                "Please provide --dataset-key. "
                f"Available: {', '.join(keys)}"
            )

    metadata = load_and_prepare_data(dataset_key=dataset_key)
