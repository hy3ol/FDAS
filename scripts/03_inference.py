"""
Inference Script for V11 (forward-looking).

Generate predictions with stride=1 sliding windows.
Identical mechanics to V10 — V11's forward score is computed downstream
in 04_score_compute.py.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.iTransformer import Model
from artifact_paths import DATA_DIR, get_models_dir, get_outputs_dir, load_data_metadata


class Config:
    """Minimal config for inference"""
    def __init__(self, metadata):
        self.seq_len = metadata['lookback']
        self.pred_len = metadata['pred_len']
        self.enc_in = metadata['num_channels']
        self.dec_in = metadata['num_channels']
        self.c_out = metadata['num_channels']

        # iTransformer params (must match training)
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False
        # V13 default; the actual value is overwritten by train_config.json /
        # checkpoint config (apply_saved_model_config) so train and inference
        # cannot diverge in practice.
        self.use_norm = True

        # Embedding
        self.embed = 'timeF'
        self.freq = 'h'
        self.factor = 1
        self.class_strategy = 'projection'

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_saved_model_config(config, saved_config):
    if not isinstance(saved_config, dict):
        return config

    fields = [
        'seq_len',
        'pred_len',
        'enc_in',
        'dec_in',
        'c_out',
        'd_model',
        'n_heads',
        'e_layers',
        'd_ff',
        'dropout',
        'activation',
        'output_attention',
        'use_norm',
        'embed',
        'freq',
        'factor',
        'class_strategy',
    ]
    for field in fields:
        if field in saved_config:
            setattr(config, field, saved_config[field])
    return config


def resolve_checkpoint_path(models_dir):
    checkpoint_path = os.path.join(models_dir, 'checkpoint.pth')
    best_model_path = os.path.join(models_dir, 'best_model.pth')

    if os.path.exists(checkpoint_path):
        return checkpoint_path
    if os.path.exists(best_model_path):
        return best_model_path

    raise FileNotFoundError(
        f'No checkpoint found in {models_dir}. '
        'Expected checkpoint.pth or best_model.pth.'
    )


class SlidingWindowDataset(Dataset):
    """Stride=1 sliding window dataset for inference.

    V13 patch: extend the sliding range so anchors up through T-2 are
    processed (vs T-H-1 originally). Forecasts from the trailing H-1
    anchors have H-step horizons that partially extend past the data, but
    the lead positions targeting t ∈ (T-H, T-1] fall within the data and
    are required by D_w to cover the full forward-computable evaluable
    range t ∈ [L+H-1, T-1].
    """

    def __init__(self, data, lookback, pred_len):
        """
        Args:
            data: (T, C) data
            lookback: L
            pred_len: H
        """
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len

        # Number of windows: input must fit in `data` (last input ends at
        # T-1, anchored at start = T-L-1 inclusive); the H-step horizon may
        # extend past T but those positions are stored anyway and simply
        # not consumed by D_w.
        self.length = max(len(data) - lookback, 0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            x: (lookback, C) input
            window_idx: scalar (for tracking)
        """
        x = self.data[idx:idx+self.lookback]
        x_mark = np.zeros((self.lookback, 1))

        # Dummy y_mark (required by model but not used in inference)
        y_mark = np.zeros((self.pred_len, 1))

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y_mark),
            idx
        )


def generate_predictions(model, data, config, desc="Inference"):
    """
    Generate predictions for given data.

    Args:
        model: trained iTransformer
        data: (T, C) numpy array
        config: Config object
        desc: description for progress bar

    Returns:
        predictions: (N, H, C) array
    """
    dataset = SlidingWindowDataset(data, config.seq_len, config.pred_len)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device.type == 'cuda'
    )

    N = len(dataset)
    H = config.pred_len
    C = config.enc_in

    predictions = np.zeros((N, H, C), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for x_enc, x_mark, y_mark, indices in pbar:
            x_enc = x_enc.to(config.device)
            # Forward pass
            y_pred = model(x_enc, None, None, None)  # (B, H, C)

            # Store predictions
            y_pred_np = y_pred.cpu().numpy()
            for i, idx in enumerate(indices):
                predictions[idx] = y_pred_np[i]

    return predictions


def compute_window_error_metrics(predictions, data, lookback, pred_len, chunk_size=2048):
    """
    Compute overall window-averaged MSE/MAE across all windows/horizons/channels.

    Args:
        predictions: (N, H, C)
        data: (T, C)
        lookback: input sequence length L
        pred_len: prediction horizon H
        chunk_size: chunk size for memory-safe reduction

    Returns:
        mse, mae (float, float)
    """
    N, H, C = predictions.shape
    if H != pred_len:
        raise ValueError(
            f"pred_len mismatch: predictions H={H}, expected {pred_len}"
        )
    if data.shape[1] != C:
        raise ValueError(
            f"channel mismatch: predictions C={C}, data C={data.shape[1]}"
        )

    # (T-H+1, C, H) -> (T-H+1, H, C)
    gt_all = sliding_window_view(data, window_shape=pred_len, axis=0)
    gt_all = np.transpose(gt_all, (0, 2, 1))

    gt_start = lookback
    gt_end = lookback + N
    if gt_end > gt_all.shape[0]:
        raise ValueError(
            "Ground-truth window range exceeds available windows. "
            f"(gt_end={gt_end}, available={gt_all.shape[0]})"
        )

    gt_windows = gt_all[gt_start:gt_end]
    if gt_windows.shape != predictions.shape:
        raise ValueError(
            f"shape mismatch: pred={predictions.shape}, gt={gt_windows.shape}"
        )

    total_sq = 0.0
    total_abs = 0.0
    total_count = 0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pred_chunk = predictions[start:end].astype(np.float64, copy=False)
        gt_chunk = gt_windows[start:end].astype(np.float64, copy=False)
        diff = pred_chunk - gt_chunk

        total_sq += np.sum(diff * diff, dtype=np.float64)
        total_abs += np.sum(np.abs(diff), dtype=np.float64)
        total_count += diff.size

    mse = float(total_sq / max(total_count, 1))
    mae = float(total_abs / max(total_count, 1))
    return mse, mae


def run_inference():
    print("="*60)
    print("V11 Inference - Generating Train & Test Predictions")
    print("="*60)

    # Load metadata
    print("\n[1] Loading metadata and model")
    metadata = load_data_metadata()

    models_dir = get_models_dir(create=False)
    outputs_dir = get_outputs_dir()

    config = Config(metadata)
    train_config_path = os.path.join(models_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            saved_train_config = json.load(f)
        config = apply_saved_model_config(config, saved_train_config)
        print(f"    Loaded train config: {train_config_path}")
    print(f"    Lookback (L): {config.seq_len}")
    print(f"    Pred_len (H): {config.pred_len}")
    print(f"    Channels (C): {config.enc_in}")
    print(f"    Device: {config.device}")
    print(f"    Model dir: {models_dir}")
    print(f"    Output dir: {outputs_dir}")

    # Load data
    train_data = np.load(DATA_DIR / 'train_data.npy')
    val_data = np.load(DATA_DIR / 'val_data.npy')
    test_data = np.load(DATA_DIR / 'test_data.npy')
    test_labels = np.load(DATA_DIR / 'test_labels.npy')

    print(f"    Train data: {train_data.shape}")
    print(f"    Val data:   {val_data.shape}")
    print(f"    Test data:  {test_data.shape}")
    print(f"    Test labels: {test_labels.shape}")

    # Load model
    print("\n[2] Loading trained model")
    checkpoint_path = resolve_checkpoint_path(models_dir)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    config = apply_saved_model_config(config, checkpoint.get('config', {}))
    model = Model(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Model loaded from epoch {checkpoint['epoch']}")
    print(f"    Best val loss: {checkpoint['val_loss']:.6f}")

    # Generate train predictions (for calibration)
    # V13: N = len(data) - lookback (was: - lookback - pred_len + 1).
    # The trailing pred_len-1 windows have H-step horizons that extend past
    # the data end; they're stored anyway so D_w can cover the full
    # forward-computable evaluable range t ∈ [L+H-1, T-1].
    print("\n[3] Generating TRAIN predictions (for calibration)")
    N_train = len(train_data) - config.seq_len
    N_train_full = len(train_data) - config.seq_len - config.pred_len + 1
    T_train = len(train_data)
    print(f"    Train windows (N_train): {N_train} (full-GT subset: {N_train_full})")
    print(f"    Train prediction shape: (N={N_train}, H={config.pred_len}, C={config.enc_in})")

    predictions_train = generate_predictions(model, train_data, config, desc="Train Inference")
    print(f"    ✓ Train predictions: {predictions_train.shape}")

    # Generate VAL predictions (for clean baseline normalization in 04).
    # Val was held out from gradient updates, so D_w_c on val reflects
    # generalization-gap variance — the right reference for z-scoring test
    # D_w_c without the train-overfit bias that an in-sample baseline has.
    print("\n[3b] Generating VAL predictions (for clean baseline)")
    if len(val_data) >= config.seq_len + config.pred_len:
        N_val = len(val_data) - config.seq_len
        N_val_full = len(val_data) - config.seq_len - config.pred_len + 1
        print(f"    Val windows (N_val): {N_val} (full-GT subset: {N_val_full})")
        predictions_val = generate_predictions(model, val_data, config, desc="Val Inference")
        print(f"    ✓ Val predictions: {predictions_val.shape}")
    else:
        N_val = 0
        N_val_full = 0
        predictions_val = np.empty((0, config.pred_len, config.enc_in), dtype=np.float32)
        print(f"    Val too short ({len(val_data)} < {config.seq_len + config.pred_len}); "
              f"empty predictions_val.npy will fall back to train baseline downstream.")

    # Generate test predictions
    print("\n[4] Generating TEST predictions")
    N_test = len(test_data) - config.seq_len
    N_test_full = len(test_data) - config.seq_len - config.pred_len + 1
    T_test = len(test_data)
    print(f"    Test windows (N_test): {N_test} (full-GT subset: {N_test_full})")
    print(f"    Test prediction shape: (N={N_test}, H={config.pred_len}, C={config.enc_in})")

    predictions_test = generate_predictions(model, test_data, config, desc="Test Inference")
    print(f"    ✓ Test predictions: {predictions_test.shape}")

    # Calculate prediction error (verification)
    print("\n[5] Calculating prediction error (verification)")

    # First-window error (quick sanity check)
    first_pred_train = predictions_train[0]  # (H, C)
    first_gt_train = train_data[config.seq_len:config.seq_len+config.pred_len]
    mse_train_first = np.mean((first_pred_train - first_gt_train) ** 2)
    mae_train_first = np.mean(np.abs(first_pred_train - first_gt_train))

    first_pred_test = predictions_test[0]
    first_gt_test = test_data[config.seq_len:config.seq_len+config.pred_len]
    mse_test_first = np.mean((first_pred_test - first_gt_test) ** 2)
    mae_test_first = np.mean(np.abs(first_pred_test - first_gt_test))

    # Overall window-averaged error (main metric) — only first N_*_full rows
    # have a fully in-bounds H-step ground-truth window, so slice to those
    # rows when computing the sanity MSE/MAE.
    mse_train, mae_train = compute_window_error_metrics(
        predictions_train[:N_train_full], train_data, config.seq_len, config.pred_len
    )
    mse_test, mae_test = compute_window_error_metrics(
        predictions_test[:N_test_full], test_data, config.seq_len, config.pred_len
    )

    print(f"    Train first window:   MSE={mse_train_first:.6f}, MAE={mae_train_first:.6f}")
    print(f"    Train overall windows: MSE={mse_train:.6f}, MAE={mae_train:.6f}")
    print(f"    Test first window:    MSE={mse_test_first:.6f}, MAE={mae_test_first:.6f}")
    print(f"    Test overall windows:  MSE={mse_test:.6f}, MAE={mae_test:.6f}")

    # Save predictions
    print("\n[6] Saving outputs")
    np.save(os.path.join(outputs_dir, 'predictions_train.npy'), predictions_train)
    np.save(os.path.join(outputs_dir, 'predictions_val.npy'), predictions_val)
    np.save(os.path.join(outputs_dir, 'predictions_test.npy'), predictions_test)
    np.save(os.path.join(outputs_dir, 'test_labels.npy'), test_labels)

    # Save inference metadata
    inference_meta = {
        'N_train': int(N_train),
        'N_test': int(N_test),
        'H': int(config.pred_len),
        'C': int(config.enc_in),
        'T_train': int(T_train),
        'T_test': int(T_test),
        'lookback': int(config.seq_len),
        'train_first_window_mse': float(mse_train_first),
        'train_first_window_mae': float(mae_train_first),
        'test_first_window_mse': float(mse_test_first),
        'test_first_window_mae': float(mae_test_first),
        'train_mse': float(mse_train),
        'train_mae': float(mae_train),
        'test_mse': float(mse_test),
        'test_mae': float(mae_test),
        'model_epoch': int(checkpoint['epoch']),
        'model_val_loss': float(checkpoint['val_loss'])
    }

    with open(os.path.join(outputs_dir, 'inference_metadata.json'), 'w') as f:
        json.dump(inference_meta, f, indent=4)

    print(f"    ✓ Saved to {outputs_dir}/")
    print(f"      - predictions_train.npy: {predictions_train.shape}")
    print(f"      - predictions_test.npy: {predictions_test.shape}")
    print(f"      - test_labels.npy: {test_labels.shape}")
    print(f"      - inference_metadata.json")

    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    print("\n✓ Train predictions will be used for calibration (μ_D^w)")
    print("✓ Test predictions will be used for anomaly detection")

    return predictions_train, predictions_test, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V11 inference with a trained iTransformer checkpoint")
    parser.parse_args()
    predictions_train, predictions_test, labels = run_inference()
