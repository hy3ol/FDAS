"""
V13 inference script — backbone-pluggable.

Generate (N, H, C) predictions on train/val/test with stride=1 sliding
windows. The backbone is selected by `--backbone <name>` (default:
iTransformer); architecture is rebuilt from train_config.json + checkpoint
config so a checkpoint trained on backbone X is consumed by the same X.
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

from model import get_backbone, list_backbones, DEFAULT_BACKBONE
from artifact_paths import (
    DATA_DIR, get_models_dir, get_outputs_dir, load_data_metadata,
)
from config_factory import build_config, apply_saved_config, load_train_config


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

    def __init__(self, data, lookback, pred_len, mark_dim=1):
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len
        self.mark_dim = mark_dim
        self.length = max(len(data) - lookback, 0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        # Anomaly setup ignores time semantics; mark is zeros. Shape just
        # has to match the backbone's embedding layer (see BackboneSpec.mark_dim).
        x_mark = np.zeros((self.lookback, self.mark_dim))
        y_mark = np.zeros((self.pred_len, self.mark_dim))
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y_mark),
            idx,
        )


def generate_predictions(model, data, config, backbone_spec, desc="Inference",
                          batch_size=64):
    """Generate (N, H, C) predictions for given data.

    Args:
        model: trained backbone nn.Module
        data: (T, C) numpy array
        config: Config (used for seq_len, pred_len, enc_in, device)
        backbone_spec: BackboneSpec — controls forward signature dispatch
        desc: tqdm description
        batch_size: inference DataLoader batch size (default 64; lower for OOM)

    Returns:
        (N, H, C) array; N = T - L (V13 inference patch).
    """
    dataset = SlidingWindowDataset(data, config.seq_len, config.pred_len,
                                    mark_dim=backbone_spec.mark_dim)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=config.device.type == 'cuda',
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
            x_mark = x_mark.to(config.device)
            y_mark = y_mark.to(config.device)
            y_pred = backbone_spec.call_forward(model, x_enc, x_mark, None, y_mark)

            y_pred_np = y_pred.cpu().numpy()
            for i, idx in enumerate(indices):
                predictions[idx] = y_pred_np[i]

    return predictions


def compute_window_error_metrics(predictions, data, lookback, pred_len, chunk_size=2048):
    """Window-averaged MSE/MAE across all windows/horizons/channels."""
    N, H, C = predictions.shape
    if H != pred_len:
        raise ValueError(f"pred_len mismatch: predictions H={H}, expected {pred_len}")
    if data.shape[1] != C:
        raise ValueError(f"channel mismatch: predictions C={C}, data C={data.shape[1]}")

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

    total_sq = total_abs = 0.0
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


def run_inference(backbone_name: str, batch_size: int = 64):
    backbone_spec = get_backbone(backbone_name)

    print("=" * 60)
    print(f"V13 Inference — backbone: {backbone_name}")
    print("=" * 60)

    print("\n[1] Loading metadata and model")
    metadata = load_data_metadata()
    models_dir = get_models_dir(backbone=backbone_name, create=False)
    outputs_dir = get_outputs_dir(backbone=backbone_name)

    config = build_config(metadata, backbone_spec)

    saved_train_config = load_train_config(models_dir)
    if saved_train_config is not None:
        config = apply_saved_config(config, saved_train_config)
        print(f"    Loaded train config: {models_dir / 'train_config.json'}")
        # Sanity: train_config's backbone field should match what we're loading
        saved_backbone = saved_train_config.get("backbone")
        if saved_backbone and saved_backbone != backbone_name:
            print(f"    [warn] train_config.backbone={saved_backbone} != "
                  f"--backbone {backbone_name}. Using --backbone for model build.")
    print(f"    Lookback (L): {config.seq_len}")
    print(f"    Pred_len (H): {config.pred_len}")
    print(f"    Channels (C): {config.enc_in}")
    print(f"    Device: {config.device}")
    print(f"    Model dir: {models_dir}")
    print(f"    Output dir: {outputs_dir}")

    train_data = np.load(DATA_DIR / 'train_data.npy')
    val_data = np.load(DATA_DIR / 'val_data.npy')
    test_data = np.load(DATA_DIR / 'test_data.npy')
    test_labels = np.load(DATA_DIR / 'test_labels.npy')

    print(f"    Train data: {train_data.shape}")
    print(f"    Val data:   {val_data.shape}")
    print(f"    Test data:  {test_data.shape}")
    print(f"    Test labels: {test_labels.shape}")

    print("\n[2] Loading trained model")
    checkpoint_path = resolve_checkpoint_path(models_dir)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    config = apply_saved_config(config, checkpoint.get('config', {}))
    model = backbone_spec.model_factory(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Model loaded from epoch {checkpoint['epoch']}")
    print(f"    Best val loss: {checkpoint['val_loss']:.6f}")

    print("\n[3] Generating TRAIN predictions (for calibration)")
    N_train = len(train_data) - config.seq_len
    N_train_full = len(train_data) - config.seq_len - config.pred_len + 1
    T_train = len(train_data)
    print(f"    Train windows (N_train): {N_train} (full-GT subset: {N_train_full})")
    predictions_train = generate_predictions(
        model, train_data, config, backbone_spec, desc="Train Inference",
        batch_size=batch_size,
    )
    print(f"    ✓ Train predictions: {predictions_train.shape}")

    print("\n[3b] Generating VAL predictions (for clean baseline)")
    if len(val_data) >= config.seq_len + config.pred_len:
        N_val = len(val_data) - config.seq_len
        N_val_full = len(val_data) - config.seq_len - config.pred_len + 1
        print(f"    Val windows (N_val): {N_val} (full-GT subset: {N_val_full})")
        predictions_val = generate_predictions(
            model, val_data, config, backbone_spec, desc="Val Inference",
            batch_size=batch_size,
        )
        print(f"    ✓ Val predictions: {predictions_val.shape}")
    else:
        N_val = N_val_full = 0
        predictions_val = np.empty((0, config.pred_len, config.enc_in), dtype=np.float32)
        print(f"    Val too short ({len(val_data)} < {config.seq_len + config.pred_len}); "
              f"empty predictions_val.npy will fall back to train baseline downstream.")

    print("\n[4] Generating TEST predictions")
    N_test = len(test_data) - config.seq_len
    N_test_full = len(test_data) - config.seq_len - config.pred_len + 1
    T_test = len(test_data)
    print(f"    Test windows (N_test): {N_test} (full-GT subset: {N_test_full})")
    predictions_test = generate_predictions(
        model, test_data, config, backbone_spec, desc="Test Inference",
        batch_size=batch_size,
    )
    print(f"    ✓ Test predictions: {predictions_test.shape}")

    print("\n[5] Calculating prediction error (verification)")
    first_pred_train = predictions_train[0]
    first_gt_train = train_data[config.seq_len:config.seq_len + config.pred_len]
    mse_train_first = float(np.mean((first_pred_train - first_gt_train) ** 2))
    mae_train_first = float(np.mean(np.abs(first_pred_train - first_gt_train)))

    first_pred_test = predictions_test[0]
    first_gt_test = test_data[config.seq_len:config.seq_len + config.pred_len]
    mse_test_first = float(np.mean((first_pred_test - first_gt_test) ** 2))
    mae_test_first = float(np.mean(np.abs(first_pred_test - first_gt_test)))

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

    print("\n[6] Saving outputs")
    np.save(os.path.join(outputs_dir, 'predictions_train.npy'), predictions_train)
    np.save(os.path.join(outputs_dir, 'predictions_val.npy'), predictions_val)
    np.save(os.path.join(outputs_dir, 'predictions_test.npy'), predictions_test)
    np.save(os.path.join(outputs_dir, 'test_labels.npy'), test_labels)

    inference_meta = {
        'backbone': backbone_name,
        'N_train': int(N_train),
        'N_test': int(N_test),
        'H': int(config.pred_len),
        'C': int(config.enc_in),
        'T_train': int(T_train),
        'T_test': int(T_test),
        'lookback': int(config.seq_len),
        'train_first_window_mse': mse_train_first,
        'train_first_window_mae': mae_train_first,
        'test_first_window_mse': mse_test_first,
        'test_first_window_mae': mae_test_first,
        'train_mse': float(mse_train),
        'train_mae': float(mae_train),
        'test_mse': float(mse_test),
        'test_mae': float(mae_test),
        'model_epoch': int(checkpoint['epoch']),
        'model_val_loss': float(checkpoint['val_loss']),
    }

    with open(os.path.join(outputs_dir, 'inference_metadata.json'), 'w') as f:
        json.dump(inference_meta, f, indent=4)

    print(f"    ✓ Saved to {outputs_dir}/")
    print(f"      - predictions_train.npy: {predictions_train.shape}")
    print(f"      - predictions_test.npy: {predictions_test.shape}")
    print(f"      - test_labels.npy: {test_labels.shape}")
    print(f"      - inference_metadata.json")

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)

    return predictions_train, predictions_test, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V13 inference with a trained backbone checkpoint")
    parser.add_argument(
        "--backbone", type=str, default=DEFAULT_BACKBONE,
        help=f"Backbone name (default: {DEFAULT_BACKBONE}). Available: {list_backbones()}",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference DataLoader batch size (default 64; lower for high-channel "
             "datasets like OPPORTUNITY=248 channels to avoid OOM).",
    )
    args = parser.parse_args()
    run_inference(args.backbone, batch_size=args.batch_size)
