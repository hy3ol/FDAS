"""
V13 training script — backbone-pluggable.

Trains a multi-horizon forecasting backbone on normal data only.
Backbone is selected via `--backbone <name>` (default: iTransformer).
HPs are sourced from the registered BackboneSpec (model/__init__.py),
so each backbone can keep its own paper-recommended training setup.

The training loop itself is backbone-agnostic — it only relies on the
TSL-style forward signature dispatched through `BackboneSpec.call_forward`.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import get_backbone, list_backbones, DEFAULT_BACKBONE
from artifact_paths import DATA_DIR, get_models_dir, load_data_metadata
from config_factory import build_config, export_train_config


class TimeSeriesDataset(Dataset):
    """Dataset for multi-step forecasting (backbone-agnostic).

    `mark_dim` defaults to 1 (legacy iTransformer / PatchTST / DLinear shape,
    bit-identical to pre-multimodel vintage). TimeMixer / TimesNet need
    mark_dim=4 (timeF + freq='h' contract); train_model() reads that from
    `BackboneSpec.mark_dim` and passes it in.
    """

    def __init__(self, data, lookback, pred_len, mark_dim=1):
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len
        self.mark_dim = mark_dim
        self.length = len(data) - lookback - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.data[index:index + self.lookback]
        y = self.data[index + self.lookback:index + self.lookback + self.pred_len]
        # Anomaly setup ignores time semantics; mark is zeros. Shape just
        # has to match the backbone's embedding layer (see BackboneSpec.mark_dim).
        x_mark = np.zeros((self.lookback, self.mark_dim))
        y_mark = np.zeros((self.pred_len, self.mark_dim))
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y),
            torch.FloatTensor(y_mark),
        )


def _build_optimizer(model, config):
    """Optimizer chosen by config.optimizer (default 'adam').

    Recognizes 'adam' and 'adamw'. Weight decay read from
    config.weight_decay (default 0). Per-backbone defaults come from
    BackboneSpec.default_training_hps.
    """
    name = getattr(config, "optimizer", "adam").lower()
    wd = float(getattr(config, "weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=wd)


def _build_scheduler(optimizer, config):
    """LR scheduler from config.scheduler (default 'none')."""
    name = getattr(config, "scheduler", "none").lower()
    if name == "step":
        step_size = int(getattr(config, "scheduler_step_size", 5))
        gamma = float(getattr(config, "scheduler_gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


def train_epoch(model, dataloader, criterion, optimizer, device, backbone_spec):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc='Training')
    for x_enc, x_mark, y_true, y_mark in pbar:
        x_enc = x_enc.to(device)
        x_mark = x_mark.to(device)
        y_true = y_true.to(device)
        y_mark = y_mark.to(device)

        optimizer.zero_grad()
        y_pred = backbone_spec.call_forward(model, x_enc, x_mark, None, y_mark)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, backbone_spec):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for x_enc, x_mark, y_true, y_mark in pbar:
            x_enc = x_enc.to(device)
            x_mark = x_mark.to(device)
            y_true = y_true.to(device)
            y_mark = y_mark.to(device)

            y_pred = backbone_spec.call_forward(model, x_enc, x_mark, None, y_mark)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def train_model(backbone_name: str):
    backbone_spec = get_backbone(backbone_name)

    print("=" * 60)
    print(f"V13 Training — backbone: {backbone_name}")
    print("=" * 60)

    # Load metadata
    print("\n[1] Loading metadata and data")
    metadata = load_data_metadata()

    config = build_config(metadata, backbone_spec)
    models_dir = get_models_dir(backbone=backbone_name)
    print(f"    Lookback (L): {config.seq_len}")
    print(f"    Pred_len (H): {config.pred_len}")
    print(f"    Channels (C): {config.enc_in}")
    print(f"    Device: {config.device}")
    print(f"    Model dir: {models_dir}")
    print(f"    Backbone HPs: {backbone_spec.default_model_hps}")
    print(f"    Training HPs: {backbone_spec.default_training_hps}")

    train_config_path = os.path.join(models_dir, 'train_config.json')
    with open(train_config_path, 'w') as f:
        json.dump(export_train_config(config, backbone_spec), f, indent=4)
    print(f"    Train config: {train_config_path}")

    # Load data
    train_data = np.load(DATA_DIR / 'train_data.npy')
    val_data = np.load(DATA_DIR / 'val_data.npy')

    print(f"    Train data: {train_data.shape}")
    print(f"    Val data: {val_data.shape}")

    # Create datasets
    print("\n[2] Creating datasets")
    train_dataset = TimeSeriesDataset(train_data, config.seq_len, config.pred_len,
                                       mark_dim=backbone_spec.mark_dim)
    val_dataset = TimeSeriesDataset(val_data, config.seq_len, config.pred_len,
                                     mark_dim=backbone_spec.mark_dim)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    print(f"    DataLoader workers: {config.num_workers}")

    # Create model via registry
    print(f"\n[3] Initializing {backbone_name} model")
    model = backbone_spec.model_factory(config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Loss / optimizer / scheduler — all driven by config (backbone HPs)
    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    print("\n[4] Training")
    print("-" * 60)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    early_stopped = False
    train_losses, val_losses = [], []

    for epoch in range(config.num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 config.device, backbone_spec)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, config.device, backbone_spec)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start_time
        print(f"\n  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time:       {epoch_time:.2f}s")

        # Best-model snapshot — saved to BOTH best_model.pth and checkpoint.pth
        # so 03_inference's checkpoint.pth-first lookup picks the best epoch.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            checkpoint_payload = {
                'epoch': epoch,
                'backbone': backbone_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': export_train_config(config, backbone_spec),
            }
            torch.save(checkpoint_payload, os.path.join(models_dir, 'best_model.pth'))
            torch.save(checkpoint_payload, os.path.join(models_dir, 'checkpoint.pth'))
            print(f"  ✓ Best model snapshot updated (val_loss = {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  · No improvement ({epochs_no_improve}/{config.patience})")
            if epochs_no_improve >= config.patience:
                early_stopped = True
                print(
                    f"  ! Early stopping triggered after epoch {epoch + 1} "
                    f"(best val_loss = {best_val_loss:.6f} at epoch {best_epoch + 1})."
                )
                break

    print("\n" + "-" * 60)
    print("Training Complete" + (" (early-stopped)." if early_stopped else "."))
    print(f"  Best val_loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
    print(f"  Final epoch:   {len(train_losses)} (val_loss = "
          f"{val_losses[-1] if val_losses else 'n/a'})")

    history = {
        'backbone': backbone_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'best_epoch': best_epoch,
        'epochs_trained': len(train_losses),
        'early_stopped': early_stopped,
        'patience': config.patience,
    }

    with open(os.path.join(models_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 60)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a V13 forecasting backbone")
    parser.add_argument(
        "--backbone", type=str, default=DEFAULT_BACKBONE,
        help=f"Backbone name (default: {DEFAULT_BACKBONE}). "
             f"Available: {list_backbones()}",
    )
    args = parser.parse_args()
    train_model(args.backbone)
