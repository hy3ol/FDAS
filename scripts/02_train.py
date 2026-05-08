"""
iTransformer Training Script for V11 (forward-looking).

Train iTransformer on normal data only for multi-step forecasting.
Identical mechanics to V10 — V11's contribution is in score computation.
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

from model.iTransformer import Model
from artifact_paths import DATA_DIR, get_models_dir, load_data_metadata


class Config:
    def __init__(self, metadata):
        # Model architecture
        self.seq_len = metadata['lookback']  # L
        self.pred_len = metadata['pred_len']  # H (6)
        self.enc_in = metadata['num_channels']  # C
        self.dec_in = metadata['num_channels']
        self.c_out = metadata['num_channels']

        # iTransformer specific
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False
        # V13: keep iTransformer's per-window normalization (Non-stationary
        # Transformer trick). Earlier we tried use_norm=False to expose the
        # anomaly-shift signal that per-window stats can absorb, but the
        # side effects (immediate overfitting → patience-3 cliff after
        # 4 epochs, and downstream metric regressions across families)
        # outweighed the theoretical gain. The right place to address the
        # channel-scale issue is per-channel D_w_c normalization in the
        # scoring stage, not by removing the model's own normalization.
        self.use_norm = True

        # Embedding
        self.embed = 'timeF'
        self.freq = 'h'
        self.factor = 1
        self.class_strategy = 'projection'

        # Training
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.patience = 3

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if self.device.type == 'cuda' else 0
        self.pin_memory = self.device.type == 'cuda'


class TimeSeriesDataset(Dataset):
    """Dataset for multi-step forecasting"""

    def __init__(self, data, lookback, pred_len):
        """
        Args:
            data: (T, C) numpy array
            lookback: input sequence length (L)
            pred_len: prediction horizon (H)
        """
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len

        # Calculate number of valid samples
        self.length = len(data) - lookback - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            x: (lookback, C) input sequence
            y: (pred_len, C) target sequence
        """
        x = self.data[idx:idx+self.lookback]
        y = self.data[idx+self.lookback:idx+self.lookback+self.pred_len]

        # Create dummy timestamps (not used but required by model)
        x_mark = np.zeros((self.lookback, 1))
        y_mark = np.zeros((self.pred_len, 1))

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(x_mark),
            torch.FloatTensor(y),
            torch.FloatTensor(y_mark)
        )


def export_train_config(config):
    return {
        'seq_len': int(config.seq_len),
        'pred_len': int(config.pred_len),
        'enc_in': int(config.enc_in),
        'dec_in': int(config.dec_in),
        'c_out': int(config.c_out),
        'd_model': int(config.d_model),
        'n_heads': int(config.n_heads),
        'e_layers': int(config.e_layers),
        'd_ff': int(config.d_ff),
        'dropout': float(config.dropout),
        'activation': str(config.activation),
        'output_attention': bool(config.output_attention),
        'use_norm': bool(config.use_norm),
        'embed': str(config.embed),
        'freq': str(config.freq),
        'factor': int(config.factor),
        'class_strategy': str(config.class_strategy),
        'batch_size': int(config.batch_size),
        'learning_rate': float(config.learning_rate),
        'num_epochs': int(config.num_epochs),
        'patience': int(config.patience),
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for x_enc, x_mark, y_true, y_mark in pbar:
        x_enc = x_enc.to(device)
        x_mark = x_mark.to(device)
        y_true = y_true.to(device)
        y_mark = y_mark.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x_enc, None, None, None)

        # Calculate loss
        loss = criterion(y_pred, y_true)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for x_enc, x_mark, y_true, y_mark in pbar:
            x_enc = x_enc.to(device)
            x_mark = x_mark.to(device)
            y_true = y_true.to(device)
            y_mark = y_mark.to(device)

            # Forward pass
            y_pred = model(x_enc, None, None, None)

            # Calculate loss
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def train_model():
    print("="*60)
    print("iTransformer Training for V11")
    print("="*60)

    # Load metadata
    print("\n[1] Loading metadata and data")
    metadata = load_data_metadata()

    config = Config(metadata)
    models_dir = get_models_dir()
    print(f"    Lookback (L): {config.seq_len}")
    print(f"    Pred_len (H): {config.pred_len}")
    print(f"    Channels (C): {config.enc_in}")
    print(f"    Device: {config.device}")
    print(f"    Model dir: {models_dir}")

    train_config_path = os.path.join(models_dir, 'train_config.json')
    with open(train_config_path, 'w') as f:
        json.dump(export_train_config(config), f, indent=4)
    print(f"    Train config: {train_config_path}")

    # Load data
    train_data = np.load(DATA_DIR / 'train_data.npy')
    val_data = np.load(DATA_DIR / 'val_data.npy')

    print(f"    Train data: {train_data.shape}")
    print(f"    Val data: {val_data.shape}")

    # Create datasets
    print("\n[2] Creating datasets")
    train_dataset = TimeSeriesDataset(train_data, config.seq_len, config.pred_len)
    val_dataset = TimeSeriesDataset(val_data, config.seq_len, config.pred_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    print(f"    DataLoader workers: {config.num_workers}")

    # Create model
    print("\n[3] Initializing iTransformer model")
    model = Model(config).to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    print("\n[4] Training")
    print("-"*60)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    early_stopped = False
    train_losses = []
    val_losses = []

    for epoch in range(config.num_epochs):
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, config.device)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        print(f"\n  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time:       {epoch_time:.2f}s")

        # Save best snapshot to BOTH best_model.pth and checkpoint.pth
        # (inference picks checkpoint.pth first, so the best epoch is what
        # downstream evaluation actually sees).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            checkpoint_payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': export_train_config(config),
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
                    f"  ! Early stopping triggered after epoch {epoch+1} "
                    f"(best val_loss = {best_val_loss:.6f} at epoch {best_epoch+1})."
                )
                break

    print("\n" + "-"*60)
    if early_stopped:
        print("Training Complete (early-stopped).")
    else:
        print("Training Complete (full epochs ran; no early-stop trigger).")
    print(f"  Best val_loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print(f"  Final epoch:   {len(train_losses)} (val_loss = "
          f"{val_losses[-1] if val_losses else 'n/a'})")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_epoch),
        'epochs_trained': len(train_losses),
        'early_stopped': bool(early_stopped),
        'patience': int(config.patience),
    }

    with open(os.path.join(models_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print("\n" + "="*60)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the V11 iTransformer model")
    parser.parse_args()
    model, history = train_model()
