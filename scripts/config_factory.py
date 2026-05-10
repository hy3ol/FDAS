"""V13 Config builder — shared by 02_train.py and 03_inference.py.

Composes a single `Config` object from three sources, in priority order:
  1. Dataset metadata (seq_len, pred_len, channel count) — required.
  2. Backbone defaults (model + training HPs from BackboneSpec).
  3. Saved overrides from train_config.json / checkpoint (inference path).

Result is a plain attribute container compatible with the iTransformer
constructor (and any TSL-style backbone) — same shape as the pre-refactor
Config classes that used to live separately in 02 and 03.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class Config:
    """Plain attribute container. Mirrors the pre-refactor V13 Config shape."""
    def __init__(self):
        # Common (backbone-agnostic) — populated by build_config
        self.seq_len: int = 0
        self.pred_len: int = 0
        self.enc_in: int = 0
        self.dec_in: int = 0
        self.c_out: int = 0
        # Training defaults (backbone may override)
        self.batch_size: int = 32
        self.learning_rate: float = 1e-4
        self.num_epochs: int = 10
        self.patience: int = 3
        # Device
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_workers: int = 4 if self.device.type == "cuda" else 0
        self.pin_memory: bool = self.device.type == "cuda"
        # Identity
        self.backbone: str = ""

    def to_dict(self, include_fields: list[str]) -> dict:
        """Serialize a chosen subset of attrs into a JSON-safe dict.

        Used by 02_train.py to persist train_config.json and embed config in
        the checkpoint. Callers pass in core fields + BackboneSpec.extra_config_fields.
        """
        out: dict[str, Any] = {}
        for k in include_fields:
            v = getattr(self, k, None)
            if isinstance(v, torch.device):
                out[k] = str(v)
            elif isinstance(v, (int, float, bool, str, list, dict)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out


CORE_PERSIST_FIELDS = [
    "backbone",
    "seq_len", "pred_len", "enc_in", "dec_in", "c_out",
    "batch_size", "learning_rate", "num_epochs", "patience",
]


def build_config(metadata: dict, backbone_spec, *, overrides: dict | None = None) -> Config:
    """Compose a Config for a (dataset, backbone) pair.

    Args:
      metadata: from artifact_paths.load_data_metadata() — provides
                lookback / pred_len / num_channels.
      backbone_spec: BackboneSpec — provides default_model_hps, default_training_hps.
      overrides: optional dict of attr-name → value (used by inference to
                 re-apply train_config.json fields, and for one-off CLI overrides).

    Note on training HP application: keys in default_training_hps are mapped
    to standard Config attribute names (batch_size, learning_rate, num_epochs,
    patience). Other recognized keys (optimizer, scheduler, weight_decay) pass
    through with their original names so the training loop can read them via
    getattr(config, key, default).
    """
    cfg = Config()

    # 1. dataset
    cfg.seq_len = int(metadata["lookback"])
    cfg.pred_len = int(metadata["pred_len"])
    cfg.enc_in = cfg.dec_in = cfg.c_out = int(metadata["num_channels"])

    # 2. backbone identity + HPs
    cfg.backbone = backbone_spec.name
    for k, v in backbone_spec.default_model_hps.items():
        setattr(cfg, k, v)
    for k, v in backbone_spec.default_training_hps.items():
        setattr(cfg, k, v)

    # 3. overrides (last wins)
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)

    return cfg


def export_train_config(cfg: Config, backbone_spec) -> dict:
    """Snapshot exactly the fields needed to rebuild Config at inference time.

    = core fields + backbone-specific extra_config_fields. Anything not in
    this list is treated as ephemeral (won't be replayed at inference).
    """
    fields = CORE_PERSIST_FIELDS + list(backbone_spec.extra_config_fields)
    return cfg.to_dict(fields)


def apply_saved_config(cfg: Config, saved: dict | None) -> Config:
    """Replay every field present in `saved` into `cfg`. No filtering.

    Used by 03_inference.py to apply train_config.json then checkpoint['config'].
    """
    if not isinstance(saved, dict):
        return cfg
    for k, v in saved.items():
        # device fields are stored as strings on disk; rehydrate
        if k == "device" and isinstance(v, str):
            v = torch.device(v)
        setattr(cfg, k, v)
    return cfg


def load_train_config(models_dir: Path) -> dict | None:
    """Read train_config.json if present, else None."""
    p = Path(models_dir) / "train_config.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)
