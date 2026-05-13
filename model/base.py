"""V13 backbone abstraction — `BackboneSpec`.

A backbone is a multi-horizon forecasting model with the Time-Series-Library
forward signature:

    Model(configs).forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        → Tensor of shape (B, pred_len, c_out)

Each entry in `model.BACKBONES` (model/__init__.py) is a `BackboneSpec` that
bundles:
  • the model factory (configs → nn.Module)
  • the model HPs the original paper recommends (d_model, n_heads, ...)
  • the training HPs the original paper recommends (lr, batch_size, epochs, ...)
  • which fields to persist into train_config.json (so inference can reconstruct
    the same architecture from disk)
  • the forward signature variant (some backbones don't take time-mark inputs)

Why training HPs in the spec (not unified across backbones):
  Each forecasting model has well-tuned defaults from its own paper. Forcing
  every backbone onto a single (lr, batch_size, epochs) setup would cripple
  some of them and weaken the comparison. We instead claim "FDAS works on
  top of each backbone trained at its own best HP."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    """Canonical backbone name. Must be unique across BACKBONES.
    Becomes a directory component under models/<dataset>/<name>/ and
    results/<dataset>/<name>/, and is persisted in train_config.json."""

    model_factory: Callable[[object], nn.Module]
    """configs → nn.Module. configs is a Config built by config_factory."""

    default_model_hps: dict
    """Backbone-specific architectural HPs (e.g. {'d_model': 512, 'n_heads': 8}).
    Injected into Config via setattr at build time."""

    default_training_hps: dict
    """Backbone-specific training HPs that the original paper recommends.
    Recognized keys: batch_size, learning_rate, num_epochs, patience,
    optimizer ('adam'|'adamw'), weight_decay, scheduler ('none'|'step',
    {'step_size': N, 'gamma': X}). Anything else passes through to Config."""

    extra_config_fields: list[str] = field(default_factory=list)
    """Names of Config attributes (drawn from default_model_hps + a few common
    overrides) to write into train_config.json so that inference can rebuild
    the exact same model from disk without consulting the registry."""

    forward_signature: str = "tsl"
    """How forward() is called.
       'tsl'    → model(x_enc, x_mark_enc, x_dec, x_mark_dec)  (iTransformer, PatchTST, ...)
       'x_only' → model(x_enc)                                  (DLinear, etc., if added)"""

    is_zero_shot: bool = False
    """If True, the backbone is a pretrained foundation model evaluated
    without any dataset-specific training. 02_train.py skips the train
    loop and writes a minimal checkpoint.pth so the rest of the V13
    pipeline can proceed unchanged. Used by TimesFM and other zero-shot
    foundation models."""

    mark_dim: int = 1
    """Number of channels in the dummy time-mark tensor (x_mark, y_mark).

    Anomaly-detection setup ignores time semantics — masks are zeros. But
    the *shape* must match what the backbone's embedding layer expects:
      - iTransformer / PatchTST / DLinear: mark unused or shape-agnostic → 1 OK
      - TimeMixer / TimesNet: DataEmbedding(_wo_pos) uses TimeFeatureEmbedding
        with freq='h' → mark must be 4-dim (hour, day, weekday, month).

    Per-backbone setting so legacy iTransformer artifacts trained with
    mark_dim=1 keep producing identical outputs after this refactor."""

    def call_forward(self, model: nn.Module, x_enc, x_mark_enc=None,
                     x_dec=None, x_mark_dec=None):
        """Single dispatch point so train/inference loops are signature-agnostic."""
        if self.forward_signature == "tsl":
            return model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.forward_signature == "x_only":
            return model(x_enc)
        raise ValueError(f"unknown forward_signature: {self.forward_signature}")
