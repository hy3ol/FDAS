"""V13 backbone registry.

Add a new forecasting backbone in 3 steps:
  1. Drop a `<name>.py` file under V13/model/ with a `Model(configs)` class
     following Time-Series-Library's forward signature.
  2. Register a BackboneSpec entry below.
  3. Train + infer with `--backbone <name>`. No other code change.

The registry is the single source of truth for what's installed. 02_train.py,
03_inference.py, run_all.py all read from BACKBONES.

Default backbone is "iTransformer" — V13's original entry, kept identical to
the legacy hardcoded HPs so existing 200-dataset artifacts keep working.
"""
from __future__ import annotations

from .base import BackboneSpec
from . import iTransformer as _itransformer

DEFAULT_BACKBONE = "iTransformer"

BACKBONES: dict[str, BackboneSpec] = {
    # ── iTransformer (Liu et al., ICLR 2024). ──────────────────────────
    # Model HPs and training HPs match V13's pre-refactor 02_train.py
    # exactly, so re-running --backbone iTransformer is a no-op for any
    # already-trained dataset.
    "iTransformer": BackboneSpec(
        name="iTransformer",
        model_factory=lambda cfg: _itransformer.Model(cfg),
        default_model_hps=dict(
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            factor=1,
            embed="timeF",
            freq="h",
            class_strategy="projection",
            use_norm=True,                # V13 fixed: keep iTransformer per-window norm
            output_attention=False,
        ),
        default_training_hps=dict(
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=10,
            patience=3,                   # V13 fixed: early stop after 3 epochs no-improve
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=[
            "d_model", "n_heads", "e_layers", "d_ff", "dropout",
            "activation", "factor", "embed", "freq", "class_strategy",
            "use_norm", "output_attention",
        ],
        forward_signature="tsl",
    ),
    # Future backbones (DLinear, PatchTST, TimesNet, ...) get appended here.
}


def get_backbone(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise KeyError(
            f"Unknown backbone '{name}'. Registered: {sorted(BACKBONES.keys())}"
        )
    return BACKBONES[name]


def list_backbones() -> list[str]:
    return sorted(BACKBONES.keys())
