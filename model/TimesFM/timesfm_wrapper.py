"""TimesFM-1.0-200M zero-shot wrapper for FDAS.

Adapts Google's TimesFM (decoder-only foundation forecaster) into V13's
multi-horizon forecasting contract. TimesFM is *univariate-only* — for a
C-channel input we forecast each channel independently and stack back.

Why a wrapper:
  - TimesFm.forecast() takes a list of numpy float32 series and returns
    (B', H) numpy. We must batch (B, L, C) → (B*C, L), call forecast(),
    reshape to (B, H, C).
  - state_dict() is overridden to return {} so checkpoint.pth stays small
    (~100B metadata only). The actual 200M weights are loaded from the
    HuggingFace cache on every `from_pretrained()` call, but the cache is
    keyed by repo_id so first-call is ~2s and subsequent calls are
    near-instant.

Zero-shot ⇒ no training. 02_train.py detects BackboneSpec.is_zero_shot and
skips the train loop entirely, writing a minimal checkpoint.pth so that the
rest of the V13 pipeline (03_inference, 04_score_compute, 05_metrics) does
not need any backbone-specific branching.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """TimesFM-1.0-200m wrapper.

    Forward:
      x_enc:  (B, L, C) — encoder input
      returns (B, pred_len, C)

    The forward signature matches "tsl" (positional args ignored except x_enc)
    so 02_train's TimeSeriesDataset and 03_inference's SlidingWindowDataset
    work without modification.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = int(configs.pred_len)
        self.seq_len = int(configs.seq_len)

        import timesfm
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if torch.cuda.is_available() else "cpu",
                per_core_batch_size=int(getattr(configs, "tfm_batch", 32)),
                horizon_len=self.pred_len,
                context_len=self.seq_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # (B, L, C) → (B*C, L) numpy
        B, L, C = x_enc.shape
        series_np = (
            x_enc.detach().permute(0, 2, 1).contiguous()
                 .reshape(B * C, L).float().cpu().numpy()
        )
        # TimesFM forecast: high-frequency = 0 for all (we have no temporal semantics)
        point_fc, _ = self.tfm.forecast(
            inputs=[series_np[i] for i in range(B * C)],
            freq=[0] * (B * C),
        )
        # (B*C, H) → (B, C, H) → (B, H, C)
        pred = torch.from_numpy(point_fc).reshape(B, C, -1).permute(0, 2, 1)
        return pred[:, : self.pred_len, :].to(x_enc.device).float()

    # ── checkpoint shrinking ──────────────────────────────────────────
    # 200M weights live in the HF cache. The dataset-specific checkpoint.pth
    # should only carry epoch/loss metadata so we don't store 200×800MB
    # copies of the same pretrained weights.
    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state_dict, strict=False):
        # No-op — weights are reloaded from `from_pretrained` in __init__.
        return torch.nn.modules.module._IncompatibleKeys([], [])
