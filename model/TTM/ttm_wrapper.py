"""TTM-r2 (Tiny Time Mixer) zero-shot wrapper for FDAS.

Adapts IBM Granite's TinyTimeMixerForPrediction (~805K params, multivariate
channel-mixing forecaster) into V13's multi-horizon forecasting contract.

Key facts about TTM-r2 we have to work around:
  - The published model card offers checkpoints at native (context, horizon)
    pairs: (512, 96), (1024, 96), (1536, 96), (512, 192), ... — there is no
    (192, 96) variant. V13 fixes seq_len=192 / pred_len=96 for all backbones,
    so we pick the closest fit, 512-96-r2, and zero-pad the input from 192 to
    512 on the LEFT (i.e. past). The `past_observed_mask` (shape (B, L, C))
    tells TTM which positions are real — set to 1 only on the last 192
    positions, 0 on the zero-padded prefix. TTM's NopScaler/scaler honors the
    mask, so the per-series mean/scale is computed from real data only and
    the padded zeros do not contaminate normalization.
  - Output is `out.prediction_outputs` of shape (B, H=96, C) — already in
    V13's contract; no permute needed.

Why a wrapper:
  - TTM expects keyword args (past_values, past_observed_mask), not the
    TSL positional (x_enc, x_mark_enc, x_dec, x_mark_dec) signature.
  - state_dict() is overridden to return {} so checkpoint.pth stays small
    (~100B metadata). Weights live in the HF cache and are reloaded from
    `from_pretrained` on each model instantiation.

Zero-shot ⇒ no training. 02_train.py detects BackboneSpec.is_zero_shot and
skips the train loop, writing only a minimal checkpoint.pth so 03 → 04 → 05
need no backbone-specific branching (same pattern as TimesFM).
"""
from __future__ import annotations

import torch
import torch.nn as nn


_TTM_CONTEXT_LEN = 512    # native checkpoint context (we pad V13's 192 up to this)
_TTM_REPO_ID = "ibm-granite/granite-timeseries-ttm-r2"


class Model(nn.Module):
    """TTM-r2 (TinyTimeMixerForPrediction, 512-96-r2 variant) wrapper.

    Forward:
      x_enc:  (B, L=192, C) — encoder input from V13's SlidingWindowDataset
      returns (B, pred_len=96, C)

    The forward signature accepts the full TSL positional args; only x_enc
    is consumed (TTM ingests raw past_values, no temporal mark features).
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = int(configs.pred_len)
        self.seq_len = int(configs.seq_len)

        from tsfm_public import get_model
        self.ttm = get_model(
            _TTM_REPO_ID,
            context_length=_TTM_CONTEXT_LEN,
            prediction_length=self.pred_len,
        )
        # Eval-only path. TTM is zero-shot in V13; freeze and disable dropout.
        self.ttm.eval()
        for p in self.ttm.parameters():
            p.requires_grad_(False)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Left-zero-pad (B, L, C) → (B, 512, C) when L < TTM context.
        # observed_mask flags real positions; TTM's scaler ignores the pad.
        if L < _TTM_CONTEXT_LEN:
            pad = torch.zeros(B, _TTM_CONTEXT_LEN - L, C,
                              device=device, dtype=x_enc.dtype)
            past_values = torch.cat([pad, x_enc], dim=1)
            mask = torch.zeros(B, _TTM_CONTEXT_LEN, C,
                               device=device, dtype=x_enc.dtype)
            mask[:, -L:, :] = 1.0
        elif L == _TTM_CONTEXT_LEN:
            past_values = x_enc
            mask = torch.ones(B, L, C, device=device, dtype=x_enc.dtype)
        else:
            # Truncate to the latest TTM-context window if V13 ever bumps L.
            past_values = x_enc[:, -_TTM_CONTEXT_LEN:, :]
            mask = torch.ones(B, _TTM_CONTEXT_LEN, C,
                              device=device, dtype=x_enc.dtype)

        # Some TTM-r2.x revisions (notably those picked by get_model for short
        # prediction_length like 12/24/48) require an explicit `freq_token`
        # encoder input. TSB-AD-M has no canonical frequency; we pass zeros
        # (the model's "unknown frequency" slot), which is a no-op for older
        # revisions that don't consume the token and unblocks the strict ones.
        freq_token = torch.zeros(B, dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.ttm(
                past_values=past_values.float(),
                past_observed_mask=mask,
                freq_token=freq_token,
                return_loss=False,
            )
        pred = out.prediction_outputs  # (B, H, C) already
        return pred[:, : self.pred_len, :].to(x_enc.dtype)

    # ── checkpoint shrinking ──────────────────────────────────────────
    # Pretrained 1M weights live in the HF cache. The dataset-specific
    # checkpoint.pth should only carry epoch/loss metadata so we don't
    # store 200×3MB copies of the same pretrained weights.
    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state_dict, strict=False):
        # No-op — weights are reloaded from `from_pretrained` in __init__.
        return torch.nn.modules.module._IncompatibleKeys([], [])
