"""Moirai-1.1-R-small zero-shot wrapper for FDAS.

Adapts Salesforce's MoiraiForecast (~14M params, encoder-only any-variate
Transformer) into V13's multi-horizon forecasting contract. Unlike TimesFM
(univariate per-channel loop), Moirai is **multivariate-native**: it
ingests (B, L, C) directly via any-variate attention where each
(timestep, channel) pair is treated as a separate token, and cross-channel
dependencies are modeled inside the attention.

This matches V13's V_DAS philosophy — TSB-AD-M is a multi-channel anomaly
benchmark, and TTM-r2 has already shown that multivariate-native foundation
models substantially outperform univariate ones. Moirai is a second
multivariate-native foundation backbone with a completely different
architecture from TTM (Transformer vs. MLP-Mixer) and pretraining corpus
(Salesforce LOTSA vs. IBM TSPulse), allowing the framework to disentangle
"multivariate" from any specific architectural choice.

Key facts about Moirai we have to work around:
  - target_dim must be set at construction time, but V13's dataset's
    channel count C varies per dataset. So we re-build MoiraiForecast each
    `__init__` using cfg.enc_in. The actual 14M MoiraiModule weights are
    cached and reloaded from HF; only the outer MoiraiForecast wrapper
    (which holds patch_size + projection heads) is rebuilt per dataset.
  - Moirai is probabilistic: forward returns (B, num_samples, H, C).
    We take the median across samples to get a point forecast (B, H, C).
    num_samples is a speed/quality tradeoff; 20 is a standard choice that
    keeps inference fast and gives a stable median.
  - patch_size=16 gives 192/16=12 input patches + 96/16=6 output patches,
    a clean choice for V13's L=192/H=96 spec. Moirai supports
    {8, 16, 32, 64, 128}; 16 is the paper-recommended default for hourly
    resolution data.

Why a wrapper:
  - Moirai expects keyword args (past_target, past_observed_target,
    past_is_pad), not the TSL positional (x_enc, x_mark_enc, ...)
    signature. We unpack and ignore the unused TSL args.
  - state_dict() is overridden to return {} so checkpoint.pth stays small
    (~100B metadata). Weights live in the HF cache.

Zero-shot ⇒ no training. 02_train.py detects BackboneSpec.is_zero_shot and
skips the train loop, writing only a minimal checkpoint.pth so the rest of
the pipeline (03 → 04 → 05) needs no backbone-specific branching (same
pattern as TimesFM and TTM-r2).
"""
from __future__ import annotations

import torch
import torch.nn as nn


_MOIRAI_REPO_ID = "Salesforce/moirai-1.1-R-small"
_PATCH_SIZE = 16          # 192/16=12 + 96/16=6, clean for V13 spec
_NUM_SAMPLES = 20         # Probabilistic point forecast via median of N samples.
                          # num_samples=1 was tried for speed but produced
                          # heavy-tailed predictions (max 12.7M on Daphnet
                          # vs p99.9=23) — single-sample Monte Carlo is
                          # unstable when the predicted distribution has
                          # wide tails. num_samples=20 + median is the
                          # paper-standard probabilistic point estimator
                          # and gives bounded MSE for FDAS scoring.


class Model(nn.Module):
    """Moirai-1.1-R-small wrapper.

    Forward:
      x_enc:  (B, L=192, C) — encoder input from V13's SlidingWindowDataset
      returns (B, pred_len=96, C)

    Accepts the full TSL positional signature; only x_enc is consumed
    (Moirai doesn't use temporal mark features in our zero-shot setting).
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = int(configs.pred_len)
        self.seq_len = int(configs.seq_len)
        self.target_dim = int(configs.enc_in)

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        # Pretrained 14M weights — cached by HF, reloaded on each instance.
        module = MoiraiModule.from_pretrained(_MOIRAI_REPO_ID)
        self.moirai = MoiraiForecast(
            module=module,
            prediction_length=self.pred_len,
            context_length=self.seq_len,
            patch_size=_PATCH_SIZE,
            num_samples=_NUM_SAMPLES,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # Eval-only path. Zero-shot in V13; freeze and disable dropout.
        self.moirai.eval()
        for p in self.moirai.parameters():
            p.requires_grad_(False)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Moirai expects (B, past_time, tgt) for past_target,
        # (B, past_time, tgt) bool for past_observed_target,
        # (B, past_time) bool for past_is_pad.
        past_target = x_enc.float()
        past_observed = torch.ones(B, L, C, dtype=torch.bool, device=device)
        past_is_pad = torch.zeros(B, L, dtype=torch.bool, device=device)

        with torch.no_grad():
            out = self.moirai(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
                num_samples=_NUM_SAMPLES,
            )
        # out: (B, num_samples, H, C) → median over samples → (B, H, C)
        point_fc = out.median(dim=1).values
        return point_fc[:, : self.pred_len, :].to(x_enc.dtype)

    # ── checkpoint shrinking ──────────────────────────────────────────
    # 14M pretrained weights live in the HF cache. Per-dataset
    # checkpoint.pth should only carry epoch/loss metadata, not 14M weight
    # copies across 200 datasets.
    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state_dict, strict=False):
        # No-op — weights are reloaded from `from_pretrained` in __init__.
        return torch.nn.modules.module._IncompatibleKeys([], [])
