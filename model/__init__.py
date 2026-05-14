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
from . import DLinear as _dlinear
from . import PatchTST as _patchtst
from . import TimeMixer as _timemixer
from . import TimesNet as _timesnet
from . import TimeXer as _timexer
from . import TimesFM as _timesfm
from . import TTM as _ttm
from . import Moirai as _moirai

DEFAULT_BACKBONE = "iTransformer"

BACKBONES: dict[str, BackboneSpec] = {
    # ── iTransformer (Liu et al., ICLR 2024). ──────────────────────────
    # Vendored verbatim from thuml/Time-Series-Library `models/iTransformer.py`
    # (with TSL layer imports rewritten to `from ._layers import ...` for
    # folder self-containment).
    #
    # `model/iTransformer/_layers.py` bundles DataEmbedding_inverted,
    # Encoder/EncoderLayer, FullAttention/AttentionLayer (all TSL verbatim).
    #
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
    # ── DLinear (Zeng et al., AAAI 2023). ──────────────────────────────
    # Vendored verbatim from cure-lab/LTSF-Linear (official authors' repo);
    # see model/DLinear.py — no modifications. Decomposition kernel_size=25
    # is hardcoded inside the upstream Model class (line 48 of DLinear.py)
    # so it's not exposed as a Config field.
    #
    # forward_signature="x_only" because DLinear's forward takes only (x);
    # the time-mark inputs of the TSL signature are unused. V13's
    # `BackboneSpec.call_forward` dispatches accordingly.
    "DLinear": BackboneSpec(
        name="DLinear",
        model_factory=lambda cfg: _dlinear.Model(cfg),
        default_model_hps=dict(
            individual=False,             # LTSF-Linear default (channel-shared linear)
        ),
        default_training_hps=dict(
            batch_size=32,
            learning_rate=5e-3,           # LTSF-Linear paper default (--learning_rate 0.005)
            num_epochs=10,
            patience=3,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=["individual"],
        forward_signature="x_only",
    ),
    # ── PatchTST (Nie et al., ICLR 2023). ──────────────────────────────
    # Vendored verbatim from thuml/Time-Series-Library `models/PatchTST.py`
    # (with TSL layer imports rewritten to `from ._layers import ...` for
    # folder self-containment).
    #
    # `model/PatchTST/_layers.py` bundles PatchEmbedding, PositionalEmbedding,
    # Encoder/EncoderLayer, FullAttention/AttentionLayer (all TSL verbatim).
    #
    # patch_len=16, stride=8 are PatchTST.Model.__init__ defaults (paper
    # values for L=192); we set them via lambda so train_config.json
    # captures them. task_name="long_term_forecast" routes through the
    # forecast() branch of the multi-task Model.
    "PatchTST": BackboneSpec(
        name="PatchTST",
        model_factory=lambda cfg: _patchtst.Model(
            cfg, patch_len=cfg.patch_len, stride=cfg.stride
        ),
        default_model_hps=dict(
            # Architecture (TSL Optimal_Multi_algo / paper Appendix)
            d_model=128,
            n_heads=16,
            e_layers=3,
            d_ff=256,
            dropout=0.2,
            factor=1,
            activation="gelu",
            # PatchTST-specific
            patch_len=16,
            stride=8,
            # TSL signature stubs (unused by PatchTST but required by build_config)
            embed="timeF",
            freq="h",
            # Multi-task gate (we always use forecast())
            task_name="long_term_forecast",
        ),
        default_training_hps=dict(
            batch_size=128,               # TSL default
            learning_rate=1e-4,           # PatchTST paper Appendix A.2
            num_epochs=10,
            patience=3,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=[
            "d_model", "n_heads", "e_layers", "d_ff", "dropout",
            "factor", "activation", "patch_len", "stride",
            "embed", "freq", "task_name",
        ],
        forward_signature="tsl",
    ),
    # ── TimeMixer (Wang et al., ICLR 2024). ────────────────────────────
    # Vendored verbatim from thuml/Time-Series-Library `models/TimeMixer.py`
    # (with the three `from layers.X import Y` imports rewritten to
    # `from ._layers import Y` for folder self-containment).
    #
    # `model/TimeMixer/_layers.py` bundles series_decomp + DataEmbedding_wo_pos
    # + Normalize (TSL verbatim). label_len=0 because we don't use a decoder
    # window; task_name="long_term_forecast" routes through forecast().
    # use_norm=1 enables per-window normalization (TSL paper default).
    "TimeMixer": BackboneSpec(
        name="TimeMixer",
        model_factory=lambda cfg: _timemixer.Model(cfg),
        default_model_hps=dict(
            # Architecture (TSL Optimal_Multi_algo + ICLR'24 paper Appendix)
            d_model=32,
            d_ff=32,
            e_layers=2,
            dropout=0.1,
            # TimeMixer-specific
            moving_avg=25,
            channel_independence=1,         # 1 = channel-independent (paper default)
            down_sampling_layers=3,
            down_sampling_window=2,
            down_sampling_method="avg",
            decomp_method="moving_avg",     # "moving_avg" or "dft_decomp"
            use_norm=1,                     # 1 = enable per-window Normalize
            # TSL signature stubs
            embed="timeF",
            freq="h",
            label_len=0,                    # forecasting only (no decoder window)
            task_name="long_term_forecast",
        ),
        default_training_hps=dict(
            batch_size=128,
            learning_rate=1e-3,             # ICLR'24 paper Appendix A.2
            num_epochs=10,
            patience=3,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=[
            "d_model", "d_ff", "e_layers", "dropout",
            "moving_avg", "channel_independence",
            "down_sampling_layers", "down_sampling_window", "down_sampling_method",
            "decomp_method", "use_norm",
            "embed", "freq", "label_len", "task_name",
        ],
        forward_signature="tsl",
        mark_dim=4,                         # timeF + freq='h' → 4-dim mark
    ),
    # ── TimesNet (Wu et al., ICLR 2023). ───────────────────────────────
    # Vendored verbatim from thuml/Time-Series-Library `models/TimesNet.py`
    # (with TSL layer imports rewritten to `from ._layers import ...` for
    # folder self-containment).
    #
    # `model/TimesNet/_layers.py` bundles Inception_Block_V1 + V2 and the
    # DataEmbedding family (TokenEmbedding, PositionalEmbedding,
    # TemporalEmbedding, TimeFeatureEmbedding, FixedEmbedding) — all TSL
    # verbatim. top_k=5 picks the dominant 5 periods (paper default). num_kernels=6 is
    # the Inception_Block default. d_model=32, d_ff=32 keeps this heavy
    # CNN-on-period-folding model trainable across 200 datasets at reasonable
    # cost (paper uses up to d_model=64 per-dataset).
    "TimesNet": BackboneSpec(
        name="TimesNet",
        model_factory=lambda cfg: _timesnet.Model(cfg),
        default_model_hps=dict(
            # Architecture (TSL Optimal_Multi_algo + ICLR'23 paper Appendix)
            d_model=32,
            d_ff=32,
            e_layers=2,
            dropout=0.1,
            # TimesNet-specific
            top_k=5,                        # number of dominant FFT periods
            num_kernels=6,                  # Inception_Block kernel count
            # TSL signature stubs
            embed="timeF",
            freq="h",
            label_len=0,
            task_name="long_term_forecast",
        ),
        default_training_hps=dict(
            batch_size=32,                  # TSL default
            learning_rate=1e-4,             # ICLR'23 paper
            num_epochs=10,
            patience=3,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=[
            "d_model", "d_ff", "e_layers", "dropout",
            "top_k", "num_kernels",
            "embed", "freq", "label_len", "task_name",
        ],
        forward_signature="tsl",
        mark_dim=4,                         # timeF + freq='h' → 4-dim mark
    ),
    # ── TimeXer (Wang et al., NeurIPS 2024). ───────────────────────────
    # Vendored verbatim from thuml/Time-Series-Library `models/TimeXer.py`
    # (with TSL layer imports rewritten to `from ._layers import ...` for
    # folder self-containment).
    #
    # `model/TimeXer/_layers.py` bundles DataEmbedding_inverted,
    # PositionalEmbedding, FullAttention/AttentionLayer (TSL verbatim).
    #
    # features="M" routes through forecast_multi() — all channels are
    # both endogenous (en_embedding, patched) and exogenous (ex_embedding,
    # inverted) in turn. patch_len=16 with L=192 → 12 patches + 1 global
    # token. use_norm=1 enables Non-stationary Transformer normalization
    # (paper default).
    "TimeXer": BackboneSpec(
        name="TimeXer",
        model_factory=lambda cfg: _timexer.Model(cfg),
        default_model_hps=dict(
            # Architecture (NeurIPS'24 paper Appendix B)
            d_model=256,
            n_heads=8,
            e_layers=2,
            d_ff=512,
            dropout=0.1,
            factor=1,
            activation="gelu",
            # TimeXer-specific
            patch_len=16,                   # L=192 / 16 = 12 patches
            use_norm=1,                     # 1 = Non-stationary normalization
            features="M",                   # multivariate (use forecast_multi)
            # TSL signature stubs
            embed="timeF",
            freq="h",
            task_name="long_term_forecast",
        ),
        default_training_hps=dict(
            batch_size=32,                  # paper default (4-32 dataset dependent)
            learning_rate=1e-4,             # NeurIPS'24 paper Appendix B
            num_epochs=10,
            patience=3,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=[
            "d_model", "n_heads", "e_layers", "d_ff", "dropout",
            "factor", "activation",
            "patch_len", "use_norm", "features",
            "embed", "freq", "task_name",
        ],
        forward_signature="tsl",
    ),
    # ── TimesFM-1.0-200M (Das et al., ICLR 2024). ──────────────────────
    # Decoder-only foundation forecaster from Google Research. Zero-shot:
    # we load the pretrained 200M-parameter checkpoint from HuggingFace
    # (`google/timesfm-1.0-200m-pytorch`) and forecast each channel
    # independently, with no dataset-specific training.
    #
    # The wrapper lives in `model/TimesFM/timesfm_wrapper.py`. Because
    # weights are reloaded from the HF cache on each `from_pretrained`,
    # the per-dataset checkpoint.pth stays ~100B (state_dict overridden
    # to {}). 02_train.py detects `is_zero_shot=True` and skips the train
    # loop; the rest of the V13 pipeline (03 → 04 → 05) is unchanged.
    "TimesFM": BackboneSpec(
        name="TimesFM",
        model_factory=lambda cfg: _timesfm.Model(cfg),
        default_model_hps=dict(
            # TimesFM-1.0 fixed architecture (paper Table 1)
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        ),
        default_training_hps=dict(
            batch_size=32,                  # outer DataLoader batch
            tfm_batch=512,                  # internal per_core_batch_size
            # tfm_batch=512 trades GPU memory for throughput. TimesFM's
            # inference memory is dominated by the 800MB model weights;
            # patches=6 means attention is tiny (6² = 36) so activations
            # grow linearly with batch but stay sub-GB. Measured peak ~3GB
            # at 256 → ~3.5GB at 512 on a 32GB GPU, well within headroom.
            # Per-series forward is independent — batch size does not
            # affect numerical results beyond fp32 reduction order (<1e-7).
            learning_rate=0.0,              # unused (zero-shot)
            num_epochs=0,                   # unused (zero-shot)
            patience=0,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=[
            "input_patch_len", "output_patch_len",
            "num_layers", "model_dims", "tfm_batch",
        ],
        forward_signature="tsl",            # wrapper.forward accepts TSL args
        is_zero_shot=True,
    ),
    # ── TTM-r2 (Ekambaram et al., NeurIPS 2024). ───────────────────────
    # IBM Granite Tiny Time Mixer r2 — ~805K-parameter multivariate
    # channel-mixing foundation forecaster from `ibm-granite/granite-
    # timeseries-ttm-r2`. Unlike TimesFM (channel-wise univariate), TTM-r2
    # is multivariate-native, which is more aligned with TSB-AD-M's
    # multi-channel anomaly setting.
    #
    # TTM-r2's public checkpoints exist at native (context, horizon) pairs:
    # (512, 96), (1024, 96), (1536, 96), (512, 192), ... — no (192, 96)
    # variant. V13 is fixed at L=192/H=96, so the wrapper loads 512-96-r2
    # and left-zero-pads x_enc up to 512, with `past_observed_mask`=1 only
    # on the last 192 positions so TTM's scaler ignores the pad. See
    # `model/TTM/ttm_wrapper.py` for details.
    #
    # Zero-shot ⇒ no training. is_zero_shot=True routes 02_train.py to
    # write a minimal checkpoint.pth and skip the train loop, mirroring
    # the TimesFM pattern. Pretrained weights live in the HF cache.
    "TTM": BackboneSpec(
        name="TTM",
        model_factory=lambda cfg: _ttm.Model(cfg),
        default_model_hps=dict(
            # TTM-r2 512-96-r2 fixed architecture (paper / HF card).
            # No tunables exposed in zero-shot mode; the wrapper hard-codes
            # context=512 (padded from V13's 192) and prediction=96.
            ttm_context_len=512,
        ),
        default_training_hps=dict(
            batch_size=128,                 # outer DataLoader batch
            learning_rate=0.0,              # unused (zero-shot)
            num_epochs=0,                   # unused (zero-shot)
            patience=0,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=["ttm_context_len"],
        forward_signature="tsl",
        is_zero_shot=True,
    ),
    # ── Moirai-1.1-R-small (Woo et al., ICLR 2024). ────────────────────
    # Salesforce's MoiraiForecast — ~14M-parameter encoder-only Transformer
    # with **any-variate attention**: each (timestep, channel) pair is a
    # separate token, and cross-channel dependency is modeled inside the
    # attention. Truly multivariate-native (different architecture from
    # TTM-r2's MLP-Mixer channel-mixing).
    #
    # Loaded from HuggingFace `Salesforce/moirai-1.1-R-small`. Pretrained
    # on Salesforce LOTSA (27B observations), distinct corpus from IBM's
    # TSPulse used for TTM-r2 — so this row also varies the pretraining
    # data axis, not just architecture.
    #
    # patch_size=16 gives 192/16=12 input patches + 96/16=6 output patches,
    # the paper-recommended default for hourly-resolution data. Moirai is
    # probabilistic; the wrapper takes the median over num_samples=20 to
    # produce a point forecast in V13's (B, H, C) contract. See
    # `model/Moirai/moirai_wrapper.py` for details.
    #
    # Zero-shot ⇒ no training. is_zero_shot=True routes 02_train.py to
    # skip the train loop, mirroring TimesFM/TTM-r2.
    "Moirai": BackboneSpec(
        name="Moirai",
        model_factory=lambda cfg: _moirai.Model(cfg),
        default_model_hps=dict(
            # Wrapper hard-codes patch_size=16 and num_samples=20; both are
            # paper-standard for L=192/H=96 inference. Exposed here only so
            # train_config.json captures them.
            moirai_patch_size=16,
            moirai_num_samples=20,
        ),
        default_training_hps=dict(
            batch_size=64,                  # outer DataLoader batch
            learning_rate=0.0,              # unused (zero-shot)
            num_epochs=0,                   # unused (zero-shot)
            patience=0,
            optimizer="adam",
            scheduler="none",
        ),
        extra_config_fields=["moirai_patch_size", "moirai_num_samples"],
        forward_signature="tsl",
        is_zero_shot=True,
    ),
    # Future backbones (Chronos, Moirai-MoE, ...) get appended here.
}


def get_backbone(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise KeyError(
            f"Unknown backbone '{name}'. Registered: {sorted(BACKBONES.keys())}"
        )
    return BACKBONES[name]


def list_backbones() -> list[str]:
    return sorted(BACKBONES.keys())
