# Forecast Disagreement Anomaly Score (FDAS)
## Ground-Truth-Free Anomaly Detection for Multivariate Time Series

A recency-weighted multi-horizon **forecast-disagreement** score for
multivariate time-series anomaly detection.

A forecasting-based anomaly detection method that scores each timestep $t$
purely from the model's own multi-horizon forecast disagreement —
**without ever observing the actual value $y_t$ at scoring time**.
Backbone is **pluggable**: iTransformer is the reference implementation,
but any multi-horizon forecaster with a Time-Series-Library (TSL) style
forward signature can be slotted in with a 3-line registry entry. Evaluated
on TSB-AD-M (200 datasets, 17 families).

## Method overview

1. **Forecasting backbone** — multi-horizon forecaster taken verbatim from
   its source, no architectural changes. Default: **iTransformer**
   (Liu et al., ICLR 2024). The backbone is selected by `--backbone <name>`
   on every pipeline stage and is registered in
   [`model/__init__.py`](model/__init__.py); see [Adding a new backbone](#adding-a-new-backbone).
2. **Per-dataset training/inference**:
   - Train the backbone on the train split.
   - Run sliding-window inference (stride 1) on **train**, **val**, and
     **test** to get full-series predictions.
3. **Recency-weighted multi-horizon variance**. For each timestep $t$,
   collect the $H$ predictions of $t$ made by the $H$ different anchors
   $t-1, t-2, \ldots, t-H$. Compute their recency-weighted variance per
   channel:
   $$D_{w,c}(t) = \sum_{i=1}^H \tilde w_i \big(\hat y_t^{(t-i)}[c] - \bar v_t[c]\big)^2,
     \quad w_i = \lambda^{i-1}.$$
4. **Multivariate signal — per-channel time series**: each of the $C$
   channels yields a $D_{w,c}(t)$ time series.
5. **Channel z-score normalization**. The per-channel mean and std are
   computed once on the train evaluable region:
   $$z_c(t) = \frac{D_{w,c}(t) - \mu_c^{(\mathrm{train})}}{\sigma_c^{(\mathrm{train})}}.$$
6. **Channel aggregation = max**:
   $$\mathrm{score}(t) = \max_c z_c(t).$$
7. **Per-dataset evaluation on the full series** (train + test
   concatenated, TSB-AD-M aligned): VUS-PR, VUS-ROC, AUC-PR, AUC-ROC,
   Standard-F1, PA-F1.

The score is **ground-truth-free**: $(\mu_c, \sigma_c)$ are constants
fixed at training time, and no $y_t$ is read at scoring time. Steps 3-7
are completely backbone-agnostic — they only consume the `(N, H, C)`
prediction tensor that any registered backbone produces.

## Repository layout

```
.
├── README.md                              — this file
├── requirements.txt                       — Python deps
├── .gitignore                             — generated/large artifacts excluded
│
├── model/                                 — backbone registry + implementations
│   ├── __init__.py                        — BACKBONES dict (registry)
│   ├── base.py                            — BackboneSpec dataclass
│   └── iTransformer.py                    — vendored, untouched
├── layers/                                — TSL standard layers (vendored)
├── utils/                                 — vendored helpers
│
├── scripts/                               — production pipeline (all backbone-aware)
│   ├── artifact_paths.py                  — path helpers (per-backbone subdirs)
│   ├── config_factory.py                  — shared Config builder
│   ├── score_utils.py                     — D_w, z-score, TSB-AD wrappers
│   ├── 01_data_preparation.py             — split + StandardScaler + bundle_meta
│   ├── 02_train.py                        — backbone training (--backbone)
│   ├── 03_inference.py                    — train/val/test predictions (--backbone)
│   ├── 04_score_compute.py                — D_w + D_w_z (--backbone)
│   ├── 05_metrics.py                      — TSB-AD-M metrics on full series
│   ├── 06_cross_dataset.py                — cross-dataset summary stats
│   ├── 07_visualization.py                — figures
│   ├── run_all.py                         — end-to-end driver (--backbone)
│   └── migrate_to_backbone_layout.py      — one-time mv from legacy layout
│
├── ablations/                             — experimental scripts + outputs
│   ├── README.md
│   ├── scripts/
│   └── results/
│
├── datasets/                              — TSB-AD-M CSVs (or symlink)
├── data/                                  — preparation intermediates (gitignored)
├── models/<dataset_key>/<backbone>/       — checkpoints (gitignored)
│   ├── best_model.pth
│   ├── checkpoint.pth
│   ├── train_config.json
│   └── training_history.json
├── results/                               — outputs
│   ├── <dataset_key>/                     — per-dataset (backbone-agnostic)
│   │   ├── bundle_meta.json               — split + scaler used by 04/05
│   │   └── <backbone>/                    — per-backbone predictions + scores
│   │       ├── predictions_{train,val,test}.npy
│   │       ├── inference_metadata.json
│   │       ├── scores.parquet             — D_w + D_w_z, per timestep
│   │       └── scores_per_ch.npz          — per-channel D_w_c
│   ├── 04_metrics/
│   │   ├── per_dataset_metrics__<backbone>.csv  — main result CSV (200 × metrics)
│   │   └── metrics_tsb_format__<backbone>.csv   — TSB-AD-M benchmark format
│   ├── 05_cross_dataset/<backbone>/
│   ├── figures/<backbone>/
│   └── <backbone>/statistics_table.md
└── run_logs/<backbone>/                   — per-(backbone, key) training logs (gitignored)
```

Path helpers in [`scripts/artifact_paths.py`](scripts/artifact_paths.py)
take a `backbone` keyword argument; `bundle_meta.json` is the only
per-dataset artifact that lives outside the backbone subdirectory (it
describes the data split, which is backbone-independent).

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

**External dependency — TSB-AD** (evaluation metrics library; not on
PyPI). Install from source as a sibling clone:

```bash
git clone https://github.com/TheDatumOrg/TSB-AD ../TSB-AD
pip install -e ../TSB-AD
```

`scripts/05_metrics.py` calls `TSB_AD.evaluation.metrics.get_metrics`
for VUS-PR/ROC and other range-based metrics.

### Datasets

The TSB-AD-M dataset collection is **not bundled** with this repo
(several GB total). Download from the TSB-AD-M release and place the
CSV files under `datasets/`:

```bash
# Example layout under V13/datasets/
datasets/001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
datasets/002_MSL_id_1_Sensor_tr_500_1st_900.csv
...
```

Filenames follow the TSB-AD-M convention
`{id}_{family}_id_{entity}_{kind}_tr_{train_size}_1st_{first_anomaly}.csv`.

Source: https://github.com/TheDatumOrg/TSB-AD (`Datasets/TSB-AD-M/`).

### Pretrained checkpoints

The 200 trained iTransformer checkpoints (~5 GB total) are not in this
repo. Two options:

- **Reproduce locally** (~2-3 hours on a single GPU):
  ```bash
  python scripts/run_all.py --all-keys --skip-existing
  ```
- **Download pretrained**: (TBD — link to Hugging Face / Zenodo release
  once published). After download, place under `models/<dataset_key>/iTransformer/`.

### Run the full pipeline

`--backbone` defaults to `iTransformer`; pass it explicitly when running
multiple backbones side by side.

```bash
# Train + infer for every dataset (200 keys; ~2-3 hours on a single GPU)
python scripts/run_all.py --all-keys --skip-existing
# Equivalent: --backbone iTransformer

# Compute scores + metrics + figures
python scripts/run_all.py --analyze
```

Or stage by stage:

```bash
# Single dataset (data prep + train + infer)
python scripts/01_data_preparation.py --dataset-key Genesis
python scripts/02_train.py                     # default --backbone iTransformer
python scripts/03_inference.py

# Score + metrics + figures (across every dataset that has predictions)
python scripts/04_score_compute.py             # ~5 min  (workers=8)
python scripts/05_metrics.py                   # ~15 min
python scripts/06_cross_dataset.py
python scripts/07_visualization.py
```

### Inspect main results

`results/04_metrics/per_dataset_metrics__iTransformer.csv` has one row
per dataset with the 6 TSB-AD-M metrics for the production score
`D_w_z` (= `z_train_max`, the channel-z-scored, channel-max-aggregated
$D_w$). When you run additional backbones the file naming pattern is
`per_dataset_metrics__<backbone>.csv`, so per-backbone results never
clobber each other.

## Adding a new backbone

The full ablation table on a new forecaster is a 3-step change. No
modification to `01`, `04`, `05`, `06`, `07`, `score_utils.py`, or
`config_factory.py` is required — they're all backbone-agnostic.

**Step 1 — Drop the model file.** Add `model/<name>.py` with a
`Model(configs)` class following the Time-Series-Library forward
signature:

```python
class Model(nn.Module):
    def __init__(self, configs): ...
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # → Tensor[B, configs.pred_len, configs.c_out]
```

`configs` exposes `seq_len`, `pred_len`, `enc_in`, `dec_in`, `c_out`
(from the dataset) plus every key in your `default_model_hps` and
`default_training_hps` (from the registry entry below).

**Step 2 — Register in [`model/__init__.py`](model/__init__.py).**
Each registry entry bundles the model factory with its own paper-
recommended HPs (we deliberately *don't* unify training HPs across
backbones — every forecaster has its own well-tuned defaults):

```python
"DLinear": BackboneSpec(
    name="DLinear",
    model_factory=lambda cfg: _dlinear.Model(cfg),
    default_model_hps=dict(moving_avg=25),
    default_training_hps=dict(
        batch_size=32, learning_rate=1e-3, num_epochs=20,
        patience=3, optimizer="adam", scheduler="none",
    ),
    extra_config_fields=["moving_avg"],
    forward_signature="tsl",  # or "x_only" if forward(x_enc) only
),
```

`extra_config_fields` is the subset of attributes to persist into
`train_config.json` so inference can rebuild the exact same
architecture from disk without consulting the registry.

**Step 3 — Run the same pipeline with `--backbone <name>`.**

```bash
python scripts/run_all.py --all-keys --backbone DLinear --skip-existing
python scripts/run_all.py --analyze --backbone DLinear
```

Results land in `results/<dataset_key>/DLinear/...` and
`results/04_metrics/per_dataset_metrics__DLinear.csv`. iTransformer
artifacts are untouched, so you can compare side-by-side immediately.

## Key design choices

- **GT-free at scoring time**: $(\mu_c, \sigma_c)$ are constants saved
  alongside the model. No test-region observation is read by the score.
- **Train baseline (not val) for z-score**: the in-sample shrinkage of
  $\sigma_c^{\mathrm{train}}$ amplifies anomaly z-scores in the
  channel-max aggregation. Wins on all 6 metrics versus raw_max,
  z_median, z_mean (Wilcoxon p < 0.05; see `V14_RESULTS_REPORT.md` §4).
- **Channel max** > mean / median in the global average. Median wins
  in scale-shift-heavy domains (Exathlon family); see report §6.
- **Full-series evaluation** (train + test concatenated, TSB-AD-M
  aligned). No filter step.
- **Backbone-pluggable, HP-respecting**: each backbone keeps its own
  paper-recommended training HPs (lr / batch_size / epochs /
  scheduler). The narrative is *"FDAS works on top of each backbone
  trained at its own best HP"*, not *"FDAS works under a single
  arbitrary unified setup"*.

## Reports

- [`results/V15_RESULTS_REPORT.md`](results/V15_RESULTS_REPORT.md) —
  **current vintage** (backbone-pluggable framework; iTransformer
  results identical to V14, bit-for-bit verified).
- [`results/V14_RESULTS_REPORT.md`](results/V14_RESULTS_REPORT.md) —
  full-series eval, 200 datasets, 4-way agg comparison.
- [`results/V13_RESULTS_REPORT.md`](results/V13_RESULTS_REPORT.md) —
  earliest vintage (test-only eval, 9-variant comparison).

All current results are iTransformer-only. Cross-backbone reports
(DLinear, PatchTST, TimesNet, …) will be added as new backbones are
registered in `model/__init__.py`.

## Citation

```bibtex
@inproceedings{rwmfd-tba,
  title={...},
  author={...},
  booktitle={ICML},
  year={2026},
}
```

## License

(TBD — see `LICENSE` once added.)
