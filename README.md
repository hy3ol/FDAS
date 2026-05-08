# Recency-Weighted Multi-Horizon Forecast Disagreement (RWMFD)
## Ground-Truth-Free Anomaly Detection for Multivariate Time Series

A forecasting-based anomaly detection method that scores each timestep $t$
purely from the model's own multi-horizon forecast disagreement —
**without ever observing the actual value $y_t$ at scoring time**. Built
on iTransformer; evaluated on TSB-AD-M (200 datasets, 17 families).

## Method overview

1. **Forecasting backbone**: iTransformer (Liu et al., 2024) — taken
   verbatim, no architectural changes. Pluggable: any multi-horizon
   forecaster with the same I/O contract works.
2. **Per-dataset training/inference**:
   - Train iTransformer on the train split.
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
fixed at training time, and no $y_t$ is read at scoring time.

## Repository layout

```
.
├── README.md                       — this file
├── requirements.txt                — Python deps
├── .gitignore                      — generated/large artifacts excluded
│
├── model/                          — iTransformer (vendored, untouched)
│   └── iTransformer.py
├── layers/                         — iTransformer layers (vendored)
├── utils/                          — iTransformer utils (vendored)
│
├── scripts/                        — production pipeline
│   ├── artifact_paths.py           — path helpers
│   ├── score_utils.py              — D_w, z-score, TSB-AD wrappers
│   ├── 01_data_preparation.py      — split + StandardScaler + bundle_meta
│   ├── 02_train.py                 — iTransformer training (early-stop)
│   ├── 03_inference.py             — train/val/test predictions
│   ├── 04_score_compute.py         — D_w + D_w_z (production score)
│   ├── 05_metrics.py               — TSB-AD-M metrics on full series
│   ├── 06_cross_dataset.py         — cross-dataset summary stats
│   ├── 07_visualization.py         — figures
│   └── run_all.py                  — end-to-end driver
│
├── ablations/                      — experimental scripts + outputs
│   ├── README.md
│   ├── scripts/
│   │   ├── _ablation_no_drop.py
│   │   ├── _ablation_channel_mask.py
│   │   ├── _ablation_zscore_agg_compare.py
│   │   └── compare_agg_normalize.py
│   └── results/
│
├── datasets/                       — TSB-AD-M CSVs (or symlink)
├── data/                           — preparation intermediates (gitignored)
├── models/                         — checkpoints (gitignored)
├── results/                        — outputs
│   ├── 04_metrics/
│   │   ├── per_dataset_metrics.csv — main result CSV (200 × metrics)
│   │   └── metrics_tsb_format.csv  — TSB-AD-M benchmark format
│   ├── 05_cross_dataset/
│   ├── figures/
│   ├── V13_RESULTS_REPORT.md       — test-only vintage report
│   ├── V14_RESULTS_REPORT.md       — full-series vintage report (current)
│   └── {key}/                       — per-dataset (predictions, scores)
└── run_logs/                       — per-key training logs (gitignored)
```

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
  once published).

### Run the full pipeline

```bash
# Train + infer for every dataset (200 keys; ~2-3 hours on a single GPU)
python scripts/run_all.py --all-keys --skip-existing

# Compute scores + metrics + figures (no filter step)
python scripts/run_all.py --analyze
```

Or stage by stage:

```bash
# Single dataset (data prep + train + infer)
python scripts/01_data_preparation.py --dataset-key Genesis
python scripts/02_train.py
python scripts/03_inference.py

# Score + metrics + figures (across every dataset that has predictions)
python scripts/04_score_compute.py            # ~5 min  (workers=8)
python scripts/05_metrics.py                  # ~15 min
python scripts/06_cross_dataset.py
python scripts/07_visualization.py
```

### Inspect main results

`results/04_metrics/per_dataset_metrics.csv` has one row per dataset
with the 6 TSB-AD-M metrics for the production score `D_w_z`
(= `z_max`, the channel-z-scored, channel-max-aggregated $D_w$).

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

## Reports

- [`results/V14_RESULTS_REPORT.md`](results/V14_RESULTS_REPORT.md) —
  current vintage (full-series, 200 datasets, 4-way agg comparison).
- [`results/V13_RESULTS_REPORT.md`](results/V13_RESULTS_REPORT.md) —
  earlier vintage (test-only eval, 9-variant comparison).

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
