# Ablations

Experimental scripts that **do not run as part of the main pipeline**.
They reproduce the per-component analyses cited in the V13/V14 results
reports. Each script reads the same artifacts produced by the
production pipeline (`../../results/<dataset_key>/<backbone>/predictions_*.npy`,
`scores_per_ch.npz`) and writes its own output CSV under `./results/`.

All ablation scripts append `../../scripts/` to `sys.path` so they can
import `score_utils`, `artifact_paths`, etc. Run them from the
repository root:

```bash
cd <repo-root>
python ablations/scripts/<script_name>.py
```

> **Backbone scope.** These ablation scripts were written before the
> backbone-pluggable refactor. They currently read iTransformer
> artifacts from the legacy path resolution; results files have no
> per-backbone suffix. If you want to re-run them on a different
> backbone, point the path helpers (or the script's hardcoded glob) at
> `results/<dataset_key>/<backbone>/...` and rename the output CSV
> accordingly to avoid clobbering the iTransformer baseline.

## Scripts

| Script | Purpose | Cited in |
|:--|:--|:--|
| `_ablation_no_drop.py` | Decompose the gain of `z_train_max` into (a) z-score normalization vs (b) σ ≤ ε channel drop. | V13 §4.7 |
| `_ablation_channel_mask.py` | Isolate the effect of the channel mask alone (no z), as a *raw_max with σ-drop* control. | V13 §4.7 |
| `_ablation_zscore_agg_compare.py` | 4-way comparison of the channel aggregation step under z-score normalization (max / median / mean) vs raw_max. | V14 §4.1–4.3 |
| `compare_agg_normalize.py` | Full 9-variant grid: `{raw, z_train, z_val} × {max, median, mean}`. Used for V13 production-winner selection. | V13 §4.1 |
| `horizon_ablation/run_horizon_ablation.py` | Sweep prediction horizon H ∈ {192, 336} for iTransformer (L=192 fixed). Full pipeline per (H, dataset). | new |
| `horizon_ablation/aggregate.py` | Combine production H=96 + ablation H∈{192,336} into a single comparison summary + per-H Wilcoxon vs H=96. | new |

## Horizon ablation (iTransformer, L=192, H ∈ {192, 336})

This sweep keeps lookback fixed at 192 and varies pred_len, asking how FDAS
accuracy scales with forecast horizon under the production iTransformer
backbone. Production H=96 results in `results/04_metrics/iTransformer/` are
the baseline; this ablation adds H=192 and H=336 in an isolated tree.

**Design choice — wrapper-only, zero production code change.**
The full pipeline (01 split → 02 train → 03 infer → 04 score → 05 metrics) is
already encoded in `scripts/`. `01_data_preparation.py` hardcodes `PRED_LEN=96`
and `04_score_compute.py` hardcodes `pred_len=96` at one call site, so simply
running them as-is with a different H is not possible. Rather than modify those
production scripts (which would alter the paper-grade frozen pipeline), the
horizon ablation lives entirely under `ablations/scripts/horizon_ablation/`:

- `_split.py` reimplements 01's split + StandardScaler with explicit
  `(lookback, pred_len)`. Writes `data/` for 02/03 to consume.
- 02/03 are invoked as subprocesses (unchanged) and write into the production
  singleton `models/<key>/iTransformer/` and `results/<key>/iTransformer/`.
- `run_horizon_ablation.py` backs up any pre-existing H=96 artifacts in those
  locations to `<name>.h96.bak` *before* the subprocesses run, then moves the
  new H≠96 artifacts to `ablations/results/horizon/H<H>/...`, then restores
  the H=96 backups. Wrapped in try/finally so the H=96 originals are recovered
  even if 02/03 fails mid-run.
- `_score.py` reimplements 04's process_one with `pred_len` plumbed correctly
  (production 04 hardcodes 96 at the call site to `compute_train_baseline_stats`).
  Monkey-patches `artifact_paths.RESULTS_ROOT` for the duration of the call so
  `score_utils.prepare_dataset_bundle` reads the ablation-tree bundle_meta.
- 05_metrics.process_one is reused verbatim through the same RESULTS_ROOT patch.

### Run

```bash
# Default sweep is H ∈ {192, 336}; add 48 explicitly for the full grid.
# ~3-4 hours on a single GPU; H=48 alone is ~90 min.
python ablations/scripts/horizon_ablation/run_horizon_ablation.py --pred-lens 48 192 336

# Single H, single dataset (smoke test)
python ablations/scripts/horizon_ablation/run_horizon_ablation.py \
    --pred-lens 192 --only MSL_id_15

# Resume after interruption — merges with existing run_log.csv
python ablations/scripts/horizon_ablation/run_horizon_ablation.py --skip-existing

# Aggregate H=96 (production) + H ∈ {192, 336} into summary tables
python ablations/scripts/horizon_ablation/aggregate.py
```

### Outputs

```
ablations/results/horizon/
├── H48/                                 — see H192/ layout
├── H192/
│   ├── <key>/iTransformer/   (predictions_*.npy, scores.parquet, scores_per_ch.npz)
│   ├── <key>/bundle_meta.json
│   ├── models/<key>/iTransformer/best_model.pth + train_config.json
│   ├── _logs/<key>.log
│   └── per_dataset_metrics.csv         — H=192 slice
├── H336/                                — same layout
├── run_log.csv                          — (H × key) status + per-phase timing
├── per_dataset_metrics.csv              — (H × key) × 6 TSB-AD-M metrics
├── summary.csv                          — H ∈ {48, 96, 192, 336} long-format
└── summary_aggregate.csv                — per-H mean/median + Wilcoxon vs H=96
```

### Fittable dataset count per H

Each split needs ≥ `L + H` consecutive timesteps. Datasets shorter than that
in any of train/val/test are skipped with status `skip_too_short`.

| H | required_span (= L+H) | fittable / 200 | notes |
|---|---|---|---|
| **12** | 204 | 200 | |
| **24** | 216 | 200 | |
| **48** | 240 | 200 | |
| 96 (production) | 288 | 200 | |
| 192 | 384 | 200 | |
| 336 | 528 | 173 | 26 `skip_too_short` (train_size=500 datasets); 1 `skip_oom` (SWaT\_id\_2: predictions tensor ~13 GB × 3 splits > 60 GB RAM). |

Per-H summary tables are computed over each $n_\textrm{ok}$ subset; rows
are aligned by `dataset_key` so paired Wilcoxon vs H=96 in `summary_aggregate.csv`
uses only datasets that succeeded at both H values.

### Results — U-curve, peak at H=24

Headline VUS-PR (micro-average across $n_\textrm{ok}$ datasets):

| H | $n_\textrm{ok}$ | VUS-PR | Δ vs H=96 | paired Wilcoxon p | dataset win/loss vs H=96 |
|---|:--:|---:|---:|---:|---:|
| 12 | 200 | 0.3334 | +0.0305 | 1.6e-05 | 133 / 67 |
| **24** | 200 | **0.3414** | **+0.0385** | 2.0e-09 | **142 / 58** |
| 48 | 200 | 0.3243 | +0.0215 | 4.6e-07 | 135 / 65 |
| 96 (production) | 200 | 0.3028 | — | — | — |
| 192 | 200 | 0.2673 | −0.0355 | 3.8e-11 | 63 / 137 |
| 336 | 173 | 0.2149 | −0.0592 | 4.0e-12 | 48 / 125 |

**6 / 6 TSB-AD-M metrics form a U-curve with peak at H=24**. H=12 → H=24 →
H=48 → H=96 → H=192 → H=336 is non-monotonic only at the very-short end:
H=12 vs H=24 paired Wilcoxon is non-significant across all 6 metrics
(p > 0.2 for all, $|\Delta\,\textrm{VUS-PR}| = 0.008$), so the
recency-weighted multi-horizon variance is roughly *flat* between H=12
and H=24 and then **decreases monotonically with H** beyond H=24.

Conclusion: the current production choice of H=96 is suboptimal — **H=24
yields +0.039 VUS-PR (≈ +13 % relative) with no extra inference cost,
winning 142/200 paired datasets and beating H=96 on all 6 metrics at
$p < 10^{-4}$**. The optimum sits between the "too short → not enough
anchors for a meaningful per-step variance estimate" regime (H ≤ 12,
recency weight $\lambda^H \gtrsim 0.89$ saturates) and the "too long →
recency weighting collapses" regime (H ≥ 192, $\lambda^H \lesssim 0.15$).
At H=24, $\lambda^{24} \approx 0.79$ — recency is still meaningfully
emphasised, while 24 anchors per timestep give a reasonably-stable
variance estimate.

See `ablations/results/table/horizon_table_{1,2}.{tex,pdf,png}` for the
paper-grade tables (overall 6-metric comparison + per-family VUS-PR
breakdown across all six horizons).

### Methodological note

For H=720 we did not sweep — `λ^720 ≈ 0.0007` reduces the recency weighting in
$D_{w,c}(t) = \sum_h \tilde w_h (\hat y_t^{(t-h)}[c] - \bar v_t[c])^2$ to near-uniform,
collapsing the recency-weighted variance to plain variance over a long horizon.
H=336 (`λ^336 ≈ 0.034`) is already at the edge of the regime where recency
weighting is meaningful at λ=0.99.

At the short end, H=12 is roughly tied with the optimum H=24 (no
significant difference at $p > 0.2$) but slightly lower-mean, suggesting
that the 12-anchor variance estimate is the noisier of the two without
gaining anything from the larger recency weight $\lambda^{12} \approx 0.89$
vs $\lambda^{24} \approx 0.79$. Going below H=12 (e.g. H=6) would
risk degenerating into something close to single-step error, since the
variance estimate has too few participating predictions per timestep.
We did not sweep below H=12.

## Outputs

| CSV (under `./results/etc/`) | Produced by | Schema notes |
|:--|:--|:--|
| `agg_normalize_per_dataset.csv` | `compare_agg_normalize.py` | Per-dataset × 9 variants × 6 metrics. |
| `agg_normalize_summary.csv` | `compare_agg_normalize.py` | Cross-dataset means + Wilcoxon. |
| `ablation_no_drop_per_dataset.csv` | `_ablation_no_drop.py` | Per-dataset, no-drop counterpart of agg_normalize. |
| `ablation_no_drop_summary.csv` | `_ablation_no_drop.py` | Cross-dataset summary. |
| `ablation_zscore_agg_compare.csv` | `_ablation_zscore_agg_compare.py` | Per-dataset × {max, median, mean}. |

## Production score (for reference)

The production score reported in the main paper is `D_w_z` =
`z_train_max`:

> Per-channel recency-weighted forecast variance → channel z-score
> against train baseline (μ_c, σ_c) → channel-max aggregation.

It is computed by `scripts/04_score_compute.py` (no flag needed, but
respects `--backbone <name>`) and stored in
`results/<dataset_key>/<backbone>/scores.parquet` under the column
`D_w_z`. The ablation scripts here exist only to justify that choice
within the iTransformer baseline.
