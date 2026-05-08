# Ablations

Experimental scripts that **do not run as part of the main pipeline**.
They reproduce the per-component analyses cited in the V13/V14 results
reports. Each script reads the same artifacts produced by the
production pipeline (`../../results/{key}/predictions_*.npy`,
`scores_per_ch.npz`) and writes its own output CSV under
`./results/`.

All ablation scripts append `../../scripts/` to `sys.path` so they can
import `score_utils`, `artifact_paths`, etc. Run them from the
repository root:

```bash
cd <repo-root>
python ablations/scripts/<script_name>.py
```

## Scripts

| Script | Purpose | Cited in |
|:--|:--|:--|
| `_ablation_no_drop.py` | Decompose the gain of `z_train_max` into (a) z-score normalization vs (b) σ ≤ ε channel drop. | V13 §4.7 |
| `_ablation_channel_mask.py` | Isolate the effect of the channel mask alone (no z), as a *raw_max with σ-drop* control. | V13 §4.7 |
| `_ablation_zscore_agg_compare.py` | 4-way comparison of the channel aggregation step under z-score normalization (max / median / mean) vs raw_max. | V14 §4.1–4.3 |
| `compare_agg_normalize.py` | Full 9-variant grid: `{raw, z_train, z_val} × {max, median, mean}`. Used for V13 production-winner selection. | V13 §4.1 |

## Outputs

| CSV (under `./results/`) | Produced by | Schema notes |
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

It is computed by `scripts/04_score_compute.py` (no flag needed) and
stored in `results/{key}/scores.parquet` under the column `D_w_z`. The
ablation scripts here exist only to justify that choice.
