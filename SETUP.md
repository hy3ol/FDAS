# V13 / FDAS — Setup Guide

This repo trains six forecasting backbones and bolts on three pretrained
foundation-model backbones (TimesFM, TTM-r2, Moirai). The foundation-model
packages pin mutually incompatible torch versions in their metadata, so
**install order matters** — following these steps top-to-bottom avoids the
breakage paths.

## 1. Hardware

| Spec | Required | Recommended |
|---|---|---|
| GPU | CUDA-capable, ≥16 GB VRAM | RTX 4090 / 5090 / A100 (≥24 GB) |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB free | 1 TB (predictions are large, see note) |
| CUDA arch | sm_75+ | sm_120 (Blackwell) needs torch ≥ 2.8 (see below) |

> **NVIDIA Blackwell (RTX 5090, sm_120)**: torch < 2.5 ships with cu121
> binaries that lack `sm_120` kernels, so any GPU op crashes with
> `no kernel image is available for execution on the device`. You MUST
> use torch 2.8.0 + cu128. This repo's `requirements.txt` pins that.

> **Prediction artifacts are large.** `results/<dataset>/<backbone>/predictions_test.npy`
> for SWaT_id_2 is ~7 GB; across 200 datasets × 9 backbones the cumulative
> size is in the hundreds of GB. They're intermediate artifacts that 04
> consumes once — feel free to `find results -name 'predictions_*.npy' -delete`
> after analysis (see Section 8).

## 2. Python environment

```bash
# Python 3.10 recommended — matches what we've validated.
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3. torch first (Blackwell-capable)

```bash
pip install "torch==2.8.0" "torchvision==0.23.0"

# Verify GPU is reachable
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.8.0+cu128 True NVIDIA GeForce RTX 5090   (or your card)
```

## 4. Core requirements

```bash
pip install -r requirements.txt
```

This installs pandas, einops, scikit-learn, pyarrow, tqdm, matplotlib,
seaborn, **granite-tsfm** (TTM-r2), **timesfm**, and **uni2ts** (Moirai).
On a fresh environment, granite-tsfm and timesfm install cleanly.

> **uni2ts (Moirai) caveat.** uni2ts 2.0.0 declares `torch<2.5,>=2.1` and
> `einops==0.7.*` in its metadata, even though its code runs fine on
> torch 2.8 + einops 0.8. Plain `pip install uni2ts` will downgrade torch
> back to 2.4.x and break Blackwell again.
>
> Fix — reinstall uni2ts with `--no-deps` so it uses our newer torch/einops:
>
> ```bash
> pip install --no-deps "uni2ts==2.0.0"
> ```
>
> Verify (must say 2.8.0):
>
> ```bash
> python -c "import torch; import uni2ts.model.moirai as m; print('torch', torch.__version__, '/ moirai ok')"
> ```

## 5. TSB-AD evaluation library

`scripts/05_metrics.py` depends on TSB-AD for range-based metrics
(VUS-PR, VUS-ROC, etc.). Not on PyPI — install from source as a sibling
clone:

```bash
git clone https://github.com/TheDatumOrg/TSB-AD ../TSB-AD
pip install -e ../TSB-AD
```

## 6. Datasets

The 200 TSB-AD-M CSVs are not bundled (several GB). Place them under
`datasets/` (the existing entry is a symlink that you can replace):

```bash
ls datasets/   # should list 200 CSVs like 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
```

Source: https://github.com/TheDatumOrg/TSB-AD (`Datasets/TSB-AD-M/`).

## 7. Smoke test

A 30-second sanity check on Daphnet that exercises the full pipeline
including a foundation-model backbone:

```bash
python scripts/01_data_preparation.py --dataset-key Daphnet
python scripts/02_train.py --backbone Moirai     # zero-shot, no training
python scripts/03_inference.py --backbone Moirai --batch-size 16
python scripts/04_score_compute.py --backbone Moirai
ls results/Daphnet/Moirai/                       # should show predictions + scores
```

Substitute `iTransformer`, `TimeMixer`, etc. to test trained backbones
(they'll actually train for a few minutes per dataset).

## 8. Run the full benchmark for one backbone

Per-backbone full-benchmark scripts under `scripts/` handle the OPP
batch-size override (OPP_id_* has 248 channels and OOMs at the default
batch size for several backbones):

```bash
nohup bash scripts/_run_moirai_full.sh   > run_logs/moirai_full.log  2>&1 & disown
nohup bash scripts/_run_ttm_full.sh      > run_logs/ttm_full.log     2>&1 & disown
nohup bash scripts/_run_timesfm_full.sh  > run_logs/timesfm_full.log 2>&1 & disown
```

> ⚠️ **Don't run multiple `_run_*.sh` scripts in parallel.** The
> `V13/data/` directory is a shared staging area that each `01_data_preparation`
> call overwrites — concurrent runs corrupt each other. The runners are
> sequential by design.

For trained backbones (no per-backbone runner needed; `run_all.py`
suffices):

```bash
python scripts/run_all.py --all-keys --backbone iTransformer --skip-existing
python scripts/run_all.py --analyze   --backbone iTransformer
```

`run_all.py` now accepts `--batch-size` and forwards it to both
`02_train.py` and `03_inference.py`. Use it when a backbone needs a
smaller batch than the script's default of 64.

## 9. Disk cleanup after analysis

After `04_score_compute.py` finishes for a backbone, the predictions
`.npy` files are no longer needed (scores live in `scores.parquet` +
`scores_per_ch.npz`):

```bash
find results -maxdepth 3 -name "predictions_*.npy" -printf "%s\n" \
  | awk '{s+=$1} END {printf "Would free: %.1f GB\n", s/1024/1024/1024}'
find results -maxdepth 3 -name "predictions_*.npy" -delete
```

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` | torch built without sm_120 | Ensure `torch==2.8.0` (Step 3). |
| `Could not import module 'PreTrainedModel'` (from tsfm_public) | torchvision/torch version mismatch | Reinstall torchvision: `pip install --force-reinstall "torchvision==0.23.0"`. |
| `unrecognized arguments: --batch-size` from `run_all.py` | Older copy of `run_all.py` | Pull latest; this flag was added when foundation models needed it. |
| OOM on OPP_* at batch ≥ 16 | OPP has 248 channels; any-variate attention is O(C²) | The `_run_*.sh` scripts already drop batch to 4 or 8 for OPP; for ad-hoc runs pass `--batch-size 4`. |
| Moirai produces predictions with magnitudes ~1e6 | `num_samples=1` (single MC sample) is heavy-tailed | The shipped wrapper uses `num_samples=20` + median — don't lower this. |

## 11. What's NOT installed by `requirements.txt`

- **TSB-AD** — install separately from GitHub (Step 5).
- **Datasets** — see Step 6.
- **Pretrained foundation-model weights** — auto-downloaded from
  HuggingFace on first call (cached under `~/.cache/huggingface/`).
  First-run downloads: TTM-r2 ≈ 3 MB, Moirai ≈ 55 MB,
  TimesFM ≈ 800 MB.
