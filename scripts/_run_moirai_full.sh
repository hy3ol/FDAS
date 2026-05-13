#!/bin/bash
# Full Moirai-1.1-R-small zero-shot run on TSB-AD-M.
#
# Moirai is *zero-shot* (no per-dataset training). 02_train.py detects
# BackboneSpec.is_zero_shot=True and skips the train loop — it only writes
# a minimal checkpoint.pth. The wrapper loads pretrained weights (~14M
# params, encoder-only any-variate Transformer) from the HF cache on each
# model instantiation, then forecasts the full (B, L, C) tensor in one
# pass via any-variate attention.
#
# Speed profile is C-dependent. Moirai's any-variate attention treats
# (timestep, channel) as a token, so attention compute is O((patches·C)²):
#   - C=9   (Daphnet etc.): ~1500 samples/s at batch=16 → minutes/dataset
#   - C=248 (OPP):          ~17  samples/s at batch=4  → ~50min/dataset
# Numerically, num_samples=20 + median(samples) is the paper-standard
# point estimator and gives bounded MSE (num_samples=1 was tried but
# produces heavy-tailed Monte Carlo outliers, max 1e7 vs p99.9=23).
#
# Batch sizing:
#   - Default (low/mid C): 16 — fits comfortably for C ≤ 100
#   - OPP (C=248):         4  — peak ~6.6GB at C=248, B=4
#
# Sequential — V13/data/ is a shared staging dir; never run two pipelines
# in parallel (see git log).
#
# Usage:  nohup bash scripts/_run_moirai_full.sh > run_logs/moirai_full.log 2>&1 & disown
set -e
cd /home/heeyeolkim/Projects/timeseries/models/00_Claude_Demo/V13
PY=/home/heeyeolkim/Projects/timeseries/models/00_Claude_Demo/.venv/bin/python

# Build dataset key list, drop OPPORTUNITY_*.
KEYS=$($PY -c "
import sys
sys.path.append('scripts')
from artifact_paths import list_available_dataset_keys
keys = [k for k in list_available_dataset_keys() if not k.startswith('OPPORTUNITY')]
print(' '.join(keys))
")
N=$(echo $KEYS | wc -w)
echo "=== [$(date '+%F %T')] Moirai-1.1-R zero-shot: $N datasets at batch=16 (OPP excluded) ==="
$PY scripts/run_all.py --datasets $KEYS --skip-existing --backbone Moirai --batch-size 16

echo
echo "=== [$(date '+%F %T')] Moirai-1.1-R zero-shot: OPP 8 datasets at batch=4 ==="
for DS in OPPORTUNITY_id_1 OPPORTUNITY_id_2 OPPORTUNITY_id_3 OPPORTUNITY_id_4 \
          OPPORTUNITY_id_5 OPPORTUNITY_id_6 OPPORTUNITY_id_7 OPPORTUNITY_id_8; do
  echo "--- [$(date '+%F %T')] $DS (Moirai, batch=4) ---"
  $PY scripts/01_data_preparation.py --dataset-key "$DS"
  $PY scripts/02_train.py --backbone Moirai --batch-size 4
  $PY scripts/03_inference.py --backbone Moirai --batch-size 4
done

echo
echo "=== [$(date '+%F %T')] Moirai: analysis (04 → 05) ==="
$PY scripts/run_all.py --analyze --backbone Moirai

echo "=== [$(date '+%F %T')] All done. ==="
