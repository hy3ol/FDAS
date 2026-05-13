#!/bin/bash
# Full TTM-r2 (Tiny Time Mixer) zero-shot run on TSB-AD-M.
#
# TTM-r2 is *zero-shot* (no per-dataset training). 02_train.py detects
# BackboneSpec.is_zero_shot=True and skips the train loop — it only writes
# a minimal checkpoint.pth so the pipeline shape stays consistent. The
# wrapper loads pretrained weights (~805K params, 512-96-r2 variant) from
# the HF cache on each model instantiation, then left-zero-pads V13's
# L=192 window up to TTM's 512 context with past_observed_mask=1 only on
# the last 192 positions (so TTM's scaler ignores the pad).
#
# Unlike TimesFM (univariate, channel-wise looped → memory scales with
# batch × num_channels), TTM-r2 is multivariate-native (one forward per
# window regardless of C). So OPP's 248-channel input does NOT inflate
# memory the same way — we drop to batch=32 only as a conservative guard.
#
# Sequential — V13/data/ is a shared staging dir; never run two pipelines
# in parallel (see git log).
#
# Usage:  nohup bash scripts/_run_ttm_full.sh > run_logs/ttm_full.log 2>&1 & disown
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
echo "=== [$(date '+%F %T')] TTM-r2 zero-shot: $N datasets at default batch (OPP excluded) ==="
$PY scripts/run_all.py --datasets $KEYS --skip-existing --backbone TTM

echo
echo "=== [$(date '+%F %T')] TTM-r2 zero-shot: OPP 8 datasets at batch=32 ==="
for DS in OPPORTUNITY_id_1 OPPORTUNITY_id_2 OPPORTUNITY_id_3 OPPORTUNITY_id_4 \
          OPPORTUNITY_id_5 OPPORTUNITY_id_6 OPPORTUNITY_id_7 OPPORTUNITY_id_8; do
  echo "--- [$(date '+%F %T')] $DS (TTM, batch=32) ---"
  $PY scripts/01_data_preparation.py --dataset-key "$DS"
  $PY scripts/02_train.py --backbone TTM --batch-size 32
  $PY scripts/03_inference.py --backbone TTM --batch-size 32
done

echo
echo "=== [$(date '+%F %T')] TTM: analysis (04 → 05) ==="
$PY scripts/run_all.py --analyze --backbone TTM

echo "=== [$(date '+%F %T')] All done. ==="
