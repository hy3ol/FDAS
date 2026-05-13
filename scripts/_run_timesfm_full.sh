#!/bin/bash
# Full TimesFM-1.0-200M zero-shot run on TSB-AD-M.
#
# TimesFM is *zero-shot* (no per-dataset training). 02_train.py detects
# BackboneSpec.is_zero_shot=True and skips the train loop — it only writes
# a minimal checkpoint.pth so the pipeline shape stays consistent with the
# other backbones. 03_inference.py loads pretrained weights from the HF
# cache (~200M parameters) and forecasts each channel independently.
#
# OPP 8 datasets use batch=8 because per_core_batch_size × num_channels
# inflates memory for 248-channel inputs (same root cause as TimeMixer/
# TimesNet/TimeXer OPP).
#
# Sequential — V13/data/ is a shared staging dir; never run two pipelines
# in parallel (see git log).
#
# Usage:  nohup bash scripts/_run_timesfm_full.sh > run_logs/timesfm_full.log 2>&1 & disown
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
echo "=== [$(date '+%F %T')] TimesFM zero-shot: $N datasets at default batch (OPP excluded) ==="
$PY scripts/run_all.py --datasets $KEYS --skip-existing --backbone TimesFM

echo
echo "=== [$(date '+%F %T')] TimesFM zero-shot: OPP 8 datasets at batch=8 ==="
for DS in OPPORTUNITY_id_1 OPPORTUNITY_id_2 OPPORTUNITY_id_3 OPPORTUNITY_id_4 \
          OPPORTUNITY_id_5 OPPORTUNITY_id_6 OPPORTUNITY_id_7 OPPORTUNITY_id_8; do
  echo "--- [$(date '+%F %T')] $DS (TimesFM, batch=8) ---"
  $PY scripts/01_data_preparation.py --dataset-key "$DS"
  $PY scripts/02_train.py --backbone TimesFM --batch-size 8
  $PY scripts/03_inference.py --backbone TimesFM --batch-size 8
done

echo
echo "=== [$(date '+%F %T')] TimesFM: analysis (04 → 05) ==="
$PY scripts/run_all.py --analyze --backbone TimesFM

echo "=== [$(date '+%F %T')] All done. ==="
