#!/bin/bash
# Convenience runner — full 200-dataset training + analysis for PatchTST.
# Invoked via:  nohup bash scripts/_run_patchtst.sh > run_logs/patchtst_full_run.log 2>&1 &
set -e
cd /home/heeyeolkim/Projects/timeseries/models/00_Claude_Demo/V13
PY=/home/heeyeolkim/Projects/timeseries/models/00_Claude_Demo/.venv/bin/python

echo "=== [$(date '+%F %T')] PatchTST: train + infer on 200 datasets ==="
$PY scripts/run_all.py --all-keys --skip-existing --backbone PatchTST

echo
echo "=== [$(date '+%F %T')] PatchTST: analysis (04→05→06→07) ==="
$PY scripts/run_all.py --analyze --backbone PatchTST

echo
echo "=== [$(date '+%F %T')] Done. ==="
