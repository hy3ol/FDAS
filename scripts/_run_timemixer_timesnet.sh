#!/bin/bash
# Convenience runner — full 200-dataset training + analysis for TimeMixer
# and TimesNet sequentially. Invoked via:
#   nohup bash scripts/_run_timemixer_timesnet.sh > run_logs/tm_tn_full_run.log 2>&1 &
set -e
cd /home/heeyeolkim/Projects/timeseries/models/00_Claude_Demo/V13
PY=/home/heeyeolkim/Projects/timeseries/models/00_Claude_Demo/.venv/bin/python

for BB in TimeMixer TimesNet; do
  echo "=== [$(date '+%F %T')] $BB: train + infer on 200 datasets ==="
  $PY scripts/run_all.py --all-keys --skip-existing --backbone "$BB"

  echo
  echo "=== [$(date '+%F %T')] $BB: analysis (04 → 05) ==="
  $PY scripts/run_all.py --analyze --backbone "$BB"

  echo
  echo "=== [$(date '+%F %T')] $BB done ==="
  echo
done

echo "=== [$(date '+%F %T')] All done. ==="
