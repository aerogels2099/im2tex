#!/usr/bin/env bash
# Run one baseline over every test set (the same fixed subset for every model) and score it.
# Re-runnable: test sets already scored are skipped.
#   bash baselines/run_suite.sh got_ocr ~/venvs/im2tex/bin/python [limit=300]
set -euo pipefail
MODEL=$1 PY=$2 LIMIT=${3:-300}
D=${IM2TEX_DATA:-$HOME/im2tex-data}
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
SETS="unimer_spe:unimer/test/spe unimer_cpe:unimer/test/cpe unimer_sce:unimer/test/sce unimer_hwe:unimer/test/hwe
      crohme2014:crohme/2014 crohme2016:crohme/2016 crohme2019:crohme/2019
      mathwriting:mathwriting/prepared/test im2latex:im2latex100k/test"
for s in $SETS; do
  name=${s%%:*} dir=${s#*:}
  out=runs/$MODEL/$name.jsonl
  [ -f "runs/$MODEL/$name.metrics.json" ] && continue
  echo "== $MODEL / $name"
  "$PY" baselines/run_baseline.py "$MODEL" "$D/$dir" "$out" --limit "$LIMIT"
  "$HOME/venvs/im2tex/bin/python" scripts/evaluate.py "$out" > /dev/null
done
"$HOME/venvs/im2tex/bin/python" scripts/results_table.py runs
