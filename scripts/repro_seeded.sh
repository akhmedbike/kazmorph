#!/bin/bash
# Full seeded reproduction of the aligned-protocol experiments (one command):
# archive the unseeded-era artifacts, then re-run LR sweep -> promote winners
# -> calibrate -> seed replicas -> full evaluation -> per-class McNemar ->
# 5-repeat benchmark -> figures, all under true RNG seeding (train_neural.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M)

echo "=== [0/8] archive unseeded-era artifacts ==="
if [ -d reports/baseline_sweep_aligned ]; then
  mv reports/baseline_sweep_aligned "reports/baseline_sweep_unseeded_$STAMP"
fi
if [ -d models/sweep_aligned ]; then
  mv models/sweep_aligned "models/sweep_unseeded_$STAMP"
fi
mkdir -p reports/baseline_sweep_aligned

echo "=== [1/8] LR sweep (9 runs, seed-controlled) ==="
bash scripts/sweep_aligned.sh || exit 1

echo "=== [2/8] promote winners ==="
$PY scripts/promote_aligned.py || exit 1

echo "=== [3/8] calibrate all three ==="
for m in neural_pos neural_pos_xlmr neural_pos_kazr; do
  $PY scripts/calibrate.py --model models/$m > /dev/null 2>&1 || { echo "calibrate $m FAILED"; exit 1; }
  echo "calibrated models/$m: $($PY -c "import json;c=json.load(open('models/$m/calibration.json'));print('T='+str(c['temperature']),'ECE',round(c['ece_before'],3),'->',round(c['ece_after'],3))")"
done

echo "=== [4/8] seed replicas (6 runs) + guarded all_seeds.json ==="
bash scripts/seeds_all.sh || exit 1

echo "=== [5/8] full A-H evaluation ==="
$PY scripts/evaluate.py --config all > reports/evaluate_seeded.log 2>&1 || exit 1
grep -E "^\s+\[" reports/evaluate_seeded.log | grep "Accuracy"

echo "=== [6/8] per-class McNemar (Family 2) ==="
$PY scripts/per_class_mcnemar.py 2>&1 | grep -vE "^Loading|it/s" | tail -12

echo "=== [7/8] benchmark x5 ==="
bash scripts/benchmark_x5.sh 2>&1 | grep -E "tp_median|written" || exit 1

echo "=== [8/8] figures ==="
$PY scripts/make_figures.py 2>&1 | tail -2

echo "=== SEEDED REPRODUCTION COMPLETE $(date +%H:%M:%S) ==="
