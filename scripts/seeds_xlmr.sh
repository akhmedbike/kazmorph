#!/bin/bash
# Two extra training seeds for the WINNING XLM-R configuration (paper A10):
# quantifies run-to-run variance of the headline baseline. The winning LR is
# read from reports/baseline_sweep_aligned/xlmr_winner.json (written by
# scripts/promote_aligned.py). Each seed is then evaluated on the test split
# with the same Pipeline wiring as configuration G.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# winner looks like "xlmr_lr5e5"; --lr needs "5e-5"
RAW=$($PY -c "import json;print(json.load(open('reports/baseline_sweep_aligned/xlmr_winner.json'))['winner'].split('_',1)[1])")
LR="${RAW#lr}"; LR="${LR/e/e-}"
echo "winning XLM-R lr = $LR"
case "$LR" in
  2e-5|3e-5|5e-5) ;;
  *) echo "FATAL: parsed lr '$LR' is not a valid grid value" >&2; exit 1 ;;
esac

for SEED in 1 2; do
  TAG="xlmr_${LR}_seed${SEED}"
  DIR="models/sweep_aligned/$TAG"
  if [ -f "reports/baseline_sweep_aligned/${TAG}_fold.json" ]; then
    echo "=== SKIP $TAG (done) ==="; continue
  fi
  echo "=== TRAIN $TAG seed=$SEED $(date +%H:%M:%S) ==="
  $PY scripts/train_neural.py --base-model "FacebookAI/xlm-roberta-base" \
     --data data/train.conllu --dev-file data/dev.conllu \
     --folds 1 --epochs 12 --seed "$SEED" --lr "$LR" --model-dir "$DIR" \
     > "reports/baseline_sweep_aligned/${TAG}.log" 2>&1
  if [ ! -f "$DIR/fold_1/model.safetensors" ]; then
    echo "FATAL: $TAG produced no checkpoint — see reports/baseline_sweep_aligned/${TAG}.log" >&2
    exit 1
  fi
  grep -E "Epoch .*best|Mean" "reports/baseline_sweep_aligned/${TAG}.log" | tail -4
  cp "$DIR/cv_results.json" "reports/baseline_sweep_aligned/${TAG}_cv.json" 2>/dev/null
  cp "$DIR/fold_1/fold_metric.json" "reports/baseline_sweep_aligned/${TAG}_fold.json" 2>/dev/null
done

# Test-split accuracy per seed (seed 42 = the sweep winner, already canonical)
$PY - <<EOF
import sys, json, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from kazmorph.pipeline import Pipeline
from evaluate import parse_conllu, evaluate_pipeline
sents = parse_conllu("data/test.conllu")
# RMA-only reference: catches the silent no-model fallback that once produced
# phantom "seed" accuracies identical to configuration A.
rma_only = Pipeline()
toks_rma = evaluate_pipeline(rma_only, sents)
pred_rma = [t["pred_pos"] for t in toks_rma]
accs = {}
for seed, d in [(42, "models/neural_pos_xlmr"),
                (1, "models/sweep_aligned/xlmr_${LR}_seed1/fold_1"),
                (2, "models/sweep_aligned/xlmr_${LR}_seed2/fold_1")]:
    assert os.path.exists(os.path.join(d, "model.safetensors")), f"missing model: {d}"
    pipe = Pipeline(neural_model_path=d, lemma_model_path="models/neural_lemma")
    toks = evaluate_pipeline(pipe, sents)
    pred = [t["pred_pos"] for t in toks]
    assert pred != pred_rma, f"seed {seed}: predictions identical to RMA-only — model did not load!"
    accs[seed] = sum(t["pred_pos"] == t["gold_pos"] for t in toks) / len(toks)
    print(f"seed {seed}: test UPOS accuracy = {accs[seed]*100:.2f}%")
vals = list(accs.values())
mean = sum(vals) / len(vals)
sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
print(f"mean = {mean*100:.2f}%  sd = {sd*100:.2f} pp")
json.dump({"lr": "$LR", "accs": accs, "mean": mean, "sd": sd},
          open("reports/baseline_sweep_aligned/xlmr_seeds.json", "w"), indent=2)
EOF
echo "written reports/baseline_sweep_aligned/xlmr_seeds.json"
