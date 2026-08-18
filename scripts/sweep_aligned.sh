#!/bin/bash
# Aligned-protocol LR sweep (closes review A1 + A2):
#   train = FULL data/train.conllu (754 sents / 7,376 tokens — identical to CRF),
#   dev   = data/dev.conllu (162 sents — identical to CRF's hyperparameter split),
#   grid  = {2,3,5}e-5 for ALL THREE encoders (incl. KazRoBERTa @5e-5, previously missing).
# Identical to the paper protocol otherwise: 12 epochs, patience 3, seed 42, batch 16.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p models/sweep_aligned reports/baseline_sweep_aligned

run () {  # run <tag> <base_model> <lr>
  local tag=$1 base=$2 lr=$3
  local dir="models/sweep_aligned/$tag"
  if [ -f "reports/baseline_sweep_aligned/${tag}_fold.json" ]; then
    echo "=== SKIP $tag (already done) ==="
    return 0
  fi
  echo "=== TRAIN $tag  base=$base  lr=$lr  $(date +%H:%M:%S) ==="
  $PY scripts/train_neural.py \
      --base-model "$base" \
      --data data/train.conllu --dev-file data/dev.conllu \
      --folds 1 --epochs 12 --seed 42 --lr "$lr" \
      --model-dir "$dir" \
      > "reports/baseline_sweep_aligned/${tag}.log" 2>&1
  local rc=$?
  grep -E "External dev|Train: .* sents|Epoch|Early stopping|Mean" "reports/baseline_sweep_aligned/${tag}.log" | tail -20
  if [ $rc -ne 0 ]; then echo "!!! $tag FAILED rc=$rc (see reports/baseline_sweep_aligned/${tag}.log)"; return $rc; fi
  cp "$dir/cv_results.json" "reports/baseline_sweep_aligned/${tag}_cv.json" 2>/dev/null
  cp "$dir/fold_1/fold_metric.json" "reports/baseline_sweep_aligned/${tag}_fold.json" 2>/dev/null
  # the promoted top-level checkpoint duplicates fold_1 — drop it to halve disk use
  rm -f "$dir"/*.safetensors "$dir"/model.safetensors.index.json 2>/dev/null
  return 0
}

run kazbert_lr2e5 "Eraly-ml/KazBERT" 2e-5
run kazbert_lr3e5 "Eraly-ml/KazBERT" 3e-5
run kazbert_lr5e5 "Eraly-ml/KazBERT" 5e-5
run xlmr_lr2e5    "FacebookAI/xlm-roberta-base" 2e-5
run xlmr_lr3e5    "FacebookAI/xlm-roberta-base" 3e-5
run xlmr_lr5e5    "FacebookAI/xlm-roberta-base" 5e-5
run kazr_lr2e5    "kz-transformers/kaz-roberta-conversational" 2e-5
run kazr_lr3e5    "kz-transformers/kaz-roberta-conversational" 3e-5
run kazr_lr5e5    "kz-transformers/kaz-roberta-conversational" 5e-5

echo ""
echo "=== SWEEP COMPLETE — dev.conllu weighted-F1 per run ==="
for f in reports/baseline_sweep_aligned/*_fold.json; do
  printf "%-45s %s\n" "$(basename "$f")" "$(tr -d '\n ' < "$f")"
done
