#!/bin/bash
# Two extra training seeds for EACH encoder's winning configuration (reads
# reports/baseline_sweep_aligned/<family>_winner.json written by
# promote_aligned.py). Requires the RNG seeding in train_neural.py — with it,
# --seed genuinely controls head init, dropout, and shuffling, so these are
# true seed-controlled replicas. The final evaluation rebuilds
# all_seeds.json with the RMA-only-fallback guard AND a between-run guard.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

lr_of () {  # lr_of <family> -> "5e-5" (validated against the grid)
  local raw
  raw=$($PY -c "import json;print(json.load(open('reports/baseline_sweep_aligned/$1_winner.json'))['winner'].split('_',1)[1])")
  raw="${raw#lr}"; raw="${raw/e/e-}"
  case "$raw" in
    2e-5|3e-5|5e-5) echo "$raw"; return 0 ;;
    *) echo "FATAL: $1 parsed lr '$raw' is not a valid grid value" >&2; return 1 ;;
  esac
}

run () {  # run <tag> <base_model> <lr> <seed>
  local tag=$1 base=$2 lr=$3 seed=$4
  local dir="models/sweep_aligned/$tag"
  if [ -f "reports/baseline_sweep_aligned/${tag}_fold.json" ]; then
    echo "=== SKIP $tag (done) ==="; return 0
  fi
  echo "=== TRAIN $tag seed=$seed lr=$lr $(date +%H:%M:%S) ==="
  $PY scripts/train_neural.py --base-model "$base" \
     --data data/train.conllu --dev-file data/dev.conllu \
     --folds 1 --epochs 12 --seed "$seed" --lr "$lr" --model-dir "$dir" \
     > "reports/baseline_sweep_aligned/${tag}.log" 2>&1
  if [ ! -f "$dir/fold_1/model.safetensors" ]; then
    echo "FATAL: $tag produced no checkpoint — see reports/baseline_sweep_aligned/${tag}.log" >&2
    exit 1
  fi
  grep -E "Epoch .*best|Mean" "reports/baseline_sweep_aligned/${tag}.log" | tail -3
  cp "$dir/cv_results.json" "reports/baseline_sweep_aligned/${tag}_cv.json" 2>/dev/null
  cp "$dir/fold_1/fold_metric.json" "reports/baseline_sweep_aligned/${tag}_fold.json" 2>/dev/null
  rm -f "$dir"/*.safetensors 2>/dev/null
}

KB_LR=$(lr_of kazbert) || exit 1
XLR=$(lr_of xlmr)     || exit 1
KR_LR=$(lr_of kazr)    || exit 1
echo "winners: kazbert=$KB_LR xlmr=$XLR kazr=$KR_LR"

run "kazbert_${KB_LR}_seed1" "Eraly-ml/KazBERT" "$KB_LR" 1
run "kazbert_${KB_LR}_seed2" "Eraly-ml/KazBERT" "$KB_LR" 2
run "xlmr_${XLR}_seed1" "FacebookAI/xlm-roberta-base" "$XLR" 1
run "xlmr_${XLR}_seed2" "FacebookAI/xlm-roberta-base" "$XLR" 2
run "kazr_${KR_LR}_seed1" "kz-transformers/kaz-roberta-conversational" "$KR_LR" 1
run "kazr_${KR_LR}_seed2" "kz-transformers/kaz-roberta-conversational" "$KR_LR" 2

$PY - <<EOF
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from kazmorph.pipeline import Pipeline
from evaluate import parse_conllu, evaluate_pipeline, mcnemar_test
sents = parse_conllu("data/test.conllu")
rma = [t["pred_pos"] for t in evaluate_pipeline(Pipeline(), sents)]
pb = Pipeline(crf_model_path="models/crf_model.joblib")
cb = [t["pred_pos"] == t["gold_pos"] for t in evaluate_pipeline(pb, sents)]
ENC = {
 "kazbert": [(42, "models/neural_pos"),
             (1, "models/sweep_aligned/kazbert_${KB_LR}_seed1/fold_1"),
             (2, "models/sweep_aligned/kazbert_${KB_LR}_seed2/fold_1")],
 "xlmr":    [(42, "models/neural_pos_xlmr"),
             (1, "models/sweep_aligned/xlmr_${XLR}_seed1/fold_1"),
             (2, "models/sweep_aligned/xlmr_${XLR}_seed2/fold_1")],
 "kazr":    [(42, "models/neural_pos_kazr"),
             (1, "models/sweep_aligned/kazr_${KR_LR}_seed1/fold_1"),
             (2, "models/sweep_aligned/kazr_${KR_LR}_seed2/fold_1")],
}
out, preds = {}, {}
for enc, runs in ENC.items():
    accs, mc = {}, {}
    for seed, d in runs:
        toks = evaluate_pipeline(Pipeline(neural_model_path=d, lemma_model_path="models/neural_lemma"), sents)
        pred = [t["pred_pos"] for t in toks]
        assert pred != rma, f"{enc} seed {seed}: RMA-only fallback detected!"
        for (oenc, oseed), other in preds.items():
            if oenc == enc:
                assert pred != other, (
                    f"{enc}: seeds {oseed} and {seed} produced identical "
                    "predictions — seeding not reaching the model!")
        preds[(enc, seed)] = pred
        accs[seed] = sum(p == t["gold_pos"] for p, t in zip(pred, toks)) / len(toks)
        mc[seed] = mcnemar_test(cb, [p == t["gold_pos"] for p, t in zip(pred, toks)])
    vals = list(accs.values()); mean = sum(vals)/len(vals)
    sd = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
    bs = max(accs, key=accs.get)
    out[enc] = {"accs": {str(k): v for k, v in accs.items()}, "mean": mean, "sd": sd}
    print(f"{enc:8s} seeds: {[f'{a*100:.2f}' for a in vals]}  mean={mean*100:.2f} sd={sd*100:.2f}  "
          f"best(seed {bs})-vs-B: chi2={mc[bs]['chi2']:.2f} p={mc[bs]['p_value']:.2e}")
json.dump(out, open("reports/baseline_sweep_aligned/all_seeds.json","w"), indent=2)
print("written reports/baseline_sweep_aligned/all_seeds.json")
EOF
