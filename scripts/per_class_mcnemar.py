#!/usr/bin/env python3
"""Per-error-class McNemar tests: RMA+CRF (B) vs each transformer baseline.

Uses the same fixed, model-independent taxonomy partition as evaluate.py
(standalone RMA classifier, θ = 0.30), so per-class token sets are identical
across configurations and directly comparable.

Statistical protocol (paper §4): two declared families of McNemar tests.
  Family 1 — all pairwise configuration comparisons (C(8,2) = 28), Holm-corrected
             inside scripts/evaluate.py's report.
  Family 2 — THIS script: per-class tests B vs each transformer baseline
             (classes BOR/AFC/NONE with n ≥ 100 × available baselines among
             C/G/H), Holm-corrected within this family only. Rare classes
             (SEG n = 17, AGR n = 16) are excluded: McNemar on <20 discordant
             tokens is meaningless.

Usage:
    python3 scripts/per_class_mcnemar.py [--test-file data/test.conllu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from kazmorph.pipeline import Pipeline
from kazmorph.rma import ReverseMorphAnalyzer
from kazmorph.taxonomy import classify_scored
from evaluate import parse_conllu, evaluate_pipeline, mcnemar_test

CLASSES = ("BOR", "AFC", "NONE")  # n ≥ 100 on the fixed partition
RARE = ("SEG", "AGR")             # reported descriptively only


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down)."""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(order):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-file", default="data/test.conllu")
    ap.add_argument("--out", default="reports/per_class_mcnemar.json")
    args = ap.parse_args()

    sentences = parse_conllu(args.test_file)

    # Fixed partition (identical to evaluate.py / sensitivity.py)
    standalone_rma = ReverseMorphAnalyzer()
    fixed_classes = [
        classify_scored(standalone_rma.analyze(t["form"])).difficulty_class
        for s in sentences for t in s
    ]
    counts = Counter(fixed_classes)
    print("Fixed partition:", ", ".join(f"{c}={counts[c]}" for c in (*CLASSES, *RARE)))

    crf = "models/crf_model.joblib"
    baselines = {
        "C (KazBERT)": ("models/neural_pos", "models/neural_lemma"),
        "G (XLM-R)": ("models/neural_pos_xlmr", "models/neural_lemma"),
        "H (KazRoBERTa)": ("models/neural_pos_kazr", "models/neural_lemma"),
    }

    print("Running B (RMA+CRF) ...")
    pipe_b = Pipeline(crf_model_path=crf)
    toks_b = evaluate_pipeline(pipe_b, sentences)
    correct_b = [t["pred_pos"] == t["gold_pos"] for t in toks_b]

    results: dict[str, dict] = {}
    for name, (neural_path, lemma_path) in baselines.items():
        if not os.path.exists(os.path.join(neural_path, "config.json")):
            print(f"Skipping {name}: {neural_path} not trained")
            continue
        print(f"Running {name} ...")
        pipe = Pipeline(neural_model_path=neural_path,
                        lemma_model_path=lemma_path if os.path.exists(lemma_path) else None)
        toks = evaluate_pipeline(pipe, sentences)
        correct = [t["pred_pos"] == t["gold_pos"] for t in toks]
        assert len(correct) == len(correct_b) == len(fixed_classes)
        results[name] = {}
        for cls in CLASSES:
            sel = [i for i, c in enumerate(fixed_classes) if c == cls]
            m = mcnemar_test([correct_b[i] for i in sel], [correct[i] for i in sel])
            acc_b = sum(correct_b[i] for i in sel) / len(sel)
            acc_n = sum(correct[i] for i in sel) / len(sel)
            results[name][cls] = {"acc_B": round(acc_b, 4),
                                  "acc_baseline": round(acc_n, 4), **m}

    # Holm within Family 2 (all per-class tests computed above)
    flat = {f"{name} · {cls}": v["p_value"]
            for name, per in results.items() for cls, v in per.items()}
    adjusted = holm(flat)
    print("\nFamily 2 — per-class McNemar, B vs each transformer (Holm-adjusted):")
    for name, per in results.items():
        for cls, v in per.items():
            key = f"{name} · {cls}"
            print(f"  {key:26s} acc {v['acc_B']:.3f} vs {v['acc_baseline']:.3f}  "
                  f"n01(B+/{name.split()[0]}-)={v['n01']}  n10={v['n10']}  "
                  f"p={v['p_value']:.4g}  p_holm={adjusted[key]:.4g}  "
                  f"{'SIG' if adjusted[key] < 0.05 else 'ns'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"partition": dict(counts), "tests": results,
                   "holm_adjusted": adjusted}, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
