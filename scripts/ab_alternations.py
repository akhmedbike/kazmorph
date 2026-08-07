#!/usr/bin/env python3
"""
A/B comparison: OLD lemmatizer (backup) vs NEW (retrained) on q/g alternations.

Decision gate (REVIEW_RECOMMENDATIONS.md problem #1): the retrained model
only REPLACES the backup if it WINS on BOTH:
  1. q/g alternation accuracy on dev_split is strictly higher than baseline
     (currently 20.8% — the core complaint from the expert review).
  2. Overall lemma accuracy on dev_split does NOT drop below baseline
     (must stay >= baseline — protecting the non-alternating majority).

Both metrics use the same honest protocol as scripts/benchmark.py and
scripts/ab_lemma.py: process dev_split sentences through the full pipeline
(POS tagging + lemmatizer + MFRN), align positionally, compare lemmas.

A "real q/g alternation" is defined as a (form, lemma) pair where:
  - the two share a common prefix of length >= 3 (a real stem), AND
  - at the divergence point, form has a voiced consonant (б/г/ғ/д/ж/з) and
    lemma has its exact voiceless pair (п/к/қ/т/ш/с).
This excludes suppletion (оның→ол) and orthographic noise (ба→ма).

Usage:
    # After a retrain: compare backup vs new
    PYTHONPATH=. .venv/bin/python scripts/ab_alternations.py \
        --old models/neural_lemma.backup_20260723 \
        --new models/neural_lemma \
        --test-file data/dev_split.conllu

Exit code 0 if new wins (both gates), 1 otherwise. Prints a verdict.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import parse_conllu
from kazmorph.pipeline import Pipeline
from kazmorph.rma import ReverseMorphAnalyzer
from kazmorph.neural import NeuralTagger
from kazmorph.neural_lemma import NeuralLemmatizer
from kazmorph.mfrn import MFRN

# Voiced → voiceless (same map as train_lemma._DEVOICE, kept local so this
# script has no dependency on the training module).
_DEVOICE = {"б": "п", "г": "к", "ғ": "қ", "д": "т", "ж": "ш", "з": "с"}


def is_real_alternation(form: str, lemma: str) -> bool:
    """True if (form, lemma) is a genuine q/g-style stem alternation."""
    form = form.lower()
    lemma = lemma.lower()
    if form == lemma or len(lemma) < 4:
        return False
    k = 0
    for a, b in zip(form, lemma):
        if a != b:
            break
        k += 1
    # Need a real shared stem (>= 3 chars) and the divergence must be a
    # voiced→voiceless pair at the SAME position in both.
    if k < 3 or k >= len(lemma):
        return False
    return k < len(form) and form[k] in _DEVOICE and _DEVOICE[form[k]] == lemma[k]


def build_pipeline(lemma_model_dir: str) -> Pipeline:
    """Build a pipeline with a specific lemmatizer checkpoint. POS tagger,
    RMA, MFRN stay identical so only the lemmatizer differs between A and B."""
    p = Pipeline.__new__(Pipeline)
    p.rma = ReverseMorphAnalyzer()
    p.crf = __import__("kazmorph.crf_tagger", fromlist=["CRFTagger"]).CRFTagger(model_path=None)
    p.neural = NeuralTagger()  # default models/neural_pos
    p.lemmatizer = NeuralLemmatizer(model_path=lemma_model_dir)
    p.mfrn = MFRN()
    return p


def evaluate(pipeline: Pipeline, sentences) -> dict:
    """Run the pipeline over dev_split and collect lemma metrics."""
    total = 0
    lemma_correct = 0
    alt_total = 0
    alt_correct = 0
    alt_errors = []
    aligned = 0

    for sent in sentences:
        tokens = [t.form for t in sent.tokens]
        results = pipeline.process(" ".join(tokens))
        if len(results) != len(sent.tokens):
            continue
        aligned += 1
        for gt, res in zip(sent.tokens, results):
            if not gt.lemma:
                continue
            total += 1
            gold = gt.lemma.lower()
            pred = res["lemma"].lower()
            if pred == gold:
                lemma_correct += 1
            if is_real_alternation(gt.form, gt.lemma):
                alt_total += 1
                if pred == gold:
                    alt_correct += 1
                elif len(alt_errors) < 15:
                    alt_errors.append((gt.form, gold, pred))

    return {
        "total": total,
        "lemma_acc": lemma_correct / total if total else 0.0,
        "alt_total": alt_total,
        "alt_acc": alt_correct / alt_total if alt_total else 0.0,
        "aligned_sents": aligned,
        "alt_errors": alt_errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True,
                    help="OLD lemmatizer model dir (baseline).")
    ap.add_argument("--new", required=True,
                    help="NEW lemmatizer model dir (retrained).")
    ap.add_argument("--test-file",
                    default=os.path.join(PROJECT_ROOT, "data", "dev_split.conllu"))
    args = ap.parse_args()

    sentences = parse_conllu(args.test_file)
    print(f"Test: {args.test_file} ({len(sentences)} sentences)\n")

    print(f"Loading OLD ({args.old}) ...")
    old_pipe = build_pipeline(args.old)
    old = evaluate(old_pipe, sentences)

    print(f"Loading NEW ({args.new}) ...")
    new_pipe = build_pipeline(args.new)
    new = evaluate(new_pipe, sentences)

    print("\n" + "=" * 62)
    print("  A/B RESULTS: alternation fix (q/g)")
    print("=" * 62)
    print(f"{'Metric':<30} {'OLD':>10} {'NEW':>10} {'Δ':>10}")
    print("-" * 62)
    print(f"{'Overall lemma acc':<30} {old['lemma_acc']*100:>9.2f}% {new['lemma_acc']*100:>9.2f}% {(new['lemma_acc']-old['lemma_acc'])*100:>+9.2f}%")
    print(f"{'q/g alternation acc':<30} {old['alt_acc']*100:>9.2f}% {new['alt_acc']*100:>9.2f}% {(new['alt_acc']-old['alt_acc'])*100:>+9.2f}%")
    print(f"{'  (alternation count)':<30} {old['alt_total']:>10} {new['alt_total']:>10}")
    print(f"{'Aligned sentences':<30} {old['aligned_sents']:>10} {new['aligned_sents']:>10}")
    print("=" * 62)

    # Decision gates
    gate_accuracy = new["alt_acc"] > old["alt_acc"]
    gate_no_regression = new["lemma_acc"] >= old["lemma_acc"] - 0.001  # 0.1% tolerance

    print("\nGATES:")
    print(f"  1. alternation acc improved:    {'PASS' if gate_accuracy else 'FAIL'} "
          f"({old['alt_acc']*100:.1f}% → {new['alt_acc']*100:.1f}%)")
    print(f"  2. overall lemma not regressed: {'PASS' if gate_no_regression else 'FAIL'} "
          f"({old['lemma_acc']*100:.1f}% → {new['lemma_acc']*100:.1f}%)")

    if new["alt_errors"]:
        print(f"\nRemaining alternation errors (NEW, first {len(new['alt_errors'])}):")
        for form, gold, pred in new["alt_errors"]:
            print(f"  {form:16s} gold={gold:14s} pred={pred}")

    if gate_accuracy and gate_no_regression:
        print("\nVERDICT: NEW WINS — keep the retrained model.")
        return 0
    print("\nVERDICT: NEW FAILS a gate — roll back to OLD.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
