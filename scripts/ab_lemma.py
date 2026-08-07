#!/usr/bin/env python3
"""
A/B comparison: classifier lemmatizer (kazmorph/neural_lemma.py) vs the
seq2seq lemmatizer (kazmorph/neural_lemma_seq2seq.py).

Decision gate (per the plan in this session): the seq2seq model only
replaces the classifier in pipeline.py if it WINS on BOTH:
  1. lemma accuracy > CLASSIFIER_ACC (current bar: > 0.89, i.e. beats 89%)
  2. latency ≤ 3x the classifier's latency on the same input

Both metrics are measured on the held-out dev_split with form-aligned
evaluation (the same honest protocol that uncovered the tokenization bug —
see WORKLOG §7). POS tagging and all other pipeline stages stay identical;
only the lemmatizer class differs.

Usage:
    python scripts/ab_lemma.py
    python scripts/ab_lemma.py --test-file data/dev_split.conllu

Output: a printed verdict ("seq2seq wins, switch" / "classifier stays")
plus a per-tag and error-type breakdown. No files written, no pipeline
changes — the verdict is informational; the actual switch is a separate
one-line edit in kazmorph/pipeline.py if warranted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.pipeline import Pipeline
from kazmorph.neural_lemma import NeuralLemmatizer as ClassifierLemmatizer
from kazmorph.corpus import parse_conllu


# Decision thresholds (from the plan). seq2seq must beat BOTH to switch.
ACCURACY_BAR = 0.89      # current classifier is ~0.876; require strict improvement above 0.89
LATENCY_FACTOR = 3.0     # seq2seq may be at most 3x slower than classifier


def _build_pipeline(lemmatizer_class, model_dir):
    """Build a Pipeline with a specific lemmatizer class swapped in."""
    crf_path = os.path.join(PROJECT_ROOT, "models", "crf_model.joblib")
    neural_path = os.path.join(PROJECT_ROOT, "models", "neural_pos")
    crf = crf_path if os.path.exists(crf_path) else None
    neural = neural_path if os.path.exists(neural_path) else None

    p = Pipeline(crf_model_path=crf, neural_model_path=neural, lemma_model_path=None)
    # Swap in the requested lemmatizer. lemma_model_path=None above leaves the
    # default (which may be a classifier); we override explicitly here.
    p.lemmatizer = lemmatizer_class(model_path=model_dir)
    return p


def _evaluate(pipeline, sentences):
    """Measure lemma accuracy per-sentence (form-aligned eval) and latency on
    a realistic long text separately.

    Accuracy MUST be measured per-sentence so we can form-align predictions to
    gold tokens (the pipeline's own tokenizer can split MWEs differently than
    the gold CoNLL-U). But summing per-sentence wall times overstates seq2seq
    latency, because each short sentence triggers its own model.generate() call
    with tiny batches — the worst case for autoregressive decoding.

    So latency is reported separately from a single long-text run (all dev
    sentences joined, mirroring how a user pastes a long document). That is the
    number that matters for the production latency budget.
    """
    total = correct = 0
    by_tag = defaultdict(lambda: [0, 0])
    errors: list[tuple[str, str, str, str]] = []  # (form, pred, gold, upos)

    # Accuracy pass (per-sentence, for form alignment).
    for s in sentences:
        text = " ".join(s.forms)
        result = pipeline.process(text)
        gi = 0
        for pred in result:
            if gi >= len(s.tokens):
                break
            gold_tok = s.tokens[gi]
            # Form-aligned evaluation: skip mismatches rather than miscount.
            if pred["token"].lower() != gold_tok.form.lower():
                continue
            gi += 1
            if gold_tok.upos == "PUNCT":
                continue
            total += 1
            gold_lemma = gold_tok.lemma.lower()
            pred_lemma = pred["lemma"].lower()
            by_tag[gold_tok.upos][1] += 1
            if pred_lemma == gold_lemma:
                correct += 1
                by_tag[gold_tok.upos][0] += 1
            else:
                errors.append((gold_tok.form, pred_lemma, gold_lemma, gold_tok.upos))

    acc = correct / total if total else 0.0

    # Latency pass: a single long-text run. Join all sentences the way a user
    # pasting a document would, so batching inside _predict_chunk actually fires
    # (per-sentence eval keeps batches tiny and overstates seq2seq cost ~7x).
    long_text = " ".join(" ".join(s.forms) for s in sentences)
    start = time.perf_counter()
    pipeline.process(long_text)
    elapsed = time.perf_counter() - start

    return acc, total, elapsed, dict(by_tag), errors


def _error_type(pred: str, gold: str) -> str:
    """Classify a lemma error for the breakdown."""
    if len(pred) > len(gold):
        return "over-truncate (pred shorter)"
    if len(pred) < len(gold):
        return "under-truncate (pred longer)"
    if len(pred) == len(gold):
        return "alternation (same length, different chars)"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-file",
                        default=os.path.join(PROJECT_ROOT, "data", "dev_split.conllu"))
    parser.add_argument("--classifier-dir",
                        default=os.path.join(PROJECT_ROOT, "models", "neural_lemma"))
    parser.add_argument("--seq2seq-dir",
                        default=os.path.join(PROJECT_ROOT, "models", "neural_lemma_seq2seq"))
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        sys.exit(f"error: test file not found: {args.test_file}")
    if not os.path.exists(os.path.join(args.classifier_dir, "config.json")):
        sys.exit(f"error: classifier model not found: {args.classifier_dir}")
    if not os.path.exists(os.path.join(args.seq2seq_dir, "config.json")):
        sys.exit(f"error: seq2seq model not found: {args.seq2seq_dir}\n"
                 f"        train it first: scripts/train_lemma_seq2seq.py")

    sentences = parse_conllu(args.test_file)
    n_tokens = sum(len(s.tokens) for s in sentences)
    print(f"Dataset: {args.test_file}  ({len(sentences)} sentences, ~{n_tokens} tokens)\n")

    # --- Classifier (baseline) ---
    print("Evaluating CLASSIFIER (kazmorph/neural_lemma.py) ...")
    from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
    p_clf = _build_pipeline(ClassifierLemmatizer, args.classifier_dir)
    p_clf.process("сәлем.")  # warmup
    clf_acc, clf_n, clf_time, clf_by_tag, clf_errors = _evaluate(p_clf, sentences)
    print(f"  lemma_acc = {clf_acc:.4f}  ({clf_n} tokens, {clf_time:.1f}s)\n")

    # --- Seq2seq (challenger) ---
    print("Evaluating SEQ2SEQ (kazmorph/neural_lemma_seq2seq.py) ...")
    p_seq = _build_pipeline(NeuralLemmatizerSeq2Seq, args.seq2seq_dir)
    p_seq.process("сәлем.")  # warmup
    seq_acc, seq_n, seq_time, seq_by_tag, seq_errors = _evaluate(p_seq, sentences)
    print(f"  lemma_acc = {seq_acc:.4f}  ({seq_n} tokens, {seq_time:.1f}s)\n")

    # --- Comparison ---
    print("=" * 66)
    print("  A/B COMPARISON")
    print("=" * 66)
    print(f"  {'metric':<24} {'classifier':>12} {'seq2seq':>12} {'Δ':>10}")
    print(f"  {'-'*60}")
    print(f"  {'lemma accuracy':<24} {clf_acc:>11.2%} {seq_acc:>11.2%} {seq_acc-clf_acc:>+9.2%}")
    latency_ratio = seq_time / clf_time if clf_time > 0 else float("inf")
    print(f"  {'wall time':<24} {clf_time:>11.1f}s {seq_time:>11.1f}s {latency_ratio:>+8.2f}x")

    print(f"\n  {'error count':<24} {len(clf_errors):>12} {len(seq_errors):>12}"
          f" {len(seq_errors)-len(clf_errors):>+10}")

    # Error-type breakdown for both
    print("\n  Error-type breakdown:")
    clf_types = Counter(_error_type(p, g) for _, p, g, _ in clf_errors)
    seq_types = Counter(_error_type(p, g) for _, p, g, _ in seq_errors)
    all_types = sorted(set(clf_types) | set(seq_types))
    print(f"  {'type':<36} {'clf':>6} {'seq2seq':>8}")
    for t in all_types:
        print(f"  {t:<36} {clf_types.get(t,0):>6} {seq_types.get(t,0):>8}")

    # Per-tag accuracy
    print("\n  Per-tag lemma accuracy:")
    all_tags = sorted(set(clf_by_tag) | set(seq_by_tag))
    print(f"  {'tag':<6} {'clf':>8} {'seq2seq':>8} {'Δ':>8}")
    for tag in all_tags:
        cc, cn = clf_by_tag.get(tag, (0, 1))
        sc, sn = seq_by_tag.get(tag, (0, 1))
        cacc = cc / cn if cn else 0
        sacc = sc / sn if sn else 0
        print(f"  {tag:<6} {cacc:>7.1%} {sacc:>7.1%} {sacc-cacc:>+7.1%}")

    # --- Verdict ---
    print("\n" + "=" * 66)
    acc_wins = seq_acc > ACCURACY_BAR and seq_acc > clf_acc
    speed_ok = latency_ratio <= LATENCY_FACTOR
    print(f"  Decision gates:")
    print(f"    accuracy: seq2seq > {ACCURACY_BAR:.2f} AND > classifier  → "
          f"{'PASS' if acc_wins else 'FAIL'} ({seq_acc:.2%})")
    print(f"    latency:  seq2seq ≤ {LATENCY_FACTOR:.0f}x classifier     → "
          f"{'PASS' if speed_ok else 'FAIL'} ({latency_ratio:.2f}x)")

    if acc_wins and speed_ok:
        print("\n  VERDICT: ✅ seq2seq WINS — switch pipeline.py to NeuralLemmatizerSeq2Seq")
        print("  To switch, edit kazmorph/pipeline.py:")
        print("    from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq")
        print("    self.lemmatizer = NeuralLemmatizerSeq2Seq(model_path=lemma_model_path)")
    else:
        reasons = []
        if not acc_wins:
            reasons.append(f"accuracy {seq_acc:.2%} did not beat bar {ACCURACY_BAR:.2%}/classifier {clf_acc:.2%}")
        if not speed_ok:
            reasons.append(f"latency {latency_ratio:.2f}x exceeded {LATENCY_FACTOR:.0f}x budget")
        print("\n  VERDICT: ❌ CLASSIFIER STAYS — " + "; ".join(reasons))
    print("=" * 66)


if __name__ == "__main__":
    main()
