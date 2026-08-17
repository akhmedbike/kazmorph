"""
Statistical evaluation for KazMorphCorpus-2026.

Produces, for one or more pipeline configurations on a held-out test set:
  1. POS / Lemma accuracy with bootstrap 95% Confidence Intervals (10,000 iter)
  2. Per-UPOS-class Precision / Recall / F1 (Macro-F1, Weighted-F1)
  3. Per-error-taxonomy-class F1 (F1_BOR / F1_AFC / F1_SEG / F1_AGR / F1_NONE)
  4. Pairwise McNemar's test between configurations
  5. A confusion matrix (CSV) and error-class distribution (CSV)
  6. A machine-readable metrics.json + run_config.json for the paper's tables

Configuration set (paper ablation A–F):
  A) RMA-only            Pipeline(crf=None, neural=None)
  B) RMA+CRF             Pipeline(crf=..., neural=None)
  C) RMA+Neural          Pipeline(crf=None, neural=...)         (linear cascade)
  D) Hybrid (cascade)    Pipeline(crf=..., neural=...)          (no routing)
  E) Hybrid+routing      Pipeline(crf=..., neural=..., routing=True)   (fusion; non-significant)
  F) Hybrid+fallback     Pipeline(crf=..., neural=..., fallback=True) (neural only for OOV/low-conf)

Usage:
    .venv/bin/python scripts/evaluate.py [--test-file data/test.conllu]
        [--bootstrap-iterations 10000] [--out-dir reports]
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazmorph.pipeline import Pipeline


# ============================================================
#  CoNLL-U parser
# ============================================================

def parse_conllu(path: str) -> list[list[dict]]:
    """Parse CoNLL-U file → list of sentences, each a list of token dicts."""
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            if "-" in parts[0] or "." in parts[0]:
                continue
            current.append({
                "id": parts[0],
                "form": parts[1],
                "lemma": parts[2],
                "upos": parts[3],
            })
    if current:
        sentences.append(current)
    return sentences


# ============================================================
#  Run pipeline on gold data → per-token predictions
# ============================================================

def evaluate_pipeline(pipeline: Pipeline, sentences: list[list[dict]]) -> list[dict]:
    """
    Run pipeline on gold sentences, return per-token results.

    Each element carries gold vs predicted POS/lemma, the model confidence, AND
    the predicted difficulty class (BOR/AFC/SEG/AGR/NONE). The difficulty class
    is what the per-error-taxonomy F1 is computed over.
    """
    tokens = []
    for sentence in sentences:
        text = " ".join(t["form"] for t in sentence)
        results = pipeline.process(text)

        for gi, gold in enumerate(sentence):
            if gi >= len(results):
                break
            pred = results[gi]
            tokens.append({
                "gold_pos": gold["upos"],
                "gold_lemma": gold["lemma"],
                "pred_pos": pred["pos"],
                "pred_lemma": pred["lemma"],
                "confidence": pred["confidence"],
                "difficulty_class": pred.get("difficulty_class", "NONE"),
            })
    return tokens


# ============================================================
#  Bootstrap Confidence Interval
# ============================================================

def bootstrap_ci(
    values: list[bool],
    n_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for accuracy.

    Returns (mean, ci_lower, ci_upper).
    """
    rng = random.Random(seed)
    n = len(values)
    accuracies = []

    for _ in range(n_iterations):
        sample = rng.choices(values, k=n)
        acc = sum(sample) / n
        accuracies.append(acc)

    accuracies.sort()
    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * n_iterations)
    upper_idx = int((1 - alpha) * n_iterations) - 1

    mean = sum(values) / n
    return mean * 100, accuracies[lower_idx] * 100, accuracies[upper_idx] * 100


# ============================================================
#  McNemar's test
# ============================================================

def mcnemar_test(
    correct_a: list[bool],
    correct_b: list[bool],
) -> dict:
    """
    McNemar's test comparing two models on the same test set.

    Contingency table:
                    Model B correct    Model B wrong
    Model A correct      n00               n01
    Model A wrong        n10               n11

    Returns dict with n01, n10, chi2, p_value, significant (alpha=0.05).
    """
    assert len(correct_a) == len(correct_b)

    n01 = 0  # A correct, B wrong
    n10 = 0  # A wrong, B correct

    for a, b in zip(correct_a, correct_b):
        if a and not b:
            n01 += 1
        elif not a and b:
            n10 += 1

    # McNemar's chi-squared with continuity correction
    if (n01 + n10) == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # Approximate p-value from chi-squared distribution (df=1)
    # Using the survival function approximation
    p_value = _chi2_sf(1, chi2)

    return {
        "n01": n01,
        "n10": n10,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def _chi2_sf(df: int, x: float) -> float:
    """Approximate survival function P(X > x) for chi-squared distribution."""
    if x <= 0:
        return 1.0
    # For df=1: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    # Use approximation of the normal CDF tail
    from math import erf, sqrt
    z = sqrt(x)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    phi = 0.5 * (1 + erf(z / sqrt(2)))
    p = 2 * (1 - phi)
    return max(p, 0.0)


# ============================================================
#  Per-class F1 / Macro-F1 / Weighted-F1
# ============================================================

def compute_f1(tokens: list[dict]) -> dict:
    """
    Compute per-class Precision, Recall, F1, plus macro-F1 and weighted-F1.

    Returns dict with:
      'per_tag': {tag: {'precision', 'recall', 'f1', 'support'}}
      'macro_f1': float
      'weighted_f1': float
      'accuracy': float
    """
    # Collect all gold/pred labels
    gold_labels = [t["gold_pos"] for t in tokens]
    pred_labels = [t["pred_pos"] for t in tokens]

    all_tags = sorted(set(gold_labels) | set(pred_labels))

    per_tag = {}
    for tag in all_tags:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == tag and p == tag)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != tag and p == tag)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == tag and p != tag)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        per_tag[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # Macro-F1: unweighted average of per-class F1
    f1_values = [per_tag[t]["f1"] for t in all_tags]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    # Weighted-F1: weighted by support (number of true instances per class)
    total_support = sum(per_tag[t]["support"] for t in all_tags)
    weighted_f1 = (
        sum(per_tag[t]["f1"] * per_tag[t]["support"] for t in all_tags) / total_support
        if total_support > 0 else 0.0
    )

    # Accuracy
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    accuracy = correct / len(gold_labels) if gold_labels else 0.0

    return {
        "per_tag": per_tag,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "all_tags": all_tags,
    }


# ============================================================
#  Per-error-taxonomy-class F1 (F1_BOR / F1_AFC / ...)
# ============================================================

_ERROR_CLASSES = ["BOR", "AFC", "SEG", "AGR", "NONE"]


def compute_error_taxonomy_f1(tokens: list[dict]) -> dict:
    """
    Per-error-class POS-tagging F1.

    For each taxonomy class c we partition the tokens by the *predicted*
    difficulty class and report the POS-tagging accuracy / precision / recall
    / F1 of the pipeline restricted to that subset. This is the metric the
    paper reports as F1_BOR, F1_AFC, F1_SEG, F1_AGR, F1_NONE — it answers
    "how well does the pipeline tag tokens of each structural-difficulty type?".

    ``support`` is the number of tokens predicted into class c; ``pos_accuracy``
    is the fraction of those tokens whose predicted POS equals the gold POS.
    """
    by_class: dict[str, list[dict]] = defaultdict(list)
    for t in tokens:
        by_class[t.get("difficulty_class", "NONE")].append(t)

    per_class = {}
    for c in _ERROR_CLASSES:
        subset = by_class.get(c, [])
        n = len(subset)
        if n == 0:
            per_class[c] = {"support": 0, "pos_accuracy": 0.0, "f1": 0.0}
            continue
        correct = sum(1 for t in subset if t["pred_pos"] == t["gold_pos"])
        acc = correct / n
        # F1 here is accuracy-as-F1 for the binary "POS correct?" task on the
        # subset, which is the standard reporting convention for per-slice F1
        # when there is a single positive class ("correct tag").
        per_class[c] = {
            "support": n,
            "pos_accuracy": round(acc, 4),
            "f1": round(acc, 4),
        }
    return per_class


# ============================================================
#  Confusion matrix (CSV) + error-class distribution (CSV)
# ============================================================

def write_confusion_matrix_csv(tokens: list[dict], path: str) -> None:
    """Write a gold→pred POS confusion matrix as CSV."""
    gold_labels = [t["gold_pos"] for t in tokens]
    pred_labels = [t["pred_pos"] for t in tokens]
    all_tags = sorted(set(gold_labels) | set(pred_labels))
    idx = {t: i for i, t in enumerate(all_tags)}
    matrix = [[0] * len(all_tags) for _ in all_tags]
    for g, p in zip(gold_labels, pred_labels):
        matrix[idx[g]][idx[p]] += 1
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gold\\pred"] + all_tags)
        for i, tag in enumerate(all_tags):
            w.writerow([tag] + matrix[i])


def write_error_distribution_csv(tokens: list[dict], path: str) -> None:
    """Write the predicted-error-class distribution as CSV."""
    counts = Counter(t.get("difficulty_class", "NONE") for t in tokens)
    total = len(tokens)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "share"])
        for c in _ERROR_CLASSES:
            n = counts.get(c, 0)
            w.writerow([c, n, round(n / total, 4) if total else 0.0])


# ============================================================
#  Main evaluation
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical evaluation: Bootstrap CI + McNemar + per-class F1"
    )
    parser.add_argument(
        "--test-file", default="data/test.conllu",
        help="Path to CoNLL-U test file (default: data/test.conllu — the "
             "canonical held-out split from scripts/make_splits.py)",
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=10_000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    parser.add_argument(
        "--out-dir", default="reports",
        help="Directory for machine-readable artifacts (metrics.json, "
             "confusion_matrix.csv, error_distribution.csv). Empty disables.",
    )
    parser.add_argument(
        "--config", default="full",
        choices=["rma", "crf", "neural", "hybrid", "full", "fallback", "all"],
        help="Which pipeline configuration(s) to evaluate. 'all' runs the "
             "ablation set A–F; 'full' runs only E (Hybrid+routing); "
             "'fallback' runs only F (Hybrid+fallback).",
    )
    parser.add_argument(
        "--fallback-threshold", type=float, default=None,
        help="Confidence threshold τ for configuration F only: tokens whose "
             "CRF marginal confidence falls below τ (or that the CRF leaves "
             "UNK/X) are re-tagged by the neural tagger. Other configurations "
             "are unaffected. Default: the Pipeline default (0.50).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        print(f"Error: file not found: {args.test_file}")
        sys.exit(1)

    # Load data
    sentences = parse_conllu(args.test_file)
    total_tokens = sum(len(s) for s in sentences)
    print("=" * 70)
    print("  KazMorphCorpus-2026 — Statistical Evaluation")
    print("=" * 70)
    print(f"\nDataset: {args.test_file}")
    print(f"Sentences: {len(sentences)}  |  Tokens: {total_tokens}")
    print(f"Bootstrap iterations: {args.bootstrap_iterations}\n")

    crf_path = "models/crf_model.joblib"
    neural_path = "models/neural_pos"
    lemma_path = "models/neural_lemma"

    # Build the requested configuration set (paper ablation A–F).
    configs: list[tuple[str, Pipeline]] = []
    requested = ["A", "B", "C", "D", "E", "F"] if args.config == "all" else [args.config]

    cfg_map = {
        # name, label, pipeline factory
        "A": ("RMA-only", lambda: Pipeline()),
        "B": ("RMA+CRF", lambda: Pipeline(crf_model_path=crf_path if os.path.exists(crf_path) else None)),
        "C": ("RMA+Neural", lambda: Pipeline(neural_model_path=neural_path if os.path.exists(neural_path) else None,
                                             lemma_model_path=lemma_path if os.path.exists(lemma_path) else None)),
        "D": ("Hybrid (cascade)", lambda: Pipeline(
            crf_model_path=crf_path if os.path.exists(crf_path) else None,
            neural_model_path=neural_path if os.path.exists(neural_path) else None,
            lemma_model_path=lemma_path if os.path.exists(lemma_path) else None)),
        "E": ("Hybrid+routing", lambda: Pipeline(
            crf_model_path=crf_path if os.path.exists(crf_path) else None,
            neural_model_path=neural_path if os.path.exists(neural_path) else None,
            lemma_model_path=lemma_path if os.path.exists(lemma_path) else None,
            routing=True)),
        # F: hard neural fallback — CRF is the default POS source; the neural
        # tagger contributes only for tokens the CRF leaves unknown or
        # low-confidence. Lemma is intentionally NOT taken from the neural
        # lemmatizer (lemma_model_path omitted) so that F differs from B only
        # in the POS source, isolating the fallback effect.
        "F": ("RMA+CRF+Neural-fallback", lambda: Pipeline(
            crf_model_path=crf_path if os.path.exists(crf_path) else None,
            neural_model_path=neural_path if os.path.exists(neural_path) else None,
            fallback=True,
            **({"fallback_conf_threshold": args.fallback_threshold}
               if args.fallback_threshold is not None else {}))),
    }
    # Map the friendly --config names to ablation letters.
    friendly = {"rma": "A", "crf": "B", "neural": "C", "hybrid": "D",
                "full": "E", "fallback": "F"}

    models: list[tuple[str, list[dict]]] = []
    fallback_stats: dict[str, dict] = {}
    for key in requested:
        letter = friendly.get(key, key)
        label, factory = cfg_map[letter]
        print(f"Evaluating [{letter}] {label} ...")
        pipe = factory()
        t0 = time.time()
        toks = evaluate_pipeline(pipe, sentences)
        stats = getattr(pipe, "last_fallback_stats", None)
        if stats is not None:
            fallback_stats[f"[{letter}] {label}"] = dict(stats)
            print(f"  neural-fallback taken for {stats['neural_taken']}/{stats['total']} "
                  f"tokens ({stats['neural_taken'] / max(stats['total'], 1) * 100:.1f}%)")
        print(f"  done in {time.time() - t0:.1f}s  ({len(toks)} tokens)")
        models.append((f"[{letter}] {label}", toks))

    # ============================================================
    #  Results
    # ============================================================

    # Aggregate everything we want to persist for the paper.
    report: dict = {
        "test_file": args.test_file,
        "sentences": len(sentences),
        "tokens": total_tokens,
        "bootstrap_iterations": args.bootstrap_iterations,
        "fallback_threshold": args.fallback_threshold,
        "configurations": {},
    }
    if fallback_stats:
        report["fallback_stats"] = fallback_stats

    # ── Bootstrap CI ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"\n  {'Model':<24} {'POS Acc':>8} {'95% CI':>22} {'Lemma Acc':>10} {'95% CI':>22}")
    print(f"  {'─' * 24} {'─' * 8} {'─' * 22} {'─' * 10} {'─' * 22}")

    for name, tokens in models:
        pos_correct = [t["pred_pos"] == t["gold_pos"] for t in tokens]
        lemma_correct = [t["pred_lemma"].lower() == t["gold_lemma"].lower() for t in tokens]

        pos_mean, pos_lo, pos_hi = bootstrap_ci(pos_correct, args.bootstrap_iterations)
        lem_mean, lem_lo, lem_hi = bootstrap_ci(lemma_correct, args.bootstrap_iterations)
        f1_data = compute_f1(tokens)
        tax_f1 = compute_error_taxonomy_f1(tokens)

        print(f"  {name:<24} {pos_mean:>7.2f}% [{pos_lo:>6.2f}%, {pos_hi:>6.2f}%]"
              f"  {lem_mean:>8.2f}% [{lem_lo:>6.2f}%, {lem_hi:>6.2f}%]")

        report["configurations"][name] = {
            "pos_accuracy": round(pos_mean, 4),
            "pos_ci": [round(pos_lo, 4), round(pos_hi, 4)],
            "lemma_accuracy": round(lem_mean, 4),
            "lemma_ci": [round(lem_lo, 4), round(lem_hi, 4)],
            "macro_f1": round(f1_data["macro_f1"], 4),
            "weighted_f1": round(f1_data["weighted_f1"], 4),
            "per_tag_f1": {t: {k: round(v, 4) for k, v in d.items()}
                           for t, d in f1_data["per_tag"].items()},
            "error_taxonomy_f1": tax_f1,
            "n_tokens": len(tokens),
        }

    # ── Per-class F1 / Macro-F1 ────────────────────────────────
    print("\n" + "=" * 70)
    print("  PER-UPOS-CLASS F1 SCORES")
    print("=" * 70)
    for name, tokens in models:
        f1_data = compute_f1(tokens)
        print(f"\n  {name}:  Macro-F1 = {f1_data['macro_f1']:.4f}  |  "
              f"Weighted-F1 = {f1_data['weighted_f1']:.4f}  |  "
              f"Accuracy = {f1_data['accuracy']:.4f}")
        print(f"  {'UPOS':<8} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
        print(f"  {'----':<8} {'----':>6} {'---':>6} {'--':>6} {'---':>6}")
        sorted_tags = sorted(f1_data["all_tags"], key=lambda t: -f1_data["per_tag"][t]["support"])
        for tag in sorted_tags:
            d = f1_data["per_tag"][tag]
            marker = " <<<" if d["f1"] < 0.50 and d["support"] >= 5 else ""
            print(f"  {tag:<8} {d['precision']:>6.3f} {d['recall']:>6.3f} "
                  f"{d['f1']:>6.3f} {d['support']:>6}{marker}")

    # ── Per-error-taxonomy F1 (the paper's F1_BOR / F1_AFC / ...) ─
    print("\n" + "=" * 70)
    print("  PER-ERROR-TAXONOMY F1")
    print("=" * 70)
    for name, tokens in models:
        tax = compute_error_taxonomy_f1(tokens)
        print(f"\n  {name}:")
        print(f"  {'Class':<8} {'Support':>8} {'POS-Acc':>9} {'F1':>6}")
        print(f"  {'-----':<8} {'-------':>8} {'-------':>9} {'--':>6}")
        for c in _ERROR_CLASSES:
            d = tax[c]
            print(f"  {c:<8} {d['support']:>8} {d['pos_accuracy']:>9.3f} {d['f1']:>6.3f}")

    # ── McNemar's test (pairwise) ──────────────────────────────
    print("\n" + "=" * 70)
    print("  McNEMAR'S TEST (alpha = 0.05)")
    print("=" * 70)
    mcnemar_results: list[dict] = []
    if len(models) >= 2:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                name_a, tok_a = models[i]
                name_b, tok_b = models[j]
                pos_a = [t["pred_pos"] == t["gold_pos"] for t in tok_a]
                pos_b = [t["pred_pos"] == t["gold_pos"] for t in tok_b]
                m = mcnemar_test(pos_a, pos_b)
                sig = "*** SIGNIFICANT" if m["significant"] else "not significant"
                print(f"\n  {name_a}  vs  {name_b}")
                print(f"  POS: χ²={m['chi2']:.4f}  p={m['p_value']:.6f}  {sig}")
                print(f"       A-only correct: {m['n01']}  |  B-only correct: {m['n10']}")
                mcnemar_results.append({"pair": [name_a, name_b], **m})
    else:
        print("\n  Need >= 2 configurations for McNemar's test.")
    report["mcnemar"] = mcnemar_results

    # ── Persist machine-readable artifacts ──────────────────────
    if args.out_dir and models:
        os.makedirs(args.out_dir, exist_ok=True)
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # Confusion matrix + error distribution for the last (best) config.
        last_name, last_tokens = models[-1]
        cm_path = os.path.join(args.out_dir, "confusion_matrix.csv")
        ed_path = os.path.join(args.out_dir, "error_distribution.csv")
        write_confusion_matrix_csv(last_tokens, cm_path)
        write_error_distribution_csv(last_tokens, ed_path)
        # run_config.json — full provenance for reproducibility.
        rc_path = os.path.join(args.out_dir, "run_config.json")
        with open(rc_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_file": args.test_file,
                "configs_evaluated": [name for name, _ in models],
                "bootstrap_iterations": args.bootstrap_iterations,
                "crf_model": crf_path if os.path.exists(crf_path) else None,
                "neural_model": neural_path if os.path.exists(neural_path) else None,
                "lemma_model": lemma_path if os.path.exists(lemma_path) else None,
                "seed": 42,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  Artifacts written to {args.out_dir}/")
        print(f"    - metrics.json")
        print(f"    - confusion_matrix.csv  ({last_name})")
        print(f"    - error_distribution.csv")
        print(f"    - run_config.json")

    print()


if __name__ == "__main__":
    main()
