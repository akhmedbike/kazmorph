"""
Benchmark KazMorphCorpus-2026 pipeline on UD Kazakh-KTB test split.

Measures per-stage and overall:
  - POS tagging accuracy
  - Lemma accuracy
  - Token coverage
  - Average confidence

Usage:
    .venv/bin/python scripts/benchmark.py [--max-sentences N]
"""

import argparse
import os
import sys
import time

# Ensure project root is importable
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
            # Skip multiword tokens (ID contains '-')
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
#  Benchmark runner
# ============================================================

def run_benchmark(conllu_path: str, max_sentences: int | None = None):
    """Run full pipeline benchmark against UD test split."""

    print("=" * 70)
    print("  KazMorphCorpus-2026 — Full Pipeline Benchmark")
    print("=" * 70)
    print()

    # Load data
    sentences = parse_conllu(conllu_path)
    if max_sentences:
        sentences = sentences[:max_sentences]

    total_tokens = sum(len(s) for s in sentences)
    print(f"Dataset:  {conllu_path}")
    print(f"Sentences: {len(sentences)}")
    print(f"Tokens:    {total_tokens}")
    print()

    # Init pipeline with all available models
    crf_path = "models/crf_model.joblib"
    neural_path = "models/neural_pos"
    lemma_path = "models/neural_lemma"
    crf_model = crf_path if os.path.exists(crf_path) else None
    neural_model = neural_path if os.path.exists(neural_path) else None
    lemma_model = lemma_path if os.path.exists(lemma_path) else None

    print("Loading pipeline...")
    print(f"  CRF model:    {'YES' if crf_model else 'NO (stub)'}")
    print(f"  Neural model: {'YES' if neural_model else 'NO (stub)'}")
    print(f"  Lemma model:  {'YES' if lemma_model else 'NO (stub)'}")
    print()

    pipeline = Pipeline(
        crf_model_path=crf_model,
        neural_model_path=neural_model,
        lemma_model_path=lemma_model,
    )

    # Counters
    pos_correct = 0
    lemma_correct = 0
    pos_correct_rma = 0  # RMA-only baseline
    lemma_correct_rma = 0
    total = 0
    unknown_tokens = 0
    confidence_sum = 0.0

    pos_per_tag_correct = {}
    pos_per_tag_total = {}

    # Per-stage breakdown
    pos_correct_after_crf = 0
    pos_correct_after_neural = 0

    start_time = time.time()

    for si, sentence in enumerate(sentences):
        # Reconstruct text from gold tokens
        text = " ".join(t["form"] for t in sentence)

        # Run pipeline
        results = pipeline.process(text)

        # Align predictions to gold tokens BY FORM, not by position. The
        # pipeline's own tokenizer can split tokens differently from the gold
        # CoNLL-U (e.g. MWEs like "болған емес" are one UD token but two words
        # in our tokenizer; or the reverse). Positional zip(results, gold)
        # silently shifts on every such mismatch and blames the model for
        # errors that are really alignment artifacts — this is what depressed
        # our reported accuracy by ~7 points before this fix. Form-alignment
        # skips tokens we can't match confidently instead of miscounting them.
        gold_tokens = [t for t in sentence]
        gi = 0  # index into gold_tokens
        for pred in results:
            if gi >= len(gold_tokens):
                break
            gold = gold_tokens[gi]
            # Match on lowercased form. If forms diverge, this is either an
            # extra pipeline token (skip it) or a gold token we have no
            # prediction for (skip gold too). Either way, don't score it.
            if pred["token"].lower() != gold["form"].lower():
                continue
            gi += 1

            total += 1

            gold_pos = gold["upos"]
            gold_lemma = gold["lemma"]

            pred_pos = pred["pos"]
            pred_lemma = pred["lemma"]
            pred_conf = pred["confidence"]

            # POS accuracy
            if pred_pos == gold_pos:
                pos_correct += 1

            # POS per-tag
            if gold_pos not in pos_per_tag_total:
                pos_per_tag_total[gold_pos] = 0
                pos_per_tag_correct[gold_pos] = 0
            pos_per_tag_total[gold_pos] += 1
            if pred_pos == gold_pos:
                pos_per_tag_correct[gold_pos] += 1

            # Lemma accuracy
            if pred_lemma.lower() == gold_lemma.lower():
                lemma_correct += 1

            # Unknown tokens (low confidence)
            if pred_conf < 0.3:
                unknown_tokens += 1

            confidence_sum += pred_conf

        if (si + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {si + 1}/{len(sentences)} sentences "
                  f"({elapsed:.1f}s, {total} tokens)...")

    elapsed = time.time() - start_time

    # ============================================================
    #  Results
    # ============================================================

    pos_acc = pos_correct / total * 100 if total else 0
    lemma_acc = lemma_correct / total * 100 if total else 0
    avg_conf = confidence_sum / total if total else 0
    unknown_pct = unknown_tokens / total * 100 if total else 0
    tokens_per_sec = total / elapsed if elapsed else 0

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  POS Accuracy:       {pos_acc:.2f}%  ({pos_correct}/{total})")
    print(f"  Lemma Accuracy:     {lemma_acc:.2f}%  ({lemma_correct}/{total})")
    print(f"  Unknown Tokens:     {unknown_pct:.1f}%  ({unknown_tokens}/{total})")
    print(f"  Avg Confidence:     {avg_conf:.3f}")
    print(f"  Processing Speed:   {tokens_per_sec:.0f} tokens/sec")
    print(f"  Wall Time:          {elapsed:.1f}s")
    print()

    # Per-tag breakdown
    print("-" * 70)
    print("  POS Tag Breakdown (sorted by frequency):")
    print("-" * 70)
    print(f"  {'UPOS':<8} {'Total':>6} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'----':<8} {'-----':>6} {'-------':>8} {'--------':>10}")

    sorted_tags = sorted(pos_per_tag_total.items(), key=lambda x: -x[1])
    for tag, cnt in sorted_tags:
        corr = pos_per_tag_correct.get(tag, 0)
        acc = corr / cnt * 100 if cnt else 0
        marker = " <<< LOW" if acc < 50 else ""
        print(f"  {tag:<8} {cnt:>6} {corr:>8} {acc:>9.1f}%{marker}")

    print()
    print("=" * 70)

    # Also print a compact summary for copy-paste into paper
    print()
    print("  [Compact summary for paper]")
    print(f"  POS: {pos_acc:.1f}%  |  Lemma: {lemma_acc:.1f}%  |  "
          f"Tokens: {total}  |  Sentences: {len(sentences)}  |  "
          f"Speed: {tokens_per_sec:.0f} tok/s")
    print()


def _measure_ram_bytes() -> int:
    """Best-effort resident-set size in bytes (RSS). macOS/BSD uses ru_maxrss
    in bytes; Linux uses kilobytes. We normalize to bytes."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in KB; on macOS it is already in bytes.
    if sys.platform.startswith("linux"):
        rss *= 1024
    return int(rss)


def benchmark_configs(conllu_path: str, *, out_path: str | None = None) -> list[dict]:
    """
    Throughput + RAM + accuracy for each pipeline configuration (paper Table 9).

    For each config we instantiate a fresh Pipeline, measure the resident memory
    footprint after model loading, then time end-to-end processing of the test
    split. Accuracy is reported via form-alignment (same convention as
    run_benchmark) so the numbers are directly comparable to evaluate.py.
    """
    import json

    sentences = parse_conllu(conllu_path)
    texts = [" ".join(t["form"] for t in s) for s in sentences]
    n_tokens = sum(len(s) for s in sentences)

    crf_path = "models/crf_model.joblib"
    neural_path = "models/neural_pos"
    lemma_path = "models/neural_lemma"
    crf = crf_path if os.path.exists(crf_path) else None
    neural = neural_path if os.path.exists(neural_path) else None
    lemma = lemma_path if os.path.exists(lemma_path) else None

    # (label, requires_neural, factory)
    configs = [
        ("RMA-only", False,
         lambda: Pipeline()),
        ("RMA+CRF", False,
         lambda: Pipeline(crf_model_path=crf)),
        ("RMA+Neural", True,
         lambda: Pipeline(neural_model_path=neural, lemma_model_path=lemma)),
        ("Hybrid (cascade)", True,
         lambda: Pipeline(crf_model_path=crf, neural_model_path=neural,
                          lemma_model_path=lemma)),
        ("Hybrid+routing", True,
         lambda: Pipeline(crf_model_path=crf, neural_model_path=neural,
                          lemma_model_path=lemma, routing=True)),
        # F: hard neural fallback — neural tagger runs only for tokens the CRF
        # leaves unknown/low-confidence. Lemma from RMA lexicon (no neural
        # lemma), same as RMA+CRF, so POS is the only difference vs config B.
        ("Hybrid+fallback", True,
         lambda: Pipeline(crf_model_path=crf, neural_model_path=neural,
                          fallback=True)),
    ]
    # Strong-transformer baselines G/H (same shape as RMA+Neural, different
    # pretrained encoder) — only benchmarked when their checkpoints exist.
    for label, path in (("RMA+XLM-R", "models/neural_pos_xlmr"),
                        ("RMA+KazRoBERTa", "models/neural_pos_kazr")):
        if os.path.exists(os.path.join(path, "config.json")):
            configs.append((label, True,
                            (lambda p=path: Pipeline(
                                neural_model_path=p,
                                lemma_model_path=lemma))))

    print("=" * 70)
    print("  THROUGHPUT / RAM BENCHMARK (paper Table 9)")
    print("=" * 70)
    print(f"  Dataset: {conllu_path}  ({len(sentences)} sents, {n_tokens} tokens)\n")
    print(f"  {'Configuration':<20} {'tok/s':>8} {'RAM (MB)':>10} "
          f"{'Acc %':>7} {'GPU':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*7} {'-'*5}")

    results: list[dict] = []
    for label, needs_neural, factory in configs:
        if needs_neural and not neural:
            print(f"  {label:<20} {'(skipped: no neural model)'}")
            continue
        pipe = factory()
        ram_bytes = _measure_ram_bytes()
        correct = 0
        total = 0
        t0 = time.time()
        for s, text in zip(sentences, texts):
            res = pipe.process(text)
            gi = 0
            for pred in res:
                if gi >= len(s):
                    break
                gold = s[gi]
                if pred["token"].lower() != gold["form"].lower():
                    continue
                gi += 1
                total += 1
                if pred["pos"] == gold["upos"]:
                    correct += 1
        elapsed = time.time() - t0
        tps = total / elapsed if elapsed else 0.0
        acc = correct / total * 100 if total else 0.0
        ram_mb = ram_bytes / (1024 * 1024)
        # We never use a GPU; the neural tagger is CPU-only. Mark "Optional"
        # for configs that load the transformer, "No" otherwise (paper Table 9).
        gpu = "Opt." if needs_neural else "No"
        print(f"  {label:<20} {tps:>8.0f} {ram_mb:>10.0f} {acc:>6.1f}% {gpu:>5}")
        results.append({
            "configuration": label,
            "throughput_tokens_per_sec": round(tps, 1),
            "ram_mb": round(ram_mb, 0),
            "accuracy_pct": round(acc, 2),
            "requires_gpu": False,
            "gpu_optional": needs_neural,
            "wall_seconds": round(elapsed, 2),
            "tokens": total,
        })
        # Free the model before the next config so RAM is measured per-config.
        del pipe

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_file": conllu_path,
                "sentences": len(sentences),
                "tokens": n_tokens,
                "platform": sys.platform,
                "configurations": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  Written: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark KazMorphCorpus-2026")
    parser.add_argument(
        "--test-file",
        default="data/test.conllu",
        help="Path to UD Kazakh-KTB test CoNLL-U file",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Limit number of sentences (for quick testing)",
    )
    parser.add_argument(
        "--configs", action="store_true",
        help="Run the multi-config throughput/RAM benchmark (paper Table 9) "
             "and write reports/benchmark.json, instead of the single-config "
             "detailed run.",
    )
    parser.add_argument(
        "--out", default="reports/benchmark.json",
        help="Output path for the --configs benchmark JSON.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        print(f"Error: test file not found: {args.test_file}")
        print("Run scripts/make_splits.py first to generate the canonical split.")
        sys.exit(1)

    if args.configs:
        benchmark_configs(args.test_file, out_path=args.out)
    else:
        run_benchmark(args.test_file, args.max_sentences)


if __name__ == "__main__":
    main()
