#!/usr/bin/env python3
"""
Materialize the canonical 70/15/15 train/dev/test split for UD Kazakh-KTB.

UD Kazakh-KTB ships no canonical three-way split (the treebank README states it
is too small for one and recommends jack-knife / cross-validation). To obtain
honest, single-number headline results in the paper we draw ONE fixed stratified
partition from the deduplicated gold pool and commit it to disk, so that:

  - training sees ONLY data/train.conllu,
  - hyperparameter selection / calibration sees ONLY data/dev.conllu,
  - the final reported metrics are computed ONLY on data/test.conllu.

Reproducibility: the partition is a pure function of (pool, dev_frac, test_frac,
seed=42). Re-running this script regenerates byte-identical splits (modulo the
in-block shuffle, which is also seeded and therefore deterministic).

Usage:
    .venv/bin/python scripts/make_splits.py
    .venv/bin/python scripts/make_splits.py --dev-frac 0.15 --test-frac 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import (  # noqa: E402
    Sentence,
    load_pool,
    stratified_three_way_split,
    write_conllu,
)


def _stats(name: str, sentences: list[Sentence]) -> dict:
    n_sents = len(sentences)
    tokens = [t for s in sentences for t in s.tokens]
    n_tokens = len(tokens)
    avg_len = n_tokens / n_sents if n_sents else 0.0
    tags = Counter(t.upos for t in tokens)
    return {
        "split": name,
        "sentences": n_sents,
        "tokens": n_tokens,
        "avg_tokens_per_sent": round(avg_len, 2),
        "unique_upos": len(tags),
    }


def _oov_rate(train_sents: list[Sentence], eval_sents: list[Sentence]) -> float:
    """Fraction of eval tokens whose lowercase form is unseen in train."""
    train_vocab = {t.form.lower() for s in train_sents for t in s.tokens}
    eval_tokens = [t.form.lower() for s in eval_sents for t in s.tokens]
    if not eval_tokens:
        return 0.0
    oov = sum(1 for w in eval_tokens if w not in train_vocab)
    return round(oov / len(eval_tokens), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--suffix", default="",
        help="Filename suffix for the generated splits, e.g. '_v1' -> train_v1.conllu",
    )
    args = ap.parse_args()

    pool = load_pool(dedup=True)
    print(f"Loaded deduplicated gold pool: {len(pool)} sentences")

    train, dev, test = stratified_three_way_split(
        pool,
        dev_frac=args.dev_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    sfx = args.suffix
    out = {
        "train": os.path.join(args.data_dir, f"train{sfx}.conllu"),
        "dev": os.path.join(args.data_dir, f"dev{sfx}.conllu"),
        "test": os.path.join(args.data_dir, f"test{sfx}.conllu"),
    }
    for name, sents in (("train", train), ("dev", dev), ("test", test)):
        write_conllu(sents, out[name])
        print(f"  wrote {out[name]}  ({len(sents)} sentences)")

    # Statistics table for the paper.
    rows = [
        _stats("Training", train),
        _stats("Development", dev),
        _stats("Test", test),
    ]
    total_sents = sum(r["sentences"] for r in rows)
    total_tokens = sum(r["tokens"] for r in rows)
    rows.append({
        "split": "Total",
        "sentences": total_sents,
        "tokens": total_tokens,
        "avg_tokens_per_sent": round(total_tokens / total_sents, 2) if total_sents else 0.0,
        "unique_upos": 17,
    })

    print("\n=== Corpus statistics ===")
    print(f"{'Split':<14}{'Sentences':>12}{'Tokens':>10}{'Avg/Sent':>10}{'UPOS':>6}")
    for r in rows:
        print(
            f"{r['split']:<14}{r['sentences']:>12}{r['tokens']:>10}"
            f"{r['avg_tokens_per_sent']:>10}{r['unique_upos']:>6}"
        )

    print("\n=== OOV rate (vs. train vocabulary) ===")
    print(f"  Development: {_oov_rate(train, dev) * 100:.1f}%")
    print(f"  Test:        {_oov_rate(train, test) * 100:.1f}%")

    # Provenance record for reproducibility.
    import json
    meta = {
        "seed": args.seed,
        "dev_frac": args.dev_frac,
        "test_frac": args.test_frac,
        "pool_sentences": len(pool),
        "splits": {k: os.path.relpath(v, PROJECT_ROOT) for k, v in out.items()},
        "stats": rows,
        "oov_dev_vs_train": _oov_rate(train, dev),
        "oov_test_vs_train": _oov_rate(train, test),
    }
    meta_path = os.path.join(args.data_dir, f"split_meta{sfx}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nWrote split metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
