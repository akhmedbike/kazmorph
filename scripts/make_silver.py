#!/usr/bin/env python3
"""
Convert Apertium-kaz analyzed text into silver UPOS CoNLL-U for training.

This is the "silver data" pipeline described in the project roadmap. It does
NOT require compiling the Apertium finite-state transducer: it consumes the
stream-format output of `apertium -d . kaz-tagger` (or the pre-analyzed
`corpus/kaz.tagged` shipped with apertium-kaz) and maps the Apertium tag scheme
to UPOS using the official apertium-kaz.kaz.udx mapping.

Usage
-----
# 1) Use the pre-analyzed corpus bundled with apertium-kaz (no tools needed):
python scripts/make_silver.py \
    --tagged /path/to/apertium-kaz/corpus/kaz.tagged \
    --output data/silver_apertium.conllu

# 2) Analyze your own raw text with the apertium-tagger, then convert:
echo "Бүгін ауа райы жақсы." | apertium -d /path/to/apertium-kaz kaz-tagger > my.tagged
python scripts/make_silver.py --tagged my.tagged --output data/silver_mine.conllu

# 3) Drop a fraction of unknown-X tokens to reduce label noise:
python scripts/make_silver.py --tagged ... --output ... --drop-unknown 0.5
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.silver import convert_apertium_corpus


def main():
    p = argparse.ArgumentParser(
        description="Convert Apertium-kaz stream output to silver UPOS CoNLL-U.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tagged", required=True,
                   help="Path to Apertium stream-format file (^wf/lemma<tags>$).")
    p.add_argument("--output", required=True,
                   help="Output CoNLL-U path.")
    p.add_argument("--drop-unknown", type=float, default=0.0,
                   help="Fraction of unknown (X) tokens to drop (default: 0.0).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.exists(args.tagged):
        sys.exit(f"error: input not found: {args.tagged}")

    stats = convert_apertium_corpus(
        args.tagged, args.output,
        drop_unknown_rate=args.drop_unknown,
        seed=args.seed,
    )
    print("Silver conversion complete:")
    for k, v in stats.items():
        print(f"  {k:>16s}: {v}")
    print(f"\nWrote {stats['written']} sentences → {args.output}")


if __name__ == "__main__":
    main()
