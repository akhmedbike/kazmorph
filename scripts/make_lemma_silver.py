#!/usr/bin/env python3
"""
Build the silver lemma corpus (data/silver_lemmas.tsv) from Apertium-kaz.

Extracts (wordform, lemma, upos) triples from the Apertium-kaz checkout — both
the stream-format `corpus/kaz.tagged` and the CG-disambiguated
`texts/texts2/*.tagged.clean.txt` files of Kazakh classical authors — and
writes a deduplicated TSV for training the neural lemmatizer.

Usage:
    python scripts/make_lemma_silver.py \\
        --apertium /path/to/apertium-kaz \\
        --output data/silver_lemmas.tsv
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.lemma_silver import collect_lemma_pairs, write_tsv, _find_apertium_base


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apertium", default=None,
                   help="Path to apertium-kaz checkout (default: auto-detect)")
    p.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "silver_lemmas.tsv"))
    args = p.parse_args()

    base = args.apertium or _find_apertium_base()
    if not base or not os.path.isdir(base):
        sys.exit(f"error: apertium-kaz checkout not found. Pass --apertium /path/to/apertium-kaz")

    import glob
    stream_files = [os.path.join(base, "corpus", "kaz.tagged")]
    vislcg_files = sorted(glob.glob(os.path.join(base, "texts", "texts2", "*.tagged.clean.txt")))
    print(f"Apertium base: {base}")
    print(f"Stream files:  {sum(os.path.exists(f) for f in stream_files)}")
    print(f"CG files:      {len(vislcg_files)}")

    pairs = collect_lemma_pairs(stream_files=stream_files, vislcg_files=vislcg_files)
    n = write_tsv(pairs, args.output)
    inflected = sum(1 for f, (l, _) in pairs.items() if f != l)
    print(f"\nWrote {n} unique wordforms → {args.output}")
    print(f"  inflected (form != lemma): {inflected} ({inflected/n*100:.0f}%)")


if __name__ == "__main__":
    main()
