"""
Synthesize stem-alternation training pairs for the lemmatizer.

Problem (REVIEW_RECOMMENDATIONS.md #1): q/g stem alternation in Kazakh.
The dictionary lemma carries the voiceless consonant (бастық, мектеп, аймақ),
but the surface form before a vowel-initial suffix voices it
(бастығ-ы, мектеб-і, аймағ-ы). The UD gold corpus has only ~262 alternation
examples and the silver has ~362 — too few for the model to learn the
"include the alternating consonant" cut reliably.

This script takes nouns from KNOWN_NOUNS that end in a voiceless consonant
(қ, к, п, т, с, ш) and synthesizes their declined forms by:
  1. voicing the final consonant (reverse of _DEVOICE: қ→ғ, к→г, п→б, ...)
  2. appending a vowel-initial suffix (possessive -ы/-і, dative -а/-е)
  3. choosing the suffix variant by vowel harmony (front/back)

Output: data/silver_alternations.tsv with rows form<TAB>lemma<TAB>NOUN.
These are mixed into lemmatizer training via the --silver-conllu path is
for sentences; for form/lemma pairs we extend data/silver_lemmas.tsv, OR
pass this file as an additional --silver argument (train_lemma accepts
multiple via a merged approach — see its load_silver_lemmas, single file).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/synth_alternations.py \\
        --output data/silver_alternations.tsv --max-per-lemma 2
    # then merge into the training silver dict, or pass alongside the
    # existing silver_lemmas.tsv via the train_lemma.py --silver flag
    # (which currently takes a single path — see merge step below).

Vowel harmony rules (Kazakh):
  Front vowels: ә е и і ө ү (and и)  → front suffix variant (-і, -е, -ге)
  Back vowels:  а ы о ұ (and у)      → back suffix variant (-ы, -а, -ға)
  The stem's LAST vowel determines the suffix row. Russian borrowings and
  mixed stems default to back harmony.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.lexicons import KNOWN_NOUNS

# Voiceless → voiced (the FORWARD direction, inverse of _DEVOICE in train_lemma).
# A noun ending in a key here voices it before a vowel-initial suffix.
_VOICE = {"қ": "ғ", "к": "г", "п": "б", "т": "д", "с": "з", "ш": "ж"}

_FRONT_VOWELS = frozenset("әеиіөү")
_BACK_VOWELS = frozenset("аыоуұ")

# Vowel-initial suffixes to synthesize. Each entry: (back_variant, front_variant).
# These all START WITH A VOWEL, which is what triggers the voicing.
# Possessive-3 and dative are the most common alternation contexts in the corpus.
_VOWEL_SUFFIXES = [
    ("ы", "і"),    # POSS.3
    ("а", "е"),    # DAT (already voiced in the suffix itself for ға/ге, but the
                   # STEM consonant still voices: аймақ→аймағ-а, көлік→көліг-е)
    ("ын", "ін"),  # POSS.3 + ACC-style
]


def _harmony_class(stem: str) -> str:
    """Return 'front' or 'back' based on the last vowel in the stem.
    Defaults to 'back' for stems with no recognized vowel (rare)."""
    for ch in reversed(stem):
        if ch in _FRONT_VOWELS:
            return "front"
        if ch in _BACK_VOWELS:
            return "back"
    return "back"


def _voice_final(stem: str) -> str | None:
    """Return the stem with its final voiceless consonant voiced, or None if
    the stem doesn't end in a voiceless consonant that has a voiced pair."""
    if len(stem) < 2 or stem[-1] not in _VOICE:
        return None
    return stem[:-1] + _VOICE[stem[-1]]


def generate_pairs(lemma: str, max_per_lemma: int) -> list[tuple[str, str]]:
    """Generate up to max_per_lemma (form, lemma) alternation pairs for one lemma.
    Returns [] if the lemma doesn't end in a voiceless consonant."""
    voiced = _voice_final(lemma)
    if voiced is None:
        return []
    harmony = _harmony_class(lemma)
    pairs = []
    for back_suf, front_suf in _VOWEL_SUFFIXES:
        suf = front_suf if harmony == "front" else back_suf
        form = voiced + suf
        pairs.append((form, lemma))
        if len(pairs) >= max_per_lemma:
            break
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "silver_alternations.tsv"),
        help="Output TSV path (form<TAB>lemma<TAB>NOUN).",
    )
    ap.add_argument(
        "--max-per-lemma",
        type=int,
        default=2,
        help="Max synthesized forms per lemma (default: 2).",
    )
    args = ap.parse_args()

    pairs: list[tuple[str, str]] = []
    skipped = 0
    for lemma in sorted(KNOWN_NOUNS):
        if len(lemma) < 3:
            continue
        new = generate_pairs(lemma, args.max_per_lemma)
        if new:
            pairs.extend(new)
        else:
            skipped += 1

    # Dedup (a form could map to multiple lemmas only via bug; keep first).
    seen = set()
    unique = []
    for form, lemma in pairs:
        if form not in seen:
            seen.add(form)
            unique.append((form, lemma))

    with open(args.output, "w", encoding="utf-8") as f:
        for form, lemma in unique:
            f.write(f"{form}\t{lemma}\tNOUN\n")

    print(f"Nouns scanned: {len(KNOWN_NOUNS)}")
    print(f"  ending in voiceless consonant: {len(KNOWN_NOUNS) - skipped}")
    print(f"  skipped (too short / no voiceless final): {skipped}")
    print(f"Pairs written: {len(unique)} → {args.output}")
    print(f"\nSample (first 15):")
    for form, lemma in unique[:15]:
        print(f"  {form:20s} → {lemma}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
