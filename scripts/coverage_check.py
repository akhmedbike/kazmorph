"""
Coverage / confidence check for the KazMorph pipeline.

Scans the most frequent word forms in a CoNLL-U corpus, runs them through
the full pipeline, and reports the low-confidence ones — i.e. the words
where the analyzer is weak. This is the diagnostic tool referenced by
REVIEW_RECOMMENDATIONS.md problem #2 ("базовая лексика low confidence").

Typical output for each low-confidence hit:
    өмір   pos=X    conf=0.54   in_lexicon=NOUN   freq=12

Interpretation guide (see REVIEW_RECOMMENDATIONS.md problem #2):
  - A genuinely missing frequent word (in_lexicon=False, pos=UNK) is a safe
    candidate to add to the KNOWN_* lexicons.
  - A word that is already in a lexicon but still low-confidence usually
    reflects genuine ambiguity in the UD training data (e.g. жұмыс appears
    as both NOUN and X in gold). Do NOT force-override those — the model's
    uncertainty is justified.

Usage:
    # Use the corpus frequency list, report words below 0.5 confidence
    PYTHONPATH=. .venv/bin/python scripts/coverage_check.py --top 1000 --threshold 0.5

    # Report only words the analyzer labels UNK (true gaps in the lexicon)
    PYTHONPATH=. .venv/bin/python scripts/coverage_check.py --top 1000 --pos UNK
"""

import argparse
import os
import sys
from collections import Counter

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazmorph.pipeline import Pipeline
from kazmorph.lexicons import KNOWN_NOUNS, KNOWN_VERBS, KNOWN_ADJ


def collect_frequencies(*corpus_paths: str, top_n: int) -> list[str]:
    """Return the `top_n` most frequent alphabetic word forms (lowercased)
    across the given CoNLL-U files, most frequent first."""
    freq: Counter[str] = Counter()
    for path in corpus_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4 or "-" in parts[0] or "." in parts[0]:
                    continue
                form = parts[1]
                if any(c.isalpha() for c in form):
                    freq[form.lower()] += 1
    return [w for w, _ in freq.most_common(top_n)]


def lexicon_status(lemma: str) -> str:
    """Return which KNOWN_* lexicon a lemma belongs to, or '-'."""
    if lemma in KNOWN_NOUNS:
        return "NOUN"
    if lemma in KNOWN_VERBS:
        return "VERB"
    if lemma in KNOWN_ADJ:
        return "ADJ"
    return "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--corpus",
        nargs="+",
        default=["data/train_split.conllu", "data/dev_split.conllu"],
        help="CoNLL-U file(s) to derive the frequency list from.",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Consider the N most frequent word forms (default: 1000).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Report words with confidence below this value (default: 0.5).",
    )
    ap.add_argument(
        "--pos",
        help="Restrict the report to words the analyzer tagged with this POS "
        "(e.g. UNK to find genuine lexicon gaps).",
    )
    args = ap.parse_args()

    top_words = collect_frequencies(*args.corpus, top_n=args.top)
    if not top_words:
        print("No words found in the corpus files.", file=sys.stderr)
        return 1

    pipeline = Pipeline()

    # Process in batches so that token positions stay aligned (one token
    # per space-separated slot). Batch size kept small to avoid any
    # tokenizer drift on long inputs.
    results = []
    batch = 50
    for i in range(0, len(top_words), batch):
        chunk = top_words[i : i + batch]
        results.extend(pipeline.process(" ".join(chunk)))

    # Map back from surface form to result (pipeline lowercases lemmas, but
    # tokens may differ in case; build a lowercase index).
    by_token = {r["token"].lower(): r for r in results}

    print(f"Scanned top-{len(top_words)} frequent forms from "
          f"{', '.join(args.corpus)}")
    print(f"Filter: confidence < {args.threshold}"
          + (f" AND pos == {args.pos}" if args.pos else "") + "\n")

    rows = []
    for word in top_words:
        r = by_token.get(word)
        if r is None:
            continue
        conf = r["confidence"]
        if args.pos is not None and r["pos"] != args.pos:
            continue
        if args.pos is None and conf >= args.threshold:
            continue
        rows.append((word, r["pos"], conf, r["lemma"], lexicon_status(word)))

    if not rows:
        print("No low-confidence words found with the given filters.")
        return 0

    rows.sort(key=lambda x: (x[2], x[0]))
    print(f"{'word':<16}{'pos':<6}{'conf':>7}  {'lemma':<16}{'in_lexicon':<12}")
    print("-" * 60)
    for word, pos, conf, lemma, in_lex in rows:
        print(f"{word:<16}{pos:<6}{conf:>7.3f}  {lemma:<16}{in_lex:<12}")
    print(f"\n{len(rows)} word(s) below threshold out of {len(top_words)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
