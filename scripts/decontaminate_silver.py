#!/usr/bin/env python3
"""
Decontaminate silver training data against the held-out test set.

WHY THIS EXISTS
---------------
The silver CoNLL-U files (`data/silver_apertium.conllu`,
`data/silver_sentences.conllu`) were generated from Apertium / Wikipedia
sources. Some of them contain *exact sentence duplicates* of the gold test
split (e.g. `silver_apertium.conllu` shares 127 full sentences with
`data/test.conllu`). An exact sentence duplicate is a genuine label leak: the
model sees the identical (tokens, tags) pair during training that it will be
evaluated on, which inflates the reported POS / lemma accuracy.

This script removes those exact duplicates. The default filter key is
``sentences`` (exact de-duplication by sentence fingerprint), which is the
methodologically correct granularity: it removes only real leaks while leaving
the rest of the silver data — and its coverage of frequent Kazakh vocabulary —
intact.

WARNING on vocabulary-level filtering
-------------------------------------
Filtering by ``--by lemmas`` or ``--by forms`` removes any silver sentence that
*contains* a test lemma/form. For a low-resource corpus this is far too
aggressive: UD Kazakh-KTB's 717 test lemmas are mostly frequent function words
and high-frequency nouns, so nearly every Kazakh sentence contains at least one
of them. In practice this removes ~99.5% of the silver data (verified) without
removing any additional sentence-level leak beyond what ``sentences`` already
catches. Use ``lemmas``/``forms`` only if you understand you are discarding
legitimate training data along with vocabulary overlap.

USAGE
-----
    # Recommended: remove exact sentence duplicates only.
    .venv/bin/python scripts/decontaminate_silver.py \\
        --test-file data/test.conllu \\
        --silver data/silver_apertium.conllu data/silver_sentences.conllu \\
        --out-dir data/decontaminated

    # Strict (destroys most silver data — see warning above):
    .venv/bin/python scripts/decontaminate_silver.py --by lemmas ...

The script never overwrites the original silver files; it only writes new ones
under ``--out-dir`` with the same basenames. A JSON report of what was removed
is saved next to the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `kazmorph` importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kazmorph.corpus import parse_conllu, Sentence  # noqa: E402


def _vocab_of(path: Path, key: str) -> set[str]:
    """Lowercased lemma/form vocabulary of a CoNLL-U file."""
    vocab: set[str] = set()
    for sent in parse_conllu(str(path)):
        for tok in sent.tokens:
            if key == "lemmas":
                lemma = (tok.lemma or "").strip().lower()
                if lemma and lemma != "_":
                    vocab.add(lemma)
            else:  # forms
                form = (tok.form or "").strip().lower()
                if form:
                    vocab.add(form)
    return vocab


def _sentence_keys(sent: Sentence, key: str) -> set[str]:
    out: set[str] = set()
    for tok in sent.tokens:
        if key == "lemmas":
            v = (tok.lemma or "").strip().lower()
        else:
            v = (tok.form or "").strip().lower()
        if v and v != "_":
            out.add(v)
    return out


def _write_conllu(sentences: list[Sentence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences, start=1):
            sid = s.sent_id or f"silver-clean-{i}"
            f.write(f"# sent_id = {sid}\n")
            for j, tok in enumerate(s.tokens, start=1):
                # columns: id form lemma upos xpos _ head deprel _ deps misc
                upos = tok.upos or "_"
                f.write(
                    f"{j}\t{tok.form}\t{tok.lemma}\t{upos}\t_\t_\t"
                    f"_\t_\t_\t_\t_\n"
                )
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-file", default="data/test.conllu",
                    help="Held-out test CoNLL-U (default: data/test.conllu)")
    ap.add_argument("--silver", nargs="+",
                    default=["data/silver_apertium.conllu",
                             "data/silver_sentences.conllu"],
                    help="Silver CoNLL-U files to decontaminate.")
    ap.add_argument("--out-dir", default="data/decontaminated",
                    help="Where to write the cleaned silver files.")
    ap.add_argument("--by", choices=["sentences", "lemmas", "forms", "both"],
                    default="sentences",
                    help="Filter granularity. 'sentences' (default) removes "
                         "only EXACT sentence duplicates with the test set "
                         "(the only genuine label leak). 'lemmas'/'forms'/"
                         "'both' additionally drop any silver sentence sharing "
                         "vocabulary with test — this is far too aggressive "
                         "for a low-resource corpus and destroys ~99% of the "
                         "silver data; see the module docstring warning.")
    args = ap.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"ERROR: test file not found: {test_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact-sentence dedup is always applied (it is the real leak). Vocabulary
    # filtering is an additional, optional strictness layer.
    do_sentence_dedup = args.by in ("sentences", "lemmas", "forms", "both")
    vocab_keys = (["lemmas", "forms"] if args.by == "both"
                  else ([args.by] if args.by in ("lemmas", "forms") else []))
    test_vocab = {k: _vocab_of(test_path, k) for k in vocab_keys}
    for k, v in test_vocab.items():
        print(f"test {k} vocabulary: {len(v)} unique")

    # Fingerprints of the test set for exact-sentence de-duplication.
    test_fps: set[tuple[str, ...]] = set()
    if do_sentence_dedup:
        test_fps = {s.fingerprint() for s in parse_conllu(str(test_path))}
        print(f"test sentences (fingerprints): {len(test_fps)}")

    report = {
        "test_file": str(test_path),
        "filter_key": args.by,
        "exact_sentence_dedup": do_sentence_dedup,
        "test_sentence_count": len(test_fps),
        "test_vocab_sizes": {k: len(v) for k, v in test_vocab.items()},
        "files": [],
    }

    for silver_path_str in args.silver:
        silver_path = Path(silver_path_str)
        if not silver_path.exists():
            print(f"  SKIP (not found): {silver_path}")
            continue

        sents = list(parse_conllu(str(silver_path)))
        kept: list[Sentence] = []
        removed_dup = 0
        removed_vocab = 0
        for s in sents:
            # Layer 1: exact sentence duplicate with test (the real leak).
            if do_sentence_dedup and s.fingerprint() in test_fps:
                removed_dup += 1
                continue
            # Layer 2 (optional, strict): vocabulary overlap.
            if vocab_keys:
                hit = False
                for k in vocab_keys:
                    if _sentence_keys(s, k) & test_vocab[k]:
                        hit = True
                        break
                if hit:
                    removed_vocab += 1
                    continue
            kept.append(s)

        out_path = out_dir / silver_path.name
        _write_conllu(kept, out_path)

        n_before = len(sents)
        n_after = len(kept)
        entry = {
            "file": str(silver_path),
            "output": str(out_path),
            "sentences_before": n_before,
            "removed_exact_duplicates": removed_dup,
            "removed_vocab_overlap": removed_vocab,
            "sentences_after": n_after,
            "removed_fraction": round((n_before - n_after) / n_before, 4) if n_before else 0.0,
        }
        report["files"].append(entry)
        print(f"  {silver_path.name}: {n_before} -> {n_after} sentences "
              f"(exact-dup removed {removed_dup}, vocab removed {removed_vocab}, "
              f"{entry['removed_fraction']*100:.1f}%)")

    report_path = out_dir / "decontamination_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
