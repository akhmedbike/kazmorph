#!/usr/bin/env python3
"""
Synthesize training sentences for rare, closed-class POS tags.

WHY THIS EXISTS
---------------
The neural POS tagger is under-fit on rare classes. The silver data (Apertium
FST output) structurally cannot emit certain UD tags, so they are absent from
silver and nearly absent from the tiny gold train (754 sentences):

  - PART: only 23 tokens in gold train (9 unique). These are the Kazakh
    interrogative/clitic particles (ма, ме, ба, бе, пе, ау, ше, шы, қой, шығар).
    It is a CLOSED class — small, well-defined, and high-frequency in real text.
  - INTJ: 18 tokens in gold train. Also largely closed (Жоқ, Иә, Рақмет, Міне...).

Because these are closed functional classes, we can safely manufacture extra
training examples by placing the particle in short, grammatical carrier
sentences. The particle's POS tag is unambiguous regardless of context, so the
synthesized labels are reliable (unlike, say, NOUN-vs-VERB which depends on
syntax). This is the standard remedy for a closed-class that a small corpus
under-represents.

WHAT THIS DOES NOT DO
---------------------
It does NOT synthesize the ``X`` tag. Inspecting gold X tokens shows they are
ordinary frequent Kazakh words that UD-KTB happened to mark X (жүзеге, жауап,
қол, жұмыс, өмір...) — there is no morphological pattern to synthesize, and
injecting random words labeled X would just add label noise. X needs better
gold annotation, not synthesis.

OUTPUT
------
A CoNLL-U file ``data/synth_rare_tags.conllu`` of synthesized sentences. Each
sentence is a minimal carrier (verb + clitic, or interjection + punctuation)
that exposes the rare tag in a clean context. The downstream trainer treats
these as GOLD (silver_weight 1.0) — they are not silver/FST output — so the
run command passes them via a separate gold-augmentation channel.

    .venv/bin/python scripts/synth_rare_tags.py \
        --out data/synth_rare_tags.conllu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Closed inventories ──────────────────────────────────────────────────────
# Kazakh interrogative/clitic particles (UPOS=PART). These attach to the host
# word but are written as separate tokens in UD Kazakh-KTB. Each gets its own
# line in the output.
PART_PARTICLES: list[str] = ["ма", "ме", "ба", "бе", "пе", "ау", "ше", "шы",
                              "қой", "шығар", "ғой"]

# Common interjections (UPOS=INTJ). Drawn from gold train + standard Kazakh.
INTJS: list[str] = ["Жоқ", "Иә", "Рақмет", "Міне", "Қане", "Ай", "Мейлі",
                    "Оғой", "Паһ", "Ә"]

# Carrier verbs (UPOS=VERB) to host a clitic particle. These are unambiguous
# verbs from the gold train, so the synthesized VERB label is safe.
HOST_VERBS: list[str] = ["бардым", "келдім", "көрдім", "айттым", "жаздым",
                          "істедім", "ұнады", "тұрды"]


def _emit_sentence(lines: list[tuple[str, str, str]], sent_id: str) -> list[str]:
    """Format one synthesized sentence as a CoNLL-U block.

    ``lines`` is a list of (form, lemma, upos) tuples in token order.
    """
    out = [f"# sent_id = {sent_id}"]
    for i, (form, lemma, upos) in enumerate(lines, start=1):
        out.append(f"{i}\t{form}\t{lemma}\t{upos}\t_\t_\t_\t_\t_\t_")
    out.append("")  # blank line separates sentences
    return out


def synthesize() -> list[str]:
    """Build all synthesized sentence blocks."""
    blocks: list[str] = []

    # Pattern 1: VERB + PART (clitic attaches to the verb). The particle is a
    # separate token with UPOS=PART, the verb keeps VERB.
    sid = 0
    for verb in HOST_VERBS:
        for part in PART_PARTICLES:
            sid += 1
            blocks.extend(_emit_sentence(
                [(verb, verb, "VERB"), (part, part, "PART"), (".", ".", "PUNCT")],
                f"synth-part-{sid}",
            ))

    # Pattern 2: INTJ + PUNCT. A standalone interjection closing an utterance.
    for ij in INTJS:
        sid += 1
        blocks.extend(_emit_sentence(
            [(ij, ij, "INTJ"), (".", ".", "PUNCT")],
            f"synth-intj-{sid}",
        ))

    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/synth/synth_rare_tags.conllu",
                    help="Output CoNLL-U path. Lives under data/synth/ so "
                         "load_pool() (which scans data/*.conllu) does not pull "
                         "synthesized sentences into the gold pool — they must "
                         "be fed to training explicitly via --silver.")
    args = ap.parse_args()

    blocks = synthesize()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks), encoding="utf-8")

    n_sents = sum(1 for b in blocks if b.startswith("# sent_id"))
    # Tally tags for the log line.
    from collections import Counter
    tag_counts: Counter[str] = Counter()
    for b in blocks:
        if b.startswith("#") or not b.strip():
            continue
        cols = b.split("\t")
        if len(cols) >= 4:
            tag_counts[cols[3]] += 1

    print(f"Wrote {n_sents} synthesized sentences to {out_path}")
    print(f"Tag distribution: {dict(tag_counts.most_common())}")
    print("\nThese should be added to POS training as GOLD-weight data (not silver):")
    print("  --gold-extra data/synth_rare_tags.conllu  (if/when supported)")
    print("or concatenated into train.conllu before the decontaminated split step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
