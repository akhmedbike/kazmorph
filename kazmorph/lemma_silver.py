"""
Silver lemma corpus: extract (wordform, lemma, upos) pairs from Apertium-kaz.

Two complementary sources are combined:

  1. Stream-format file `corpus/kaz.tagged` — the same source used by
     kazmorph.silver for POS, parsed here to additionally capture the lemma
     (the part before the first `<tag>`).

  2. CG-disambiguated files `texts/texts2/*.tagged.clean.txt` — Constraint
     Grammar output for texts of Kazakh classical authors (Абай, Әуезов,
     Махамбет, Байтұрсынұлы, Алтынсарин, Дулатұлы, Мақатаев). The CG has
     already resolved most reading ambiguities, so the first surviving reading
     per word is high-quality. These files are ~3x larger than kaz.tagged and
     cover a much broader vocabulary.

The result is ~9000 unique wordforms with a lemma — roughly double the gold
UD Kazakh-KTB (~4400 forms). This silver lemma corpus lets a neural
lemmatizer learn correct strip depths and stem alternations that the
rule-based RMA gets wrong (82% of RMA's errors are under-stripping).

Output format: TSV with columns  wordform \t lemma \t upos  (lowercased),
one pair per line, deduplicated. Files written start with "silver_" so
kazmorph.corpus.load_pool() automatically excludes them from the gold POS
evaluation pool.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

from kazmorph.silver import parse_stream, best_upos, apertium_to_upos, parse_reading


def _from_stream(path: str) -> dict[str, set[tuple[str, str]]]:
    """Extract {form_lower: {(lemma, upos), ...}} from an Apertium stream file."""
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    text = open(path, encoding="utf-8").read()
    for lu in parse_stream(text):
        upos, lemma = best_upos(lu)
        if lemma and not lemma.startswith("*") and lu.wordform:
            out[lu.wordform.lower()].add((lemma.lower(), upos))
    return out


# CG "clean" format: a block is
#   "<surface_form>"
#   \t"lemma" tag tag ...      <- active (surviving) reading(s)
#   ;\t"lemma" tag tag ...     <- removed by CG, skip
#   \t"*form"                  <- failed analysis, skip
# A new "<...>" line starts the next block. We take the first non-removed,
# non-star reading of each block as the disambiguated analysis.
_CG_FORM_RE = re.compile(r'^"<([^">]+)>"\s*$')
_CG_READ_RE = re.compile(r'^\t"([^"]+)"\s+(\S+)')


def _from_vislcg(path: str) -> dict[str, set[tuple[str, str]]]:
    """Extract {form_lower: {(lemma, upos), ...}} from a CG .tagged.clean file."""
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    cur_form: str | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _CG_FORM_RE.match(line)
            if m:
                cur_form = m.group(1)
                continue
            if cur_form is None:
                continue
            # removed reading (CG) starts with ';'
            if line.startswith(";"):
                continue
            rm = _CG_READ_RE.match(line)
            if not rm:
                continue
            lemma = rm.group(1)
            if lemma.startswith("*"):
                continue  # failed analysis
            primary = rm.group(2)
            # we don't have the full subtag set cheaply here; map on primary only
            upos = apertium_to_upos(primary, set()) or "X"
            out[cur_form.lower()].add((lemma.lower(), upos))
    return out


def collect_lemma_pairs(
    stream_files: list[str] | None = None,
    vislcg_files: list[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Build a {form_lower: (lemma, upos)} dictionary from the given sources.

    When a form has multiple readings, the single most common lemma/upos across
    sources wins (majority vote), with ties broken alphabetically for stability.
    This collapses the ~3% ambiguous forms to one label, which is what a
    lemmatizer needs for training.
    """
    if stream_files is None and vislcg_files is None:
        # discover defaults relative to the apertium-kaz checkout
        base = _find_apertium_base()
        stream_files = [os.path.join(base, "corpus", "kaz.tagged")] if base else []
        vislcg_files = _glob_vislcg(base) if base else []

    counts: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    for sf in stream_files or []:
        if os.path.exists(sf):
            for form, readings in _from_stream(sf).items():
                for pair in readings:
                    counts[form][pair] += 1
    for vf in vislcg_files or []:
        if os.path.exists(vf):
            for form, readings in _from_vislcg(vf).items():
                for pair in readings:
                    counts[form][pair] += 1

    # majority vote per form
    resolved: dict[str, tuple[str, str]] = {}
    for form, label_counts in counts.items():
        best = max(label_counts.items(), key=lambda kv: (kv[1], kv[0]))
        resolved[form] = best[0]
    return resolved


def _find_apertium_base() -> str | None:
    """Locate the apertium-kaz checkout (handles the Trash/restored-path case)."""
    candidates = [
        os.path.expanduser("~/apertium-kaz"),
        os.path.expanduser("~/.local/share/Trash/files/apertium-kaz"),
        "/home/maximus/.local/share/Trash/files/apertium-kaz",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _glob_vislcg(base: str) -> list[str]:
    import glob
    return sorted(glob.glob(os.path.join(base, "texts", "texts2", "*.tagged.clean.txt")))


def write_tsv(pairs: dict[str, tuple[str, str]], path: str) -> int:
    """Write {form: (lemma, upos)} as a TSV file; return number of lines."""
    with open(path, "w", encoding="utf-8") as f:
        for form in sorted(pairs):
            lemma, upos = pairs[form]
            f.write(f"{form}\t{lemma}\t{upos}\n")
    return len(pairs)


# ============================================================
#  Silver SENTENCES in CoNLL-U (for contextual training)
# ============================================================
_CG_FORM_RE_S = re.compile(r'^"<([^">]+)>"\s*$')
_CG_READ_RE_S = re.compile(r'^\t"([^"]+)"\s+(\S+)')


def parse_vislcg_sentences(path: str) -> list[list[tuple[str, str, str]]]:
    """
    Parse a CG .tagged.clean file into a list of sentences.

    Each sentence is a list of (form, lemma, upos) tuples. Sentences are split
    on blank lines. Only the first surviving (non-removed) reading per token is
    used. Failed analyses (`*form`) keep the surface form as its own lemma with
    upos=X so sentence alignment is preserved (the lemmatizer can still learn
    from surrounding tokens). Removed readings (lines starting with ';') and
    their nested feature lines are skipped.
    """
    from kazmorph.silver import apertium_to_upos

    sentences: list[list[tuple[str, str, str]]] = []
    cur: list[tuple[str, str, str]] = []
    cur_form: str | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if cur:
                    sentences.append(cur)
                    cur = []
                cur_form = None
                continue
            m = _CG_FORM_RE_S.match(line)
            if m:
                cur_form = m.group(1)
                continue
            if cur_form is None or line.startswith(";"):
                continue
            rm = _CG_READ_RE_S.match(line)
            if not rm:
                continue
            lemma = rm.group(1)
            if lemma.startswith("*"):
                cur.append((cur_form, cur_form, "X"))
                continue
            upos = apertium_to_upos(rm.group(2), set()) or "X"
            cur.append((cur_form, lemma, upos))
    if cur:
        sentences.append(cur)
    return sentences


def write_silver_conllu(vislcg_files: list[str], out_path: str) -> dict:
    """
    Convert CG .tagged.clean files to a single silver CoNLL-U file (form/lemma/upos).

    This is the *contextual* silver resource: full sentences with lemmas and
    UPOS, suitable for training both the POS tagger and the lemmatizer.
    Files written start with "silver_" so load_pool() keeps them out of the
    gold evaluation pool automatically.
    """
    from kazmorph.corpus import POS_TAGS

    total_sents = 0
    total_tokens = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for vf in vislcg_files:
            if not os.path.exists(vf):
                continue
            for sent in parse_vislcg_sentences(vf):
                if len(sent) < 2:
                    continue
                forms = " ".join(t[0] for t in sent)
                out.write(f"# sent_id = silver-sent-{total_sents}\n")
                out.write(f"# text = {forms}\n")
                for i, (form, lemma, upos) in enumerate(sent, 1):
                    if upos not in POS_TAGS:
                        upos = "X"
                    out.write(f"{i}\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n")
                out.write("\n")
                total_sents += 1
                total_tokens += len(sent)
    return {"sentences": total_sents, "tokens": total_tokens,
            "files": len([f for f in vislcg_files if os.path.exists(f)])}
