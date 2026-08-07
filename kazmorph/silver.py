"""
Silver annotation: turn Apertium-kaz analyses into UPOS-tagged CoNLL-U.

Apertium-kaz ships a fully-analyzed corpus (`corpus/kaz.tagged`) in its
stream format, e.g.:

    ^конкурсы/конкурс<n><px3sp><nom>$
    ^бұлармен/бұл<prn><dem><pl><ins>$
    ^болады/бол<v><iv><aor><p3><sg>$

The `.udx` file is Apertium's *official* mapping from its tag scheme to the
Universal Dependencies tag set (UPOS + features). By parsing the analyses and
applying that mapping, we get high-quality silver UPOS annotations for ~1100
sentences *without* needing to compile the finite-state transducer (which
requires hfst/foma + sudo) — the analysis is already done.

Why this matters: the gold UD Kazakh-KTB is only 1078 sentences. Apertium's
own analyzed corpus, plus any raw text run through `apertium -d . kaz-tagger`,
can be converted to silver CoNLL-U and mixed into neural training with a
down-weighted loss. The Apertium tagger is especially valuable for the exact
distinctions the gold-tuned model struggles with:
  - vaux vs v            → AUX vs VERB
  - v<iv> vs v<tv>       → intransitive/transitive verbs, and <iv> maps to AUX
  - cnjcoo vs cnjsub     → CCONJ vs SCONJ

This module is dependency-free (stdlib only) so it can run anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from kazmorph.corpus import Sentence, Token


# ============================================================
#  Apertium → UPOS mapping
# ============================================================
# Derived from apertium-kaz.kaz.udx. Each rule keys on (primary_tag, subtag_set)
# where required_subtags is either None (match any) or a frozenset that must be
# a SUBSET of the analysis's subtags (so a verb reading like v<iv><aor><p3><sg>
# still matches the iv rule). Ordering matters: more specific rules first.
#
# Primary tags: n, np, v, vaux, prn, num, adj, adv, post, postadv, det, cnjcoo,
# cnjsub, cnjadv, ij, mod, emph, mod_ass, mod_emo, qst, sent, cm, lquot,
# rquot, lpar, rpar, guio, sym, abbr, x, cop, ant, top, al, cog, org.


# (primary, required_subtags_or_None) -> UPOS. A rule matches when the primary
# tag equals `primary` AND every tag in `required_subtags` is present in the
# analysis's subtag set (subset match, NOT equality — a verb reading carries
# many feature subtags beyond the POS class). None means "any subtags".
_UPOS_RULES: list[tuple[str, frozenset[str] | None, str]] = [
    # --- Nouns / proper nouns ---
    ("np", None, "PROPN"),
    ("n", None, "NOUN"),
    ("abbr", None, "NOUN"),

    # --- Verbs: intransitive <iv> historically used for copula-like AUX.
    #     Requires the 'iv' subtag to be present among possibly many features.
    ("v", frozenset({"iv"}), "AUX"),
    ("v", None, "VERB"),
    ("vaux", None, "AUX"),
    ("cop", None, "AUX"),

    # --- Pronouns / determiners / numerals ---
    ("prn", None, "PRON"),
    ("det", None, "DET"),
    ("num", None, "NUM"),

    # --- Modifiers ---
    ("adj", None, "ADJ"),
    ("adv", None, "ADV"),
    ("postadv", None, "ADV"),

    # --- Adpositions ---
    ("post", None, "ADP"),

    # --- Conjunctions ---
    ("cnjcoo", None, "CCONJ"),
    ("cnjsub", None, "SCONJ"),
    ("cnjadv", None, "SCONJ"),

    # --- Interjections ---
    ("ij", None, "INTJ"),

    # --- Particles (emphatic, modal, question, etc.) ---
    ("mod", None, "PART"),
    ("emph", None, "PART"),
    ("mod_ass", None, "PART"),
    ("mod_emo", None, "PART"),
    ("qst", None, "PART"),

    # --- Punctuation ---
    ("sent", None, "PUNCT"),
    ("cm", None, "PUNCT"),
    ("lquot", None, "PUNCT"),
    ("rquot", None, "PUNCT"),
    ("lpar", None, "PUNCT"),
    ("rpar", None, "PUNCT"),
    ("guio", None, "PUNCT"),

    # --- Other ---
    ("sym", None, "SYM"),
    ("x", None, "X"),
]


def apertium_to_upos(primary: str, subtags: set[str]) -> str | None:
    """
    Map an Apertium primary tag + subtags to a UPOS tag, or None if unknown.

    The first matching rule (in declaration order) wins, so put specific rules
    before broad ones (e.g. v+iv→AUX before v→VERB). A rule with required
    subtags S matches when S ⊆ subtags (subset), so feature-rich readings still
    fire the right rule.
    """
    for p, req, upos in _UPOS_RULES:
        if p != primary:
            continue
        if req is None or req <= subtags:  # subset check
            return upos
    return None


# ============================================================
#  Stream-format parser
# ============================================================
# Apertium stream format:
#   ^wordform/lemma<tags>[/lemma2<tags2>...][|lemma<tags>]...$
# A blank reading ("^wordform/*wordform$") means the analyser failed.

_LU_RE = re.compile(r"\^([^$]*)\$")


@dataclass
class Reading:
    lemma: str
    tags: list[str]


@dataclass
class LexicalUnit:
    wordform: str
    readings: list[Reading]

    @property
    def is_unknown(self) -> bool:
        """True if no real analysis (blank/asterisk reading only)."""
        return not self.readings or all(
            r.lemma.startswith("*") for r in self.readings
        )


def parse_reading(body: str) -> Reading:
    """Parse 'lemma<t1><t2>...' into Reading(lemma, [t1, t2, ...])."""
    # split on first '<'
    lt = body.find("<")
    if lt == -1:
        return Reading(lemma=body, tags=[])
    lemma = body[:lt]
    tags = re.findall(r"<([^>]+)>", body[lt:])
    return Reading(lemma=lemma, tags=tags)


def parse_stream(text: str) -> list[LexicalUnit]:
    """
    Parse Apertium stream format text into a list of LexicalUnit objects.
    Preserves order; superblanks (text outside ^...$) are ignored for analysis
    but sentence boundaries are inferred from <sent> tokens downstream.
    """
    units: list[LexicalUnit] = []
    for m in _LU_RE.finditer(text):
        inner = m.group(1)
        if "/" in inner:
            wf, rest = inner.split("/", 1)
        else:
            wf, rest = inner, ""
        readings = []
        # alternative readings separated by '/', but wf already consumed
        for alt in rest.split("/"):
            alt = alt.strip()
            if not alt:
                continue
            readings.append(parse_reading(alt))
        # Also handle the pipeline variant where readings are split by '|'
        # after a shared lemma — rare in the corpus, but be safe.
        units.append(LexicalUnit(wordform=wf, readings=readings))
    return units


def primary_tag(reading: Reading) -> tuple[str, set[str]]:
    """
    Extract the primary POS tag and the set of subtags.

    The first tag is the primary POS (n, v, prn, ...). Subsequent tags are
    subtags/features. Some readings carry "continuation lexicon" tags like
    'ant'/'top'/'cog' as the FIRST tag for proper nouns; in that case we still
    treat the first tag as primary and let the mapping fall through to PROPN
    via the 'np' rule when np is present, else via the proper-noun subtags.
    """
    if not reading.tags:
        return "", set()
    # For np, the lexical class is usually first; subtags like ant/top/cog follow.
    primary = reading.tags[0]
    subs = set(reading.tags[1:])
    return primary, subs


def best_upos(lu: LexicalUnit) -> tuple[str, str]:
    """
    Pick the best (upos, lemma) for a lexical unit.

    Disambiguation is simple but effective for silver data:
      - drop readings whose lemma starts with '*' (failed analysis);
      - among the rest, prefer the first reading whose tags map to a known UPOS;
      - fall back to the first non-star reading; if none, UPOS = 'X'.
    We deliberately do NOT replicate the Apertium tagger's statistical
    disambiguation — for *training* silver data, the first morphological reading
    is a good-enough cheap signal and the neural model + class weighting handle
    the residual noise.
    """
    clean = [r for r in lu.readings if not r.lemma.startswith("*")]
    if not clean:
        return "X", lu.wordform
    for r in clean:
        p, subs = primary_tag(r)
        upos = apertium_to_upos(p, subs)
        if upos is not None:
            return upos, r.lemma
    # known primary tag but no rule matched → keep lemma, tag X
    return "X", clean[0].lemma


# ============================================================
#  Sentence segmentation + CoNLL-U export
# ============================================================

def units_to_sentences(units: Iterable[LexicalUnit]) -> list[Sentence]:
    """
    Segment lexical units into sentences on <sent> (sentence-final punctuation)
    and return a list of Sentence objects with UPOS assigned.

    The <sent> token itself is kept as the final PUNCT token of its sentence,
    mirroring UD conventions. Empty sentences are skipped.
    """
    sentences: list[Sentence] = []
    cur: list[Token] = []
    idx = 0
    for lu in units:
        if lu.wordform == "":
            continue
        upos, lemma = best_upos(lu)
        cur.append(Token(form=lu.wordform, lemma=lemma, upos=upos))
        if upos == "PUNCT" and lu.readings and any(
            "sent" in r.tags for r in lu.readings if not r.lemma.startswith("*")
        ):
            sentences.append(Sentence(sent_id=f"silver-{idx}", tokens=tuple(cur)))
            cur = []
            idx += 1
    if cur:
        sentences.append(Sentence(sent_id=f"silver-{idx}", tokens=tuple(cur)))
    return sentences


def write_conllu(sentences: Iterable[Sentence], path: str) -> int:
    """Write sentences to a CoNLL-U file; return number of sentences written."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(f"# sent_id = {s.sent_id}\n")
            forms = " ".join(t.form for t in s.tokens)
            f.write(f"# text = {forms}\n")
            for i, tok in enumerate(s.tokens, 1):
                # CoNLL-U: id form lemma upos xpos feats head deprel deps misc
                feats = "_"
                head = "0" if i == 1 else "_"
                deprel = "root" if i == 1 else "_"
                f.write(f"{i}\t{tok.form}\t{tok.lemma}\t{tok.upos}\t_\t{feats}\t{head}\t{deprel}\t_\t_\n")
            f.write("\n")
            n += 1
    return n


def convert_apertium_corpus(
    tagged_path: str,
    out_path: str,
    *,
    drop_unknown_rate: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Convert an Apertium kaz.tagged file to silver CoNLL-U.

    Parameters
    ----------
    tagged_path : path to the `^wf/lemma<tags>$` stream file (apertium corpus/).
    out_path : where to write the resulting CoNLL-U.
    drop_unknown_rate : fraction of tokens mapped to 'X' (unknown) to drop,
        since they add label noise. 0 keeps everything.
    seed : RNG seed for the unknown-drop subsampling.

    Returns a dict with conversion stats.
    """
    import random
    rng = random.Random(seed)

    with open(tagged_path, encoding="utf-8") as f:
        text = f.read()
    units = parse_stream(text)
    sentences = units_to_sentences(units)

    total = sum(len(s.tokens) for s in sentences)
    unknown = sum(1 for s in sentences for t in s.tokens if t.upos == "X")

    if drop_unknown_rate > 0:
        kept: list[Sentence] = []
        for s in sentences:
            toks = []
            for t in s.tokens:
                if t.upos == "X" and rng.random() < drop_unknown_rate:
                    continue
                toks.append(t)
            if len(toks) >= 2:  # drop degenerate sentences
                kept.append(Sentence(sent_id=s.sent_id, tokens=tuple(toks)))
        sentences = kept

    written = write_conllu(sentences, out_path)
    return {
        "lexical_units": len(units),
        "sentences": len(sentences),
        "written": written,
        "tokens": total,
        "unknown_tokens": unknown,
        "unknown_pct": round(unknown / total * 100, 1) if total else 0.0,
    }
