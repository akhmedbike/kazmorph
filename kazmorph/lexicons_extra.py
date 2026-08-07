"""
Hand-curated lexicon supplements for KazMorph.

`kazmorph/lexicons.py` is auto-generated from the Apertium-kaz lexc
dictionary by `scripts/import_apertium.py`. Anything written directly into
that file is lost on the next regeneration, so manual corrections and
additions live here instead. The generated module imports these sets and
folds them into its own, so callers see a single merged lexicon.

Two kinds of supplements:

  - EXTRA_NOUNS / EXTRA_VERBS / EXTRA_ADJ: frequent lemmas that the Apertium
    source is missing (country/city names that the analyzer currently tags
    UNK with confidence 0). See REVIEW_RECOMMENDATIONS.md problem #2.

  - NATIVE_LEMMAS: a whitelist of native Turkic words that Apertium's
    heuristic borrowing detector wrongly includes in BORROWINGS (e.g.
    жеке matched via the "-ке" suffix). `is_borrowing()` checks this set
    and refuses to label its members as borrowings. See REVIEW_RECOMMENDATIONS.md
    problem #4.

To regenerate lexicons.py after editing this file, run:
    PYTHONPATH=. .venv/bin/python scripts/import_apertium.py
"""

# ------------------------------------------------------------
# 1. Missing frequent lemmas (problem #2)
# ------------------------------------------------------------
# These are handled correctly by the neural POS tagger once it sees them,
# but the RMA lexicon gap leaves them at UNK/confidence 0 when the neural
# model is not loaded (unit-test mode, API cold start on unseen tokens).
EXTRA_NOUNS: frozenset[str] = frozenset({
    # Country names that currently tag as UNK (e.g. бразилия → conf 0.0).
    # Lowercase dictionary forms; the analyzer/lemmatizer restores case.
    "бразилия", "аргентина", "мексика", "канада", "ауғанстан",
    "өзбекстан", "түрікія", "германия", "франция", "ағылшын",
})

EXTRA_VERBS: frozenset[str] = frozenset()

EXTRA_ADJ: frozenset[str] = frozenset()


# ------------------------------------------------------------
# 2. Native Turkic whitelist (problem #4)
# ------------------------------------------------------------
# See module docstring. Each entry has well-attested Old Turkic / Common
# Turkic etymology and is wrongly flagged as BOR by the suffix-pattern
# heuristic in import_apertium.py::detect_borrowings.
NATIVE_LEMMAS: frozenset[str] = frozenset({
    "арыстан",   # lion — Common Turkic *arslan
    "аспан",     # sky — Old Turkic
    "ешкі",      # goat — Common Turkic *äškäk
    "жеке",      # personal/private — Common Turkic
    "жұрт",      # people/place — Common Turkic *jurt
    "кемпір",    # old woman — Common Turkic
    "көке",      # father (respectful) / blue (variant) — Turkic
    "мұрт",      # mustache — Common Turkic *murt
    "орман",     # forest — Common Turkic
    "теке",      # billy-goat — Common Turkic *teke
    "қарт",      # old man — Common Turkic
    "үлкен",     # big — Common Turkic
    "ескі",      # old — Common Turkic
    "жамбас",    # hip — Common Turkic
    "орда",      # horde/palace — Turkic (via Mongolic cognate)
})
