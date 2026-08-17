"""Corpus frequency dictionary for the frequency-ranking candidate mode.

kazmorph's suggestor has no frequency information — every lexicon is a plain
``set[str]``. This module derives a legitimate frequency dictionary from the
corpora that ship with the project, so the ``dl_freq`` / ``dl_freq_morph``
candidate modes (see :mod:`kazmorph.correction.candidate_generator`) are real,
reproducible comparisons rather than stubs.

The frequency source is the union of:
  * data/train_split.conllu     — gold UD Kazakh-KTB (normative forms)
  * data/silver_sentences.conllu — CG-disambiguated Kazakh literary texts
  * data/silver_apertium.conllu  — Apertium-analysed Kazakh text

Silver data is used ONLY for frequency estimation, never as a gold label.

Leakage note. The gold evaluation pool is the deduplicated union of every
non-silver ``.conllu`` in ``data/`` (:func:`kazmorph.corpus.load_pool`). Two of
the three frequency sources overlap that pool:
  * train_split.conllu shares ~85% of its sentences with the test/dev folds
    (it is an 85/15 stratified split of the *same* corpus the folds are drawn
    from; see the :mod:`kazmorph.corpus` docstring).
  * silver_apertium.conllu shares ~78% of its sentences with the test pool,
    because it was built by running Apertium over the same KTB texts.
Only silver_sentences.conllu is disjoint from the pool (independent literary
texts). A globally-built dictionary from the union therefore leaks the gold
forms of the very tokens being corrected into the frequency tie-break.

The leakage-free construction is per-fold: :func:`build_fold_frequency_dict`
takes the dev/test sentences of the current fold and excludes any source
sentence whose normalised token fingerprint matches one of them, from BOTH
gold and silver sources. This removes both leakages while preserving the bulk
of the frequency signal. The original global
:func:`build_frequency_dict` / :func:`get_frequency_dict` are retained for the
leaky-vs-clean delta measurement.

Adapted from the IHCONCS_2026_paper project (MIT): only the ``config`` data
paths were replaced with repo-relative resolution.
"""

from __future__ import annotations

import os
from collections import Counter

# Resolve the three canonical frequency sources relative to the package, so
# the module works regardless of the caller's working directory. The package
# lives at <repo>/kazmorph/correction/freq.py → data/ is two levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
TRAIN_CONLLU = os.path.join(_DATA_DIR, "train_split.conllu")
SILVER_SENTENCES_CONLLU = os.path.join(_DATA_DIR, "silver_sentences.conllu")
SILVER_APERTIUM_CONLLU = os.path.join(_DATA_DIR, "silver_apertium.conllu")

# Default source list for both the global and per-fold builders.
_DEFAULT_SOURCES = [TRAIN_CONLLU, SILVER_SENTENCES_CONLLU, SILVER_APERTIUM_CONLLU]


def _count_forms(conllu_path: str, counter: Counter) -> int:
    """Add lowercased surface-form counts from one CoNLL-U file.

    Returns the number of tokens counted. Skips multiword/empty rows
    (id contains '-' or '.'), comment lines, and blank lines. Lowercasing
    matches the suggestor's lookup convention.
    """
    n = 0
    if not os.path.exists(conllu_path):
        return 0
    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            tid = parts[0]
            if "-" in tid or "." in tid:
                continue
            form = parts[1].strip()
            if form:
                counter[form.lower()] += 1
                n += 1
    return n


def build_frequency_dict(source_paths: list[str] | None = None) -> dict[str, int]:
    """Build the corpus-frequency dictionary used by the DL+freq baseline.

    Combines gold train + both silver corpora (by default). Returns
    {lowercased_form: count}. The dict is cached on first call via
    :func:`get_frequency_dict`; this raw builder runs every time so a caller
    can pass a restricted ``source_paths`` list.
    """
    paths = source_paths if source_paths is not None else _DEFAULT_SOURCES
    counter: Counter = Counter()
    counts = []
    for path in paths:
        counts.append(_count_forms(path, counter))
    freq = dict(counter)
    # Stash provenance on the function object for logging (dict subclasses are
    # heavier; we attach metadata as a function attribute instead).
    build_frequency_dict.last_counts = {
        "sources": list(paths),
        "tokens_per_source": counts,
        "total_tokens": sum(counts),
        "unique_forms": len(freq),
    }
    return freq


def frequency_of(form: str, freq: dict[str, int]) -> int:
    """Return corpus frequency of `form` (case-insensitive), 0 if unseen.

    Used by candidate_generator's DL+freq ranking mode and by the energy
    function's λ_f·log(1+ν_c) term. The +1 floor in the log keeps unseen
    candidates from receiving -inf energy.
    """
    return freq.get(form.lower(), 0)


def _fingerprint_from_lines(lines) -> tuple[str, ...] | None:
    """Lowercased token-sequence fingerprint of one CoNLL-U sentence block.

    Mirrors :func:`kazmorph.corpus.Sentence.fingerprint` so exclusion matches
    the pool's own dedup key. Returns None for an empty/non-sentence block.
    """
    forms: list[str] = []
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        tid = parts[0]
        if "-" in tid or "." in tid:
            continue
        form = parts[1].strip()
        if form:
            forms.append(form.lower())
    return tuple(forms) if forms else None


def _iter_sentence_blocks(path: str):
    """Yield (fingerprint, lines) pairs, one per sentence in a CoNLL-U file."""
    if not os.path.exists(path):
        return
    block: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                fp = _fingerprint_from_lines(block)
                if fp is not None:
                    yield fp, block
                block = []
            else:
                block.append(line)
    fp = _fingerprint_from_lines(block)
    if fp is not None:
        yield fp, block


def build_fold_frequency_dict(
    excluded_sentences, *, source_paths: list[str] | None = None,
) -> dict[str, int]:
    """Leakage-free per-fold frequency dictionary.

    Counts lowercased surface forms from the union of the gold train split and
    both silver corpora (the same sources as :func:`build_frequency_dict`),
    but EXCLUDES any source sentence whose normalised fingerprint matches a
    sentence in ``excluded_sentences`` — the dev/test half of the current
    cross-validation fold. This removes the gold-form leakage from
    ``train_split.conllu`` (~85% pool overlap) and the text-overlap leakage
    from ``silver_apertium.conllu`` (~78% pool overlap), leaving only frequency
    signal from sentences the fold never evaluates on.

    ``excluded_sentences`` is an iterable of :class:`kazmorph.corpus.Sentence`
    (or any object exposing a ``.fingerprint()`` method / lowercased-forms
    tuple). ``source_paths`` defaults to the three canonical sources; callers
    may pass a restricted list (e.g. silver-only) to measure the leakage delta.
    """
    excluded_fps: set[tuple[str, ...]] = set()
    for s in excluded_sentences:
        fp = s.fingerprint() if hasattr(s, "fingerprint") else tuple(
            t.form.lower() for t in s.tokens
        )
        if fp:
            excluded_fps.add(fp)

    if source_paths is None:
        source_paths = _DEFAULT_SOURCES

    counter: Counter = Counter()
    n_excluded = 0
    n_kept = 0
    for path in source_paths:
        for fp, block in _iter_sentence_blocks(path):
            if fp in excluded_fps:
                n_excluded += 1
                continue
            # Count this sentence's surface forms.
            for line in block:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4 or "-" in parts[0] or "." in parts[0]:
                    continue
                form = parts[1].strip()
                if form:
                    counter[form.lower()] += 1
            n_kept += 1

    freq = dict(counter)
    build_fold_frequency_dict.last_counts = {
        "n_source_sentences_kept": n_kept,
        "n_source_sentences_excluded": n_excluded,
        "unique_forms": len(freq),
    }
    return freq


# Module-level cache so repeated imports don't re-scan the corpora.
_FREQ_CACHE: dict[str, int] | None = None


def get_frequency_dict() -> dict[str, int]:
    """Return the cached frequency dict, building it on first access."""
    global _FREQ_CACHE
    if _FREQ_CACHE is None:
        _FREQ_CACHE = build_frequency_dict()
    return _FREQ_CACHE
