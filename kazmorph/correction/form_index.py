"""Corpus-derived word-FORM index for candidate generation.

The default SymSpell index (:func:`kazmorph.suggestor._get_symindex`) covers
~22k *lemmas*. In an agglutinative language most running-text tokens are
inflected, so the gold form of a corrupted token is frequently absent from
the lemma index — reachable only through stem recombination. This module
adds a second index over *surface forms* harvested from the gold train split
and both silver corpora.

Leakage control (mirrors :mod:`kazmorph.correction.freq`): sentences whose
fingerprint matches an excluded file (dev / test) are skipped, so evaluation
on corrupted dev never sees the exact gold forms it is scored against. In
serving, pass no exclusions.

Caveat, kept honest: the IHCONCS experiments measured that a leakage-free
corpus-form index alone adds ≈0 candidate-reach lift, because per-suffix-
combination surface forms are near-unique in Kazakh. The index is therefore
used here primarily as a *validity oracle* for script-confusion variants
(и↔і type errors produce a form that is either a real corpus form or noise)
and only secondarily as an extra retrieval channel; the eval harness
(``scripts/eval_correction.py``) reports the actual contribution.

Adapted for kazmorph from the IHCONCS_2026_paper project (MIT).
"""

from __future__ import annotations

import os
import threading

from kazmorph.correction.freq import (
    _DEFAULT_SOURCES, _iter_sentence_blocks,
)
from kazmorph.suggestor import _SymIndex

# ── Lazy singletons ──────────────────────────────────────────────────────────
# RLock (reentrant): get_form_index() holds the lock while calling get_forms(),
# which acquires the same lock — a plain Lock self-deadlocks on the first
# get_form_index() call (observed as a hung /correct3 request).
_FORMS: set[str] | None = None
_FORM_INDEX: _SymIndex | None = None
_LOCK = threading.RLock()


def build_form_lexicon(
    source_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> set[str]:
    """Collect lowercased word forms from CoNLL-U sources, minus excluded sentences.

    ``exclude_paths`` files are read first and their sentence fingerprints
    form a block-list; any source sentence matching a fingerprint is skipped.
    Default exclusions: ``data/dev.conllu`` and ``data/test.conllu`` (the
    evaluation splits). Pass ``[]`` to disable (pure serving mode).
    """
    if source_paths is None:
        source_paths = list(_DEFAULT_SOURCES)
    if exclude_paths is None:
        _data = os.path.dirname(source_paths[0]) if source_paths else "data"
        exclude_paths = [os.path.join(_data, "dev.conllu"),
                         os.path.join(_data, "test.conllu")]
    excluded: set[tuple[str, ...]] = set()
    for path in exclude_paths:
        for fp, _lines in _iter_sentence_blocks(path):
            excluded.add(fp)

    forms: set[str] = set()
    for path in source_paths:
        for fp, lines in _iter_sentence_blocks(path):
            if fp in excluded:
                continue
            for line in lines:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                tid, form = parts[0], parts[1].strip()
                # Skip multiword-range rows (3-4) and empty nodes (3.1):
                # their FORM is a tokenisation artifact, not a lookup target.
                if "-" in tid or "." in tid:
                    continue
                fl = form.lower()
                # Alphabetic forms only: punctuation/numbers are never
                # correction candidates and would bloat the index.
                if fl and any(ch.isalpha() for ch in fl):
                    forms.add(fl)
    return forms


def get_forms() -> set[str]:
    """Process-wide singleton form set (leakage-free defaults)."""
    global _FORMS
    if _FORMS is None:
        with _LOCK:
            if _FORMS is None:
                _FORMS = build_form_lexicon()
    return _FORMS


def get_form_index() -> _SymIndex:
    """Process-wide singleton SymSpell index over corpus forms (max_dist=2)."""
    global _FORM_INDEX
    if _FORM_INDEX is None:
        with _LOCK:
            if _FORM_INDEX is None:
                _FORM_INDEX = _SymIndex(get_forms(), max_dist=2)
    return _FORM_INDEX


def form_exists(form: str) -> bool:
    """Membership oracle: is this lowercased surface form a real corpus form?"""
    return form.lower() in get_forms()
