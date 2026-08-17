"""Candidate generation for correction, in four ranking modes.

This module exposes a single :func:`generate_candidates` that returns a ranked
list of (candidate, energy) pairs, where ``energy`` is comparable within a mode
(lower energy = better). Four modes are implemented, all sharing the same
SymSpell retrieval (:func:`kazmorph.suggestor._get_symindex`):

  * ``"dl"``            — pure Damerau-Levenshtein retrieval, ranked by
                          (distance, alphabetical). No frequency, no morphology.

  * ``"dl_freq"``       — same retrieval, re-ranked by (distance, -frequency,
                          alphabetical). The corpus-frequency tie-break.

  * ``"dl_fst_filter"`` — retrieval, then drop any candidate whose reverse
                          morphological analysis fails or whose compatibility
                          coefficient κ < ``KAPPA_MIN``. The analysis-side
                          approximation of analyze–synthesize–reverify.

  * ``"dl_freq_morph"`` — dl_freq plus a stem-recombination channel: the token
                          is morphologically decomposed (RMA), near-candidates
                          for the *stem* are retrieved, and the suffix is
                          re-attached (with harmony alternates) to synthesise
                          surface candidates the edit-distance index cannot
                          reach. Empirical lift: candidate reachability +11pp.

Every mode returns at most ``MAX_CANDIDATES`` pairs. The candidate list never
includes the input token itself — the caller adds the "keep original" option
separately so token preservation is a first-class decision.

Leakage note: a naive "grow the index with corpus surface forms" path was
evaluated and rejected — the frequency dictionary is built from the train
split, so its inflected forms ARE the gold test-fold forms, and including
them makes candidate recall trivially high but circular. The leakage-free
augmented index (test-fold forms excluded) gives ~0 lift, because Kazakh
agglutinative morphology produces near-unique surface forms per suffix
combination. Stem recombination is the only leakage-free way to recover
inflected forms whose stem is in the lexicon.

Adapted from the IHCONCS_2026_paper project (MIT): only the ``config`` imports
were replaced with the module-level constants below.
"""

from __future__ import annotations

import math

from kazmorph.suggestor import _get_symindex
from kazmorph.rma import ReverseMorphAnalyzer
from kazmorph.mfrn import MFRN
from kazmorph.correction.freq import get_frequency_dict, frequency_of

# ── Tunable constants (were ``config.*`` in the source project) ──────────────
# Kept as module-level names for discoverability and so sweeps can import them.
MAX_EDIT_DISTANCE = 2      # SymSpell retrieval radius
MAX_CANDIDATES = 10        # truncate the ranked list to this many
KAPPA_MIN = 0.40           # morphological-validity gate for dl_fst_filter/morph
LAMBDA_ED = 1.0            # weight on normalised edit distance in the energy
LAMBDA_MORPH = 0.5         # weight on (1 - kappa) morphological uncertainty
LAMBDA_FREQ = 0.3          # weight on the frequency presence bonus
STEM_PENALTY = 0.15        # extra energy for stem-recombined candidates
STEM_MAX_CANDIDATES = 8    # cap on near-stems recombined per token


# Module-level singletons for the FST-filter mode. ReverseMorphAnalyzer loads
# the UD lemma dict on construction; MFRN is stateless beyond its rule table.
# Both are safe to share across folds and threads (read-only after __init__).
_RMA: ReverseMorphAnalyzer | None = None
_MFRN: MFRN | None = None
_FREQ: dict[str, int] | None = None


def _get_rma() -> ReverseMorphAnalyzer:
    global _RMA
    if _RMA is None:
        _RMA = ReverseMorphAnalyzer()
    return _RMA


def _get_mfrn() -> MFRN:
    global _MFRN
    if _MFRN is None:
        _MFRN = MFRN(scoring=False)
    return _MFRN


def _get_freq() -> dict[str, int]:
    global _FREQ
    if _FREQ is None:
        _FREQ = get_frequency_dict()
    return _FREQ


def _symspell_retrieve(token: str, idx=None) -> list[tuple[int, str]]:
    """Raw SymSpell retrieval: [(distance, word)] within MAX_EDIT_DISTANCE.

    Lowercases the token to match the index convention. The result is NOT
    sorted; each mode applies its own sort key. ``idx`` defaults to the
    process-wide singleton; a caller may pass a fold-specific index built with
    the test-fold forms excluded (leakage guard).
    """
    if idx is None:
        idx = _get_symindex()
    return idx.query(token.lower())


def _fst_valid(candidate: str) -> bool:
    """Analysis-side morphological validity check (the "re-analysis" proxy).

    A candidate is valid iff ReverseMorphAnalyzer recovers a non-UNK analysis
    for it AND the MFRN compatibility coefficient κ ≥ ``KAPPA_MIN``. This is an
    approximation of analyze(c) ⊇ (ℓ′, f′): without a full synthesiser we
    check that the candidate at least parses into a known stem with admissible
    phonology.
    """
    rma = _get_rma()
    mfrn = _get_mfrn()
    res = rma.analyze(candidate)
    # UNK means no stem and no suffixes recovered → structurally inadmissible.
    if res.pos == "UNK" and not res.suffixes:
        return False
    kappa = mfrn.compute_kappa(res)
    return kappa >= KAPPA_MIN


# ── Stem-expansion channel (dl_freq_morph mode) ──────────────────────────────
def _suffix_variants(suffixes: list[str]) -> list[list[str]]:
    """Return harmony-alternate suffix sequences for re-synthesis.

    Kazakh vowel harmony selects suffix allomorphs by stem backness. A
    corrupted token may carry the wrong allomorph (the harmony-violation
    operator) or an analysis of the corrupted form may have picked the
    back/front variant that does not match the gold. To recover the gold via
    stem recombination we therefore generate, alongside the suffixes observed
    on the corrupted token, their harmony counterparts.

    The mapping covers the productive vowel-alternating suffix initials:
    back ↔ front pairs (а/е, ы/і, о/ө, ұ/ү). This is intentionally a small,
    linguistically-grounded table rather than a full FST — it targets exactly
    the allomorph mismatches that the controlled-error operators introduce.
    """
    _HARMONY = {"а": "е", "ы": "і", "о": "ө", "ұ": "ү",
                "е": "а", "і": "ы", "ө": "о", "ү": "ұ"}
    variants = [list(suffixes)]
    for i, suf in enumerate(suffixes):
        if suf and suf[0] in _HARMONY:
            alt = [_HARMONY[suf[0]] + suf[1:]]
            variants.append(list(suffixes[:i]) + alt + list(suffixes[i + 1:]))
    return variants


def _stem_expand(token: str, idx) -> list[tuple[str, int]]:
    """Stem-recombination retrieval: return [(candidate, edit_distance)].

    Decomposes `token` into stem + suffixes via RMA, retrieves near-candidates
    for the stem in `idx`, and re-synthesises surface candidates by
    re-attaching the suffixes (and their harmony alternates). The returned
    edit distance is that of the *stem* match, since the suffix is carried
    over verbatim — this is the key difference from direct retrieval, which
    would require the full inflected form to be in the index.

    Returns [] when the token cannot be decomposed (RMA returns UNK or no
    suffixes) or when no stem candidate is found. The caller applies
    ``STEM_PENALTY`` to the energy of these candidates so they rank below
    same-distance direct hits.
    """
    rma = _get_rma()
    res = rma.analyze(token)
    # Need a real decomposition: non-UNK, with suffixes, and a stem shorter
    # than the full token (otherwise there is nothing to recombine).
    if res.pos == "UNK" or not res.suffixes or res.lemma == token:
        return []
    stem = res.lemma
    stem_hits = idx.query(stem.lower())
    if not stem_hits:
        return []
    # Re-synthesise: for each near-stem, attach each suffix variant.
    variants = _suffix_variants(res.suffixes)
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    t_lower = token.lower()
    for dist, stem_cand in stem_hits[:STEM_MAX_CANDIDATES]:
        for suf_seq in variants:
            surface = stem_cand + "".join(suf_seq)
            if surface == t_lower or surface in seen:
                continue
            seen.add(surface)
            out.append((surface, dist))
    return out


def generate_candidates(
    token: str,
    mode: str = "dl",
    max_candidates: int | None = None,
    *,
    index=None,
    freq: dict[str, int] | None = None,
) -> list[tuple[str, float]]:
    """Return ranked [(candidate, energy)] for `token` under the given mode.

    Energy is the ranking key (lower energy = better). The list is sorted
    ascending by energy and truncated to `max_candidates` (default
    ``MAX_CANDIDATES``). The input token itself is excluded.

    ``mode`` ∈ {"dl", "dl_freq", "dl_fst_filter", "dl_freq_morph"}.

    ``index`` optionally supplies a SymSpell index (e.g. a fold-specific one
    with test-fold forms excluded); it defaults to the global singleton.

    ``freq`` optionally supplies a frequency dictionary (e.g. a leakage-free
    per-fold dict built by
    :func:`kazmorph.correction.freq.build_fold_frequency_dict`); it defaults to
    the global singleton. Passing a per-fold dict removes the gold-form
    leakage that the global dict carries into the dl_freq / dl_freq_morph
    ranking.
    """
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES

    idx = index if index is not None else _get_symindex()
    raw = _symspell_retrieve(token, idx)
    freq = freq if freq is not None else _get_freq()

    # First pass: build (candidate, distance, freq) for surviving candidates.
    survivors: list[tuple[str, int, int, bool]] = []  # (word, dist, freq, is_stem)
    seen: set[str] = set()
    t_lower = token.lower()
    for dist, word in raw:
        if word == t_lower or word in seen:
            continue
        # dl_fst_filter / dl_freq_morph apply the morphological validity gate
        # here, before any scoring. Candidates that fail the gate are removed
        # entirely ("Morphologically invalid candidates are removed before
        # ranking rather than receiving only a soft penalty").
        if mode in ("dl_fst_filter", "dl_freq_morph") and not _fst_valid(word):
            continue
        seen.add(word)
        survivors.append((word, dist, frequency_of(word, freq), False))

    # Stem-expansion channel: only for dl_freq_morph. Produces additional
    # surface candidates by recombining near-stems with the token's suffixes.
    # These are tagged is_stem=True so the scorer applies STEM_PENALTY.
    if mode == "dl_freq_morph":
        for word, dist in _stem_expand(token, idx):
            if word == t_lower or word in seen:
                continue
            # Apply the same morphological validity gate to recombined forms.
            if not _fst_valid(word):
                continue
            seen.add(word)
            survivors.append((word, dist, frequency_of(word, freq), True))

    if not survivors:
        return []

    # Score every survivor with the mode-appropriate energy.
    scored: list[tuple[str, float]] = []
    for word, dist, f, is_stem in survivors:
        # The morphological-uncertainty term Ω_morph(c) is (1 - κ); for
        # dl / dl_freq modes (no MFRN pass) it is constant across candidates
        # and drops out of the argmin, so we only compute it for
        # dl_fst_filter / dl_freq_morph where κ varies.
        if mode in ("dl_fst_filter", "dl_freq_morph"):
            kappa = _get_mfrn().compute_kappa(_get_rma().analyze(word))
            omega_morph = 1.0 - kappa
        else:
            omega_morph = 0.0
        normalised_dist = dist / MAX_EDIT_DISTANCE
        energy = (
            LAMBDA_ED * normalised_dist
            + LAMBDA_MORPH * omega_morph
            - LAMBDA_FREQ * (1.0 if f > 0 else 0.0)  # binary presence bonus
        )
        # Frequency as a fine-grained tie-break: higher freq → lower energy.
        # We add a small log term so two same-distance candidates are ordered
        # by corpus frequency, realising the dl_freq ranking.
        if mode in ("dl_freq", "dl_freq_morph"):
            energy -= 0.1 * math.log(1 + f)
        # Stem-recombined candidates carry an extra penalty: their distance is
        # measured on the stem only, so they are less reliable than a direct
        # full-form edit-distance hit at the same band.
        if is_stem:
            energy += STEM_PENALTY
        scored.append((word, energy))

    scored.sort(key=lambda x: (x[1], x[0]))
    return scored[:max_candidates]


def candidate_probs(
    candidates: list[tuple[str, float]],
    include_original: bool = True,
    original_token: str = "",
) -> list[tuple[str, float]]:
    """Normalised softmax probabilities over candidates.

    P(c | X, i) = exp(-E(c)) / Σ exp(-E(c′)), with the original token added
    to the decision set as its own zero-energy option when
    ``include_original`` is set. Returns [(candidate, probability)] sorted by
    descending probability.

    The original-token-as-candidate is what makes KEEP a first-class model
    decision rather than a fallback: if no correction has lower energy, KEEP
    wins on the softmax. This is intentionally not a bonus — we do not want
    to bias toward inaction beyond what the evidence supports.
    """
    items = list(candidates)
    if include_original and original_token:
        items = [(original_token, 0.0)] + items
    if not items:
        return []
    # Softmax over -E. Subtract max for numerical stability.
    neg_e = [-e for _, e in items]
    m = max(neg_e)
    exps = [math.exp(ne - m) for ne in neg_e]
    total = sum(exps)
    probs = [e / total for e in exps]
    paired = [(word, p) for (word, _), p in zip(items, probs)]
    paired.sort(key=lambda x: -x[1])
    return paired
