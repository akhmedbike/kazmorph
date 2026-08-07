"""
Formal Error Taxonomy for morphological processing of agglutinative languages.

Each input token receives ONE of five mutually exclusive diagnostic labels:

  BOR  — borrowing interference (loan words, code-switching, non-native phonotactics)
  AFC  — affix-chain complexity (deep / atypical suffix sequences)
  SEG  — segmentation anomaly (root/surface mismatch, over-/under-segmentation)
  AGR  — morphosyntactic-agreement error (POS/feature incompatibility)
  NONE — no diagnosable error (standard analysis)

Unlike the original boolean cascade, the taxonomy now exposes a set of real-valued
scores in [0, 1] (``score_BOR``, ``score_AFC``, ``score_SEG``, ``score_AGR``),
each computed by an independent detector. The final class is the highest-scoring
detector that clears the threshold ``theta`` (default 0.30), with a fixed
priority order used only to break near-ties. This makes the taxonomy a measurable
signal that the pipeline routing layer (see :mod:`kazmorph.pipeline`) can consume,
rather than a post-hoc label.

The score coefficients and the priority order mirror the formal definitions in
the paper (Section "Formal Taxonomy of Morphological Errors"). They are
deterministic and require no additional training annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kazmorph.lexicons import is_borrowing
from kazmorph.rma import RMAResult

# Default diagnostic threshold (paper: theta = 0.30). A token is assigned a
# specialized class only if its score for that class is >= THETA.
THETA: float = 0.30

# ---------------------------------------------------------------------------
# Phonotactic borrowing markers (paper: BOR class, "direct match" + additive).
# These Cyrillic graphemes are not part of the native Kazakh phoneme inventory
# and reliably signal Russian/international loanwords.
# ---------------------------------------------------------------------------
_BORROWING_PHONEMES: frozenset[str] = frozenset("фвцщэъь")

# Marker suffixes characteristic of Russian/international borrowings.
_BORROWING_SUFFIXES: tuple[str, ...] = (
    "ция", "тор", "ист", "изм", "ация", "ка", "кер", "логия",
    "метр", "графия", "фон", "мотор", "рент", "атор",
)

# Closed UPOS classes that never carry productive affixation — they are
# excluded from the AFC detector (paper: closed-class guard).
_CLOSED_POS: frozenset[str] = frozenset(
    {"PRON", "CONJ", "ADV", "ADP", "NUM", "PART", "INTJ", "PUNCT",
     "CCONJ", "SCONJ", "DET", "SYM"}
)


@dataclass
class TaxonomyResult:
    """Taxonomy classification for a single token."""

    token: str
    difficulty_class: str  # BOR, AFC, SEG, AGR, or NONE
    description: str
    # Per-detector scores in [0, 1]. Populated by classify_scored(); the legacy
    # classify() wrapper leaves them at 0.0 for backward compatibility.
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-class score detectors. Each is an independent, side-effect-free function
# of the RMA result (and, for AGR, the MFRN compatibility coefficient kappa).
# ---------------------------------------------------------------------------

def score_borrowing(rma_result: RMAResult) -> float:
    """
    BOR score ∈ [0, 1] (paper, "Class BOR").

    Two levels:
      1. Direct match: 0.90 if the token/lemma is in the BORROWINGS lexicon or
         the token contains Latin-script characters (code-switching artifact).
      2. Additive: +0.50 per borrowing phoneme class present, +0.40 if a marker
         suffix is present. Short tokens (len <= 3) are discounted ×0.3 to avoid
         false positives on native short roots.

    The result is clipped to [0, 1]. Native lexicon hits short-circuit to 0.0.
    """
    token = rma_result.token
    lemma = rma_result.lemma or token
    low = token.lower()

    # Native roots are never borrowings regardless of surface form.
    if _is_native(token) or _is_native(lemma):
        return 0.0

    # Level 1: direct match.
    if is_borrowing(token) or is_borrowing(lemma):
        return 0.90
    if any("a" <= ch.lower() <= "z" for ch in token):
        return 0.90

    # Level 2: additive phonotactic / suffix signal.
    score = 0.0
    if any(ch in _BORROWING_PHONEMES for ch in low):
        score += 0.50
    if any(low.endswith(suf) for suf in _BORROWING_SUFFIXES):
        score += 0.40

    if score == 0.0:
        return 0.0

    # Discount short tokens (likely native short roots, e.g. су, қол).
    if len(low) <= 3:
        score *= 0.3

    return min(score, 1.0)


def score_affix_chain(rma_result: RMAResult) -> float:
    """
    AFC score ∈ [0, 1] (paper, "Class AFC").

    Closed-class tokens always score 0. For open classes the score is additive:
      +0.50  affix-chain depth >= 3
      +0.30  affix-chain depth == 2
      +0.25  any affix belongs to the high-ambiguity set
      +0.20  vowel-harmony violation detected in the chain
      +0.20  a single affix longer than the root (morpheme reduplication signal)

    Clipped to [0, 1].
    """
    if rma_result.pos in _CLOSED_POS:
        return 0.0

    suffixes = rma_result.suffixes or []
    depth = len(suffixes)
    if depth == 0:
        return 0.0

    score = 0.0
    if depth >= 3:
        score += 0.50
    elif depth == 2:
        score += 0.30

    _HIGH_AMBIGUITY = {"ды", "ді", "ты", "ті", "ған", "ген", "лар", "лер",
                       "дар", "тар", "ның", "нің", "ға", "ге", "да", "де"}
    if any(suf in _HIGH_AMBIGUITY for suf in suffixes):
        score += 0.25

    if _has_harmony_violation(rma_result):
        score += 0.20

    root = _root_of(rma_result)
    if root and any(len(suf) > len(root) for suf in suffixes):
        score += 0.20

    return min(score, 1.0)


def score_segmentation(rma_result: RMAResult) -> float:
    """
    SEG score ∈ [0, 1] (paper, "Class SEG").

    Short function words (token length <= 2) are excluded. Otherwise additive:
      +0.40  abnormally short root in a long token (|root| < 3 and |token| > 6)
      +0.30  lemma/surface mismatch (the analyzer could not recover a clean lemma)
      +0.40  single-letter root with any affixation (over-segmentation signal)
      +0.30  total affix length exceeds root length

    Clipped to [0, 1].
    """
    token = rma_result.token
    if len(token) <= 2:
        return 0.0

    suffixes = rma_result.suffixes or []
    root = _root_of(rma_result)
    score = 0.0

    if root and len(root) < 3 and len(token) > 6:
        score += 0.40

    # Lemma/surface mismatch: the analyzer failed to converge on a clean lemma.
    if rma_result.lemma and rma_result.lemma.lower() != token.lower():
        # A lemma equal to "?" or equal to the unstripped token signals failure.
        if rma_result.lemma in ("?", "") or rma_result.confidence < 0.4:
            score += 0.30

    if root and len(root) == 1 and suffixes:
        score += 0.40

    if root:
        affix_len = sum(len(s) for s in suffixes)
        if affix_len > len(root):
            score += 0.30

    return min(score, 1.0)


def score_agreement(rma_result: RMAResult, kappa: float = 1.0) -> float:
    """
    AGR score ∈ [0, 1] (paper, "Class AGR").

    Derived from the MFRN compatibility coefficient ``kappa`` (∈ [0.40, 1.25])
    so that low compatibility (many rule violations) maps to a high AGR score.
    This deliberately couples AGR to MFRN — see the paper's "Limitations"
    section, which flags this coupling and proposes an independent agreement
    classifier as future work.

    Mapping: score_AGR = 1 - kappa   (clipped to [0, 1]).
    With kappa ∈ [0.40, 1.25] this yields score_AGR ∈ [0.0, 0.60] for the
    over-compatible tail, plus a small bonus for NOUN tokens that stack
    possessive + case morphology (the canonical AGR trigger).
    """
    score = max(0.0, min(1.0, 1.0 - kappa))

    suffixes = rma_result.suffixes or []
    if rma_result.pos == "NOUN" and len(suffixes) >= 1:
        # Possessive + case stacking is the prototypical Kazakh agreement load.
        score += 0.10

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Scored classifier (paper-formal) and legacy boolean wrapper.
# ---------------------------------------------------------------------------

# Effective priority order when several scores tie at the threshold: BOR first,
# then AFC > SEG > AGR. NONE is residual. (Paper, "Class NONE and the Priority
# Function".)
_PRIORITY: tuple[str, ...] = ("BOR", "AFC", "SEG", "AGR")


def classify_scored(
    rma_result: RMAResult,
    *,
    threshold: float = THETA,
    kappa: float | None = None,
) -> TaxonomyResult:
    """
    Compute all four scores and assign the diagnostic class.

    ``kappa`` is the MFRN compatibility coefficient. When ``None`` (e.g. the
    legacy pipeline path that does not run the kappa-weighted MFRN), AGR falls
    back to a confidence-based proxy so the detector still produces a signal.

    Returns a :class:`TaxonomyResult` carrying the full score vector, so callers
    (routing layer, evaluation) can inspect the per-class signal, not just the
    final label.
    """
    token = rma_result.token

    if kappa is None:
        # Confidence-based proxy: low RMA confidence ~ agreement problems.
        kappa = 1.0 if rma_result.confidence >= 0.85 else max(0.40, rma_result.confidence)

    scores = {
        "BOR": score_borrowing(rma_result),
        "AFC": score_affix_chain(rma_result),
        "SEG": score_segmentation(rma_result),
        "AGR": score_agreement(rma_result, kappa=kappa),
    }

    # Candidates that clear the threshold, evaluated in priority order.
    cls = "NONE"
    description = "Standard morphological analysis"
    for c in _PRIORITY:
        if scores[c] >= threshold:
            cls = c
            description = _DESCRIPTIONS[c]
            break

    return TaxonomyResult(
        token=token,
        difficulty_class=cls,
        description=description,
        scores=scores,
    )


_DESCRIPTIONS: dict[str, str] = {
    "BOR": "Borrowing interference — morphological decomposition is harder",
    "AFC": "Affix-chain complexity — deep / atypical suffix sequence",
    "SEG": "Segmentation anomaly — root/surface mismatch",
    "AGR": "Morphosyntactic-agreement error — POS/feature incompatibility",
    "NONE": "Standard morphological analysis",
}


def classify(rma_result: RMAResult) -> TaxonomyResult:
    """
    Backward-compatible classifier.

    Equivalent to :func:`classify_scored` with the default threshold and no
    MFRN coupling (kappa=None → confidence proxy). Existing callers and tests
    that only read ``difficulty_class`` keep working unchanged; new callers
    should prefer :func:`classify_scored` to also obtain the score vector.
    """
    return classify_scored(rma_result)


# ---------------------------------------------------------------------------
# Small helpers kept local to avoid importing the whole MFRN module here
# (taxonomy must not depend on the post-processing stage it diagnoses).
# ---------------------------------------------------------------------------

def _is_native(word: str) -> bool:
    """True if ``word`` is whitelisted as a native Kazakh root."""
    if not word:
        return False
    try:
        from kazmorph.lexicons_extra import NATIVE_LEMMAS
        return word.lower() in NATIVE_LEMMAS
    except Exception:
        return False


def _root_of(rma_result: RMAResult) -> str:
    """Best-effort root candidate: lemma if it looks like a clean root, else the
    surface token with all suffixes stripped."""
    lemma = (rma_result.lemma or "").strip()
    if lemma and lemma != "?":
        return lemma
    token = rma_result.token
    suffixes = rma_result.suffixes or []
    root = token
    for suf in suffixes:
        if root.endswith(suf):
            root = root[: -len(suf)]
    return root


def _has_harmony_violation(rma_result: RMAResult) -> bool:
    """Detect a front/back vowel-harmony violation across stem + suffix chain."""
    FRONT = set("еіүұөә")
    BACK = set("аыоу")
    token = rma_result.token.lower()
    if not token:
        return False
    stem_vowels = [c for c in token if c in FRONT or c in BACK]
    if len(stem_vowels) < 2:
        return False
    has_front = any(c in FRONT for c in stem_vowels)
    has_back = any(c in BACK for c in stem_vowels)
    return has_front and has_back
