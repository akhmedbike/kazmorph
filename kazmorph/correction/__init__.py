"""Three-way correction subsystem for kazmorph.

This subpackage ports the correction-oriented tooling from the
IHCONCS_2026_paper project (MIT): the three-way KEEP / CORRECT / REVIEW
corrector, the multi-mode candidate generator (with stem recombination),
the correction-quality metrics (FCR, CER, WER, Recall@K, MRR, risk–coverage),
the nine-operator controlled-error generator, and the leakage-free corpus
frequency dictionary.

These modules are additive: the legacy binary corrector
(:func:`kazmorph.corrector.correct`) and its ``POST /correct`` endpoint are
unchanged. The three-way rule is exposed at ``POST /correct3``.

Quick start
-----------
>>> from kazmorph.correction import correct_sentence_3way
>>> decisions = correct_sentence_3way(pipeline, "менің текстім")  # doctest: +SKIP
>>> [d.decision for d in decisions]  # ['KEEP', 'REVIEW', ...]      # doctest: +SKIP
"""

from kazmorph.correction.corrector_3way import (
    DecisionResult,
    correct_token_3way,
    correct_token_binary,
    correct_sentence_3way,
)
from kazmorph.correction.candidate_generator import (
    generate_candidates,
    candidate_probs,
)
from kazmorph.correction import correction_metrics
from kazmorph.correction import keyboards
from kazmorph.correction import corrupt
from kazmorph.correction import freq

__all__ = [
    # corrector_3way
    "DecisionResult",
    "correct_token_3way",
    "correct_token_binary",
    "correct_sentence_3way",
    # candidate_generator
    "generate_candidates",
    "candidate_probs",
    # submodules
    "correction_metrics",
    "keyboards",
    "corrupt",
    "freq",
]
