"""
Tests for kazmorph/crf_tagger.py (the sklearn-crfsuite wrapper / Stage 2).

These tests run WITHOUT torch or sklearn-crfsuite: they inject a fake
estimator that mimics the crfsuite API (``predict`` + ``predict_marginals``)
and check that:

  1. The CRF wrapper's reported confidence is the **marginal probability** of
     the predicted tag (not the legacy ``rma.confidence + 0.1`` heuristic).
     Honest marginal confidence is what the hard-neural-fallback mode relies
     on to decide which tokens to reroute to the transformer.
  2. If the estimator lacks ``predict_marginals`` (or it raises), the wrapper
     degrades gracefully to the legacy heuristic, so older stored models keep
     working.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazmorph.crf_tagger import CRFTagger
from kazmorph.rma import RMAResult


class _FakeCRF:
    """Mimics sklearn_crfsuite.CRF: predict() and predict_marginals()."""

    def __init__(self, preds, marginals):
        self._preds = preds          # list of label strings (one per token)
        self._marginals = marginals  # list of {label: prob} (one per token)

    def predict(self, Xseq):
        # Xseq is a list with one element per sentence (each a feature list).
        return [self._preds]

    def predict_marginals(self, Xseq):
        return [self._marginals]


class _FakeCRFNoMarginals:
    """Estimator without predict_marginals (legacy/foreign estimator)."""

    def __init__(self, preds):
        self._preds = preds

    def predict(self, Xseq):
        return [self._preds]


def _rma(token, pos="NOUN", conf=0.5):
    return RMAResult(token=token, lemma=token, pos=pos, suffixes=[],
                     suffix_categories=[], confidence=conf)


def test_confidence_is_marginal_of_predicted_tag():
    t = CRFTagger()
    t.model = _FakeCRF(
        preds=["NOUN", "VERB"],
        marginals=[
            {"NOUN": 0.90, "VERB": 0.10},
            {"NOUN": 0.30, "VERB": 0.70},
        ],
    )
    out = t.tag([_rma("сөз"), _rma("барды")])
    assert out[0].pos == "NOUN"
    assert out[0].confidence == 0.90
    assert out[1].pos == "VERB"
    assert out[1].confidence == 0.70


def test_legacy_heuristic_when_no_predict_marginals():
    # rma.confidence = 0.5 -> legacy conf = min(0.5 + 0.1, 1.0) = 0.6
    t = CRFTagger()
    t.model = _FakeCRFNoMarginals(preds=["NOUN"])
    out = t.tag([_rma("сөз", conf=0.5)])
    assert out[0].pos == "NOUN"
    assert out[0].confidence == 0.6


def test_predict_marginals_raising_falls_back_to_heuristic():
    class _Raises:
        def predict(self, Xseq):
            return [["VERB"]]

        def predict_marginals(self, Xseq):
            raise RuntimeError("marginals unavailable in this build")

    t = CRFTagger()
    t.model = _Raises()
    out = t.tag([_rma("жүр", conf=0.4)])
    assert out[0].pos == "VERB"
    assert out[0].confidence == 0.5  # min(0.4 + 0.1, 1.0)


def test_no_model_uses_deterministic_fallback():
    # No model loaded at all -> _fallback_adjust applies a POS-dependent penalty.
    t = CRFTagger()
    assert t.model is None
    out = t.tag([_rma("сөз", pos="NOUN", conf=0.9), _rma("жүр", pos="UNK", conf=0.2)])
    # NOUN penalty 0.05, UNK penalty 0.25.
    assert abs(out[0].confidence - 0.85) < 1e-6
    assert abs(out[1].confidence - 0.0) < 1e-6
