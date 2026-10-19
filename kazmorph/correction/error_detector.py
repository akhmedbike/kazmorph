"""Surface error detector: p_err from lexical/morphological evidence.

The legacy three-way detector estimated p_err = 1 − POS-tagger confidence.
That signal measures *tagging* uncertainty, not orthographic validity: a
misspelled word that the tagger confidently parses (китапты with a Russian
«и») never crosses the τ_e gate and is never even shown to the candidates.
Measured on corrupted dev, this capped end-to-end recall hard.

This module learns p_err from cheap surface evidence instead:

  * in_lemma_vocab   — token is a known lemma (~22k Apertium lemmas)
  * in_form_index    — token is a real corpus word form (leakage-free index)
  * log_freq         — corpus frequency of the form
  * kappa            — MFRN morphophonological compatibility of the RMA parse
  * rma_unk          — RMA recovers no analysis at all
  * has_valid_split  — the token's tail decomposes into a known affix chain
  * pos_conf         — the (contextual) POS-tagger confidence, kept so the
                       model can still use the old signal where it helps

A logistic regression over these features is fitted by
``scripts/train_error_detector.py`` on clean train tokens (label 0) vs their
corrupted variants (label 1, produced by the same nine operators used at
evaluation) and exported to ``models/error_detector.json``.

Serving contract: :func:`estimate_p_err` returns the logistic score when the
model file exists, and falls back to the legacy ``1 − pos_conf`` otherwise —
so the corrector keeps working on checkouts without the fitted weights.

Honest ceiling: operators that corrupt an affix into ANOTHER valid word
(grammatical-affix substitution can produce real surface forms) are
orthographically undetectable by surface evidence; those errors remain for
context-aware detection (see the paper's Limitations).
"""

from __future__ import annotations

import json
import math
import os

from kazmorph.correction.candidate_generator import _valid_splits
from kazmorph.correction.freq import frequency_of, get_frequency_dict
from kazmorph.correction.form_index import form_exists
from kazmorph.suggestor import _get_vocab

# Resolve the model file relative to the repo root (kazmorph/correction/ → up 2).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "error_detector.json")

# Feature order is part of the model-file contract (weights align to it).
FEATURES: tuple[str, ...] = (
    "in_lemma_vocab", "in_form_index", "log_freq", "kappa",
    "rma_unk", "has_valid_split", "pos_conf",
)

_MODEL: dict | None = None
_MODEL_LOADED = False


def extract_features(token: str, pos_conf: float = 1.0, *, rma_result=None) -> dict[str, float]:
    """Surface features for one token (all in [0, ~10]; cheap, no neural)."""
    from kazmorph.correction.candidate_generator import _get_mfrn, _get_rma
    t = token.lower()
    if len(t) > 30:  # absurd tokens are errors by construction; clamp cost
        t = t[:30]
    if rma_result is None:
        rma_result = _get_rma().analyze(token)
    kappa = _get_mfrn().compute_kappa(rma_result)
    freq = frequency_of(t, get_frequency_dict())
    return {
        "in_lemma_vocab": 1.0 if t in _get_vocab() else 0.0,
        "in_form_index": 1.0 if form_exists(t) else 0.0,
        "log_freq": math.log1p(freq) / 10.0,
        "kappa": kappa,
        "rma_unk": 1.0 if rma_result.pos == "UNK" else 0.0,
        "has_valid_split": 1.0 if _valid_splits(t) else 0.0,
        "pos_conf": max(0.0, min(1.0, pos_conf)),
    }


def _validate(model: dict) -> dict:
    if tuple(model["features"]) != FEATURES:
        raise ValueError(f"error_detector model feature mismatch: {model['features']}")
    return model


def load_model(path: str | None = None) -> dict | None:
    """Load {weights, bias, features, meta}; None if the file is absent.

    Explicit ``path`` arguments are one-shot reads that do NOT touch the
    process-wide default cache (tests pass tmp paths; poisoning the cache
    with them would silently switch serving back to the legacy fallback).
    """
    global _MODEL, _MODEL_LOADED
    if path is not None:
        return (_validate(json.load(open(path, encoding="utf-8")))
                if os.path.exists(path) else None)
    if not _MODEL_LOADED:
        _MODEL = (_validate(json.load(open(DEFAULT_MODEL_PATH, encoding="utf-8")))
                  if os.path.exists(DEFAULT_MODEL_PATH) else None)
        _MODEL_LOADED = True
    return _MODEL


def estimate_p_err(
    token: str,
    pos_conf: float = 1.0,
    *,
    rma_result=None,
    model_path: str | None = None,
) -> float:
    """Estimated probability that the token is misspelled, in [0, 1].

    Logistic score over :data:`FEATURES` when a fitted model exists; legacy
    ``1 − pos_conf`` otherwise.
    """
    model = load_model(model_path)
    if model is None:
        return max(0.0, min(1.0, 1.0 - pos_conf))
    x = extract_features(token, pos_conf, rma_result=rma_result)
    z = model["bias"] + sum(model["weights"][f] * x[f] for f in FEATURES)
    # Stable logistic.
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
