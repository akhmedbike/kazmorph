"""Three-way correction: KEEP / CORRECT / REVIEW.

The shipped :mod:`kazmorph.corrector` is binary — it replaces any
low-confidence or UNK token with its top Levenshtein candidate. That design
has no notion of "uncertain, defer to a human": every low-confidence token is
changed, which inflates the false-correction rate on rare-but-correct forms
(proper names, low-frequency words, productive derivations).

This module implements the three-way decision rule. For each token it:

  1. Estimates p_err (the probability the token is erroneous) from the
     calibrated POS confidence: p_err ≈ 1 - confidence. Low p_err → KEEP.
  2. Generates ranked candidates
     (:func:`kazmorph.correction.candidate_generator.generate_candidates`).
  3. Computes normalised probabilities over the candidate set *including the
     original token* (:func:`candidate_probs`), so KEEP is a first-class
     decision.
  4. Applies the margin rule: CORRECT only when the top candidate is
     sufficiently probable (τ_c) AND clearly ahead of the runner-up (τ_m);
     otherwise REVIEW (retain the original, record the candidates).

The full decision set is written to a per-token trace dict so that downstream
metrics (FCR, risk–coverage, error localisation by stage) can be computed from
stored predictions without re-running the pipeline.

This corrector does NOT implement the morphological lattice, the synthesiser,
the multi-granular gated encoder, or joint training — those are out of scope
for the analyzer product and disclosed as unimplemented where relevant.

Adapted from the IHCONCS_2026_paper project (MIT): only the ``config`` imports
were replaced with the module-level constants below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kazmorph.correction.candidate_generator import (
    generate_candidates, candidate_probs,
)
from kazmorph.correction.error_detector import estimate_p_err

# ── Decision thresholds (were ``config.*`` in the source project) ────────────
# Defaults selected on validation. Override per-call for risk–coverage sweeps
# or for the ablation that removes REVIEW.
TAU_E = 0.15   # below this estimated error probability → KEEP
# Operating point selected on the HONEST corrupted-dev grid (identity
# sentences included in precision; 3 seeds × 9 operators; scripts/
# eval_correction.py → reports/correction_eval.json): F0.5-optimal
# scale=2.0 / τ_c=0.30 / τ_m=0.10 gives P≈0.66, R≈0.26, F0.5≈0.51 with the
# surface error detector and the evidence-weighted original.
TAU_C = 0.30   # top candidate normalised probability must reach this → CORRECT
TAU_M = 0.10   # margin (top − second) must reach this → CORRECT
MAX_CANDIDATES = 10
# Error-evidence weight: the original token enters the candidate softmax at
# energy α·p_err instead of 0. A confidently-flagged token (high p_err) is
# thus easier to beat than a barely-suspicious one — this couples the
# detection score into the decision instead of using it only as a gate.
# Measured on corrupted dev (scripts/eval_correction.py): without it, CORRECT
# is nearly unreachable because the zero-energy original keeps the largest
# softmax share for every realistic candidate energy.
ORIGINAL_EVIDENCE_ALPHA = 1.0
# Decision temperature: energies are scaled by this factor inside the
# candidate softmax (s > 1 sharpens). With s = 1 and a full candidate list
# the top probability tops out at ~0.2–0.3, capping CORRECT recall.
# Calibrated together with τ_c/τ_m on corrupted dev (scripts/eval_correction.py);
# the honest grid picked s = 2.0 (sharpens the softmax so the top candidate
# can dominate near-tied alternatives and clear the τ_c/τ_m gate).
DECISION_ENERGY_SCALE = 2.0
# Legacy binary-corrector threshold (kazmorph.corrector default).
LEGACY_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class DecisionResult:
    """Per-token correction decision and the full evidence behind it."""
    token: str
    pos: str
    confidence: float              # calibrated POS confidence of the input token
    p_err: float                   # estimated error probability = 1 - confidence
    decision: str                  # "KEEP" | "CORRECT" | "REVIEW"
    predicted: str                 # emitted surface form (== token on KEEP/REVIEW)
    top_candidate: str             # highest-ranked correction candidate ("-" if none)
    top_prob: float                # normalised probability of the top candidate
    margin: float                  # top_prob - second_prob (0.0 if <2 candidates)
    n_candidates: int              # number of correction candidates considered
    candidates: list[tuple[str, float]] = field(default_factory=list)  # ranked (cand, prob)

    def to_trace_dict(self, *, gold: str = "", was_corrupted: bool = False) -> dict:
        """Serialise to the dict shape correction_metrics expects."""
        return {
            "token": self.token,
            "gold": gold,
            "original": self.token,
            "predicted": self.predicted,
            "decision": self.decision,
            "pos": self.pos,
            "confidence": self.confidence,
            "p_err": self.p_err,
            "top_candidate": self.top_candidate,
            "top_prob": self.top_prob,
            "margin": self.margin,
            "n_candidates": self.n_candidates,
            "was_corrupted": was_corrupted,
            "candidates": [{"word": w, "prob": p} for w, p in self.candidates[:10]],
        }


def _classify_token(
    pipeline_result: dict,
    tau_e: float = TAU_E,
    tau_c: float = TAU_C,
    tau_m: float = TAU_M,
    candidate_mode: str = "dl_fst_filter",
    max_candidates: int = MAX_CANDIDATES,
) -> DecisionResult:
    """Apply the three-way rule to a single analysed token.

    ``pipeline_result`` is one dict from ``Pipeline.process(text)`` — it
    carries the fields the decision needs: ``token``, ``pos``, ``confidence``.
    (The ``suggestions`` field on that dict is the legacy top-3 list; we
    ignore it and regenerate the full ranked candidate set here, because the
    legacy list is capped at 3 and has no probabilities.)
    """
    token = pipeline_result["token"]
    pos = pipeline_result.get("pos", "UNK")
    confidence = float(pipeline_result.get("confidence", 0.0))
    # Surface-evidence detector (lexicon/form membership, κ, frequency, affix
    # tail validity + the tagger confidence). Falls back to the legacy
    # 1 − confidence when no fitted model is present — see error_detector.
    p_err = estimate_p_err(token, confidence)

    # ── KEEP: low estimated error probability, regardless of candidates ──
    # This is the single most important guard against false correction: a
    # high-confidence token is never second-guessed.
    if p_err < tau_e:
        return DecisionResult(
            token=token, pos=pos, confidence=confidence, p_err=p_err,
            decision="KEEP", predicted=token,
            top_candidate="-", top_prob=0.0, margin=0.0, n_candidates=0,
        )

    # Skip non-word tokens entirely. Punctuation, numerals and closed-class
    # items are never correction targets — they carry no lexical content that
    # a candidate replacement could improve.
    if pos in ("PUNCT", "NUM", "SYM") or len(token) < 2:
        return DecisionResult(
            token=token, pos=pos, confidence=confidence, p_err=p_err,
            decision="KEEP", predicted=token,
            top_candidate="-", top_prob=0.0, margin=0.0, n_candidates=0,
        )

    # ── Candidate generation + softmax over {original} ∪ candidates ──
    # The original carries error evidence: energy α·p_err (see
    # ORIGINAL_EVIDENCE_ALPHA) — a strongly-suspect token is easier to beat.
    cands = generate_candidates(token, mode=candidate_mode, max_candidates=max_candidates)
    probs = candidate_probs(
        cands, include_original=True, original_token=token,
        original_energy=ORIGINAL_EVIDENCE_ALPHA * p_err,
        energy_scale=DECISION_ENERGY_SCALE,
    )

    if not probs:
        # No correction candidate and not KEEP-by-confidence either (tau_e was
        # crossed but nothing was retrieved). Treat as REVIEW with no options:
        # the token is suspicious but we have no replacement to offer.
        return DecisionResult(
            token=token, pos=pos, confidence=confidence, p_err=p_err,
            decision="REVIEW", predicted=token,
            top_candidate="-", top_prob=0.0, margin=0.0, n_candidates=0,
        )

    # Identify the top *correction* candidate (excluding the original token)
    # and the overall top-1 / top-2 for the margin computation.
    original_lower = token.lower()
    correction_probs = [(w, p) for (w, p) in probs if w.lower() != original_lower]
    if not correction_probs:
        # Only the original survived (all retrieved candidates were equal to
        # it, or the FST filter removed everything else). KEEP wins.
        return DecisionResult(
            token=token, pos=pos, confidence=confidence, p_err=p_err,
            decision="KEEP", predicted=token,
            top_candidate="-", top_prob=probs[0][1], margin=0.0, n_candidates=0,
            candidates=probs,
        )

    top_word, top_prob = correction_probs[0]
    # Margin = gap between the top correction and the next-best alternative,
    # where "next-best" includes the original token (so a correction must beat
    # KEEP by τ_m, not just beat the second correction).
    remaining = [(w, p) for (w, p) in probs if w.lower() != top_word.lower()]
    second_prob = remaining[0][1] if remaining else 0.0
    margin = top_prob - second_prob

    # ── CORRECT vs REVIEW ──
    if top_prob >= tau_c and margin >= tau_m:
        decision = "CORRECT"
        predicted = top_word
    else:
        # REVIEW: retain the original surface form, but keep the candidate
        # list so a human (or a downstream stronger model) can pick. This is
        # the selective-abstention path that the FCR argument rests on.
        decision = "REVIEW"
        predicted = token

    return DecisionResult(
        token=token, pos=pos, confidence=confidence, p_err=p_err,
        decision=decision, predicted=predicted,
        top_candidate=top_word, top_prob=top_prob, margin=margin,
        n_candidates=len(correction_probs), candidates=correction_probs,
    )


def correct_token_3way(
    pipeline_result: dict,
    *,
    tau_e: float = TAU_E,
    tau_c: float = TAU_C,
    tau_m: float = TAU_M,
    candidate_mode: str = "dl_fst_filter",
) -> DecisionResult:
    """Public entry point: classify a single analysed token.

    Thresholds default to the module values (selected on validation); override
    them only for the risk–coverage sweep or the ablation that removes REVIEW.
    """
    return _classify_token(
        pipeline_result, tau_e=tau_e, tau_c=tau_c, tau_m=tau_m,
        candidate_mode=candidate_mode,
    )


def correct_token_binary(
    pipeline_result: dict,
    *,
    confidence_threshold: float = LEGACY_CONFIDENCE_THRESHOLD,
) -> DecisionResult:
    """The legacy binary rule, for the "− REVIEW routing" ablation/baseline.

    Mirrors :func:`kazmorph.corrector.correct`: any low-confidence / UNK token
    with a non-empty suggestion list is CORRECTed to its top candidate;
    everything else is KEEP. There is no REVIEW. This is the baseline against
    which the three-way rule's FCR reduction is measured.
    """
    token = pipeline_result["token"]
    pos = pipeline_result.get("pos", "UNK")
    confidence = float(pipeline_result.get("confidence", 0.0))
    p_err = 1.0 - confidence
    suggestions = pipeline_result.get("suggestions") or []

    low_conf = confidence < confidence_threshold
    is_unknown = pos == "UNK"
    if suggestions and (low_conf or is_unknown) and suggestions[0] != token:
        return DecisionResult(
            token=token, pos=pos, confidence=confidence, p_err=p_err,
            decision="CORRECT", predicted=suggestions[0],
            top_candidate=suggestions[0], top_prob=1.0, margin=1.0,
            n_candidates=len(suggestions),
        )
    return DecisionResult(
        token=token, pos=pos, confidence=confidence, p_err=p_err,
        decision="KEEP", predicted=token,
        top_candidate="-", top_prob=0.0, margin=0.0, n_candidates=0,
    )


def correct_sentence_3way(
    pipeline,
    text: str,
    *,
    tau_e: float = TAU_E,
    tau_c: float = TAU_C,
    tau_m: float = TAU_M,
    candidate_mode: str = "dl_fst_filter",
) -> list[DecisionResult]:
    """Run the three-way corrector over every token of `text`.

    Returns one :class:`DecisionResult` per token, in surface order. The
    caller is responsible for reconstructing the corrected string (most
    evaluations operate on the per-token decisions directly, so we don't
    synthesise text here — that avoids re-implementing whitespace handling).
    """
    results = pipeline.process(text)
    return [
        correct_token_3way(
            r, tau_e=tau_e, tau_c=tau_c, tau_m=tau_m, candidate_mode=candidate_mode,
        )
        for r in results
    ]
