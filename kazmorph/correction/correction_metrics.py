"""Correction-quality metrics for kazmorph.

The kazmorph codebase ships only POS-tagging metrics (accuracy, per-tag P/R/F1,
macro/weighted averages, confusion matrices). Correction evaluation needs a
different family of measures — correction-oriented, not tagging-oriented. This
module implements them.

Every function here is pure (no I/O, no globals) so it composes cleanly inside
cross-validation loops and serialises trivially into the per-token trace JSON
that reproducible evaluation relies on.

Adapted from the IHCONCS_2026_paper project (MIT), with no functional changes.

Conventions
-----------
* A "correction decision" is one of three labels: KEEP, CORRECT, REVIEW
  (three-way rule in :mod:`kazmorph.correction.corrector_3way`). At the token
  level we additionally record whether the decision was *correct* in the gold
  sense: a KEEP on a gold-clean token is correct, a CORRECT to the gold form is
  correct, everything else is not.
* `gold` is always the normative surface form; `pred` is what the system
  emitted. For the correction task the "reference" is the original clean
  sentence, not a tagging.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


# ── F-beta ───────────────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Weighted F-measure.

    F_β = (1+β²)·P·R / (β²·P + R).

    β=0.5 (F₀.₅) weights precision twice as much as recall, reflecting the
    editorial position that an unjustified change to a correct token is more
    costly than retaining a doubtful one. Returns 0.0 when both P and R are
    zero (the degenerate no-positive-predictions case).
    """
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall)


def precision_recall_fbeta(
    tp: int, fp: int, fn: int, beta: float = 0.5,
) -> tuple[float, float, float]:
    """Convenience: (P, R, F_β) from raw confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, f_beta(precision, recall, beta)


# ── Edit distance (character + word level) ───────────────────────────────────

def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Generic Levenshtein distance over two sequences (chars or token lists).

    Pure-Python, O(len(a)·len(b)) time, O(min(len)) space (two rolling rows).
    Used for both CER (a, b are strings → iterated as characters) and WER
    (a, b are lists of word strings). We do not need Damerau transposition
    here because corrections either match the gold form exactly or replace it;
    the SymSpell candidate generator already accounts for transpositions at
    retrieval time, so the evaluation-distance is plain Levenshtein.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure b is the shorter for the narrower inner row.
    if len(b) > len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur[j] = ins if ins < dele else dele
            if sub < cur[j]:
                cur[j] = sub
        prev = cur
    return prev[len(b)]


def cer(hyp: str, ref: str) -> float:
    """Character error rate: edit_distance(hyp, ref) / len(ref).

    Returns 0.0 for an empty reference (no division by zero; both empty strings
    are a perfect match). `hyp` is the system output, `ref` is the gold.
    """
    if not ref:
        return 0.0
    return _levenshtein(hyp, ref) / len(ref)


def wer(hyp_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    """Word error rate: edit_distance(hyp, ref) / len(ref).

    Tokens are compared as whole strings (case-sensitive). The caller is
    responsible for consistent tokenisation between hyp and ref — use the
    pipeline's own tokeniser for both, so alignment is exact.
    """
    if not ref_tokens:
        return 0.0
    return _levenshtein(list(hyp_tokens), list(ref_tokens)) / len(ref_tokens)


# ── Ranking metrics ──────────────────────────────────────────────────────────

def recall_at_k(
    gold_in_topk: Sequence[bool], k: int,
) -> float:
    """Recall@K.

    `gold_in_topk` is a per-token list of booleans: True iff the gold surface
    form appears in the system's top-K candidate list. The mean of this
    indicator over all evaluated tokens is Recall@K. We pass the already-
    resolved booleans (rather than the candidate lists) so the same trace can
    be reused for every K without re-scanning candidates.
    """
    if not gold_in_topk:
        return 0.0
    return sum(1 for hit in gold_in_topk if hit) / len(gold_in_topk)


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    """Mean reciprocal rank (MRR).

    `ranks` contains, for each token, the 1-based rank position of the gold
    form in the ranked candidate list, or 0 if the gold form is absent.
    MRR = mean(1/rank) over tokens where rank > 0; absent-gold tokens
    contribute 0 (1/0 is treated as 0). This matches the standard IR convention.
    """
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r > 0:
            total += 1.0 / r
    return total / len(ranks)


def rank_of(gold: str, ranked_candidates: Sequence[str]) -> int:
    """1-based rank of `gold` in `ranked_candidates`, or 0 if absent.

    Case-insensitive comparison, mirroring kazmorph's suggestor (which
    lowercases before lookup). Equality is exact surface match — no stemming
    or normalisation beyond case.
    """
    g = gold.lower()
    for i, c in enumerate(ranked_candidates, 1):
        if c.lower() == g:
            return i
    return 0


# ── Risk–coverage (selective prediction) ─────────────────────────────────────

def risk_coverage(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float],
) -> list[dict]:
    """Risk–coverage curve.

    For each confidence threshold τ, an automatic decision is taken on every
    token whose top-1 confidence is ≥ τ; the rest are routed to REVIEW.

      R(τ) = #(incorrect automatic decisions) / #(automatic decisions)
      C(τ) = #(automatic decisions) / #(all evaluated tokens)

    A high-quality selective system sweeps from low coverage / low risk (only
    the most confident tokens auto-decided) toward high coverage / higher risk.

    Parameters
    ----------
    confidences : per-token confidence of the system's top-1 candidate.
    correct     : per-token gold correctness of the top-1 decision.
    thresholds  : τ values to sweep.

    Returns a list of {threshold, risk, coverage, n_auto, n_incorrect, n_total}
    dicts sorted by ascending threshold.
    """
    assert len(confidences) == len(correct), "confidences and correct must align"
    n_total = len(confidences)
    out: list[dict] = []
    for tau in sorted(thresholds):
        n_auto = n_incorrect = 0
        for c, ok in zip(confidences, correct):
            if c >= tau:
                n_auto += 1
                if not ok:
                    n_incorrect += 1
        risk = n_incorrect / n_auto if n_auto else 0.0
        coverage = n_auto / n_total if n_total else 0.0
        out.append({
            "threshold": tau,
            "risk": risk,
            "coverage": coverage,
            "n_auto": n_auto,
            "n_incorrect": n_incorrect,
            "n_total": n_total,
        })
    return out


# ── Aggregating per-token decisions into correction metrics ─────────────────

def correction_prf(
    decisions: Iterable[dict],
) -> dict:
    """Aggregate per-token correction decisions into P / R / F₀.₅.

    Each input dict carries the fields written by corrector_3way:
        {
          "gold": str,            # normative surface form
          "original": str,        # input (possibly corrupted) token
          "predicted": str,       # what the system emitted (== original on KEEP/REVIEW)
          "decision": "KEEP"|"CORRECT"|"REVIEW",
          "was_corrupted": bool,  # True iff the input token was actually corrupted
        }

    Counting convention (GEC standard, fixed denominator E = tp + fn):
      * A CORRECT counts as a true positive iff predicted == gold.
      * A CORRECT to the wrong form is a false positive (the system changed a
        token, just not to the right thing). Crucially, if that token was
        itself corrupted, the original error ALSO went unfixed — the gold form
        was never produced — so it is counted as a false negative as well.
        This keeps the recall denominator E = tp + fn equal to the number of
        injected errors regardless of how often the system mis-edits; a
        system that changes many tokens to wrong forms cannot inflate its
        recall by shrinking the denominator.
      * A token that was corrupted but NOT corrected (KEEP or REVIEW-with-
        original-retained) is a false negative: the error survived.
      * FCR (false-correction rate) is computed separately, only over
        non-corrupted tokens, by :func:`false_correction_rate`.
    """
    tp = fp = fn = 0
    n_keep_correct = n_review = 0
    for d in decisions:
        pred = d["predicted"]
        gold = d["gold"]
        decision = d["decision"]
        corrupted = d["was_corrupted"]

        if decision == "CORRECT":
            if pred.lower() == gold.lower():
                tp += 1
            else:
                # Wrong correction: a false positive always (an unjustified
                # change was made), and — if the token was corrupted — also a
                # false negative, because the injected error was not resolved
                # to the gold form. Both increments apply.
                fp += 1
                if corrupted:
                    fn += 1
        else:  # KEEP or REVIEW (both retain the original input)
            if corrupted:
                # The input error survived → false negative for correction recall.
                fn += 1
            else:
                n_keep_correct += 1
        if decision == "REVIEW":
            n_review += 1

    precision, recall, f05 = precision_recall_fbeta(tp, fp, fn, beta=0.5)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "n_keep_correct": n_keep_correct,
        "n_review": n_review,
    }


def false_correction_rate(decisions: Iterable[dict]) -> dict:
    """FCR over non-corrupted tokens only.

    FCR = #(changed correct tokens) / #(correct tokens).

    On the controlled-error set a "correct token" is one whose input equals
    the gold form (an identity example, or a non-target token in a corrupted
    sentence). On a clean-text FCR evaluation every token is correct by
    construction. Either way the denominator is the count of tokens the system
    *should not have touched*, and the numerator is how many of those it
    changed (CORRECT decision). REVIEW does not count as a change — the
    original token is retained.
    """
    n_correct = n_changed = 0
    for d in decisions:
        if d["was_corrupted"]:
            continue
        n_correct += 1
        if d["decision"] == "CORRECT":
            n_changed += 1
    fcr = n_changed / n_correct if n_correct else 0.0
    return {"fcr": fcr, "n_correct": n_correct, "n_changed": n_changed}


# ── Mean ± std helper for aggregating across folds / seeds ───────────────────

def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and sample standard deviation (ddof=1).

    Returns (0.0, 0.0) for fewer than two values. Used everywhere we aggregate
    a scalar metric across folds × corruption-seed runs into `mean ± std` cells.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def confidence_interval_95(values: Sequence[float]) -> tuple[float, float]:
    """95% normal-approximation confidence interval on the mean.

    CI = mean ± 1.96 · (std / √n). Reported alongside mean ± std for the
    headline F₀.₅. For n < 2 the interval collapses to the mean.
    """
    n = len(values)
    mean, std = mean_std(values)
    if n < 2:
        return mean, mean
    half = 1.96 * std / math.sqrt(n)
    return mean - half, mean + half
