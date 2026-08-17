"""Paired significance testing for ablation / transfer experiments.

This module lifts the pure significance-testing functions into the kazmorph
package. Previously these only lived in ``scripts/`` (out-of-band) and were
not importable from the library; the package now provides:

  * :func:`cohens_d_paired`     — standardized effect size for paired samples
  * :func:`effect_label`        — negligible/small/medium/large label for a d
  * :func:`holm_bonferroni`     — step-down Holm-Bonferroni correction for
                                  multiple comparisons
  * :func:`bootstrap_ci_diff`   — percentile bootstrap 95% CI of the mean
                                  paired difference
  * :func:`paired_test`         — t-test + Wilcoxon + Cohen's d + bootstrap CI

The pure functions (:func:`cohens_d_paired`, :func:`effect_label`,
:func:`holm_bonferroni`, :func:`bootstrap_ci_diff`) have **no third-party
dependencies** — they work without scipy installed. Only :func:`paired_test`
needs scipy (for ``scipy.stats.ttest_rel`` and ``scipy.stats.wilcoxon``); it
imports scipy lazily and raises a clear error if scipy is absent, so the pure
functions remain usable in lightweight environments.

When to use which
-----------------
For ablation/transfer experiments measured per cross-validation fold, the
pairing is the *same fold* across two configurations, which removes fold
difficulty as a confound. With n = k ≈ 10 folds, Wilcoxon and Cohen's d are
well-defined (Wilcoxon cannot reach p < 0.05 below n = 6 or so). The bootstrap
CI of the mean difference is generally more informative than the p-value and
is what should be quoted as "the difference is X pp, 95% CI [lo, hi]".

For multiple comparisons (a family of ablations × tasks), apply
:func:`holm_bonferroni` to the raw p-values to control the family-wise error
rate. McNemar's test (already in ``scripts/evaluate.py``) is the right tool
for single-split token-level comparisons and is out of scope here.

Adapted from the CAGF-CBT-CRF project (MIT); the CLI/argparse ``main()`` was
dropped (this is a library module), and scipy is now imported lazily.
"""

from __future__ import annotations

import math
import random
from typing import Sequence, Tuple


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples (standardized mean of per-observation differences).

    Returns NaN for mismatched or too-short inputs, 0.0 for a zero mean with
    zero variance, and ``inf`` for a nonzero mean with zero variance.
    """
    if len(a) != len(b) or len(a) < 2:
        return float('nan')
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    # sample std of differences (ddof=1)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0 if mean == 0 else float('inf')
    return mean / sd


def effect_label(d: float) -> str:
    """Qualitative label for a Cohen's d (negligible/small/medium/large).

    Thresholds: negligible < 0.2, small < 0.5, medium < 0.8, large ≥ 0.8.
    Returns ``'n/a'`` for NaN.
    """
    if math.isnan(d):
        return 'n/a'
    ad = abs(d)
    if ad < 0.2:
        return 'negligible'
    if ad < 0.5:
        return 'small'
    if ad < 0.8:
        return 'medium'
    return 'large'


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Step-down Holm-Bonferroni correction.

    Returns adjusted p-values aligned to the input order. Step-down property:
    the adjusted p-value is non-decreasing in rank (a larger raw p never gets
    a smaller adjusted p than a preceding one), enforced via ``running_max``.
    """
    pvals = list(pvals)
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, pvals[idx] * (m - rank))
        running_max = max(running_max, corrected)
        adj[idx] = running_max
    return adj


def bootstrap_ci_diff(a: Sequence[float], b: Sequence[float],
                      n_boot: int = 10000, alpha: float = 0.05,
                      seed: int = 12345) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean paired difference (a - b).

    Resamples the per-observation differences with replacement, takes the mean
    of each resample, and returns the (alpha/2, 1-alpha/2) percentiles. Quote
    this in results as "the difference is X pp, 95% CI [lo, hi]".

    Returns (NaN, NaN) for mismatched or too-short inputs.
    """
    if len(a) != len(b) or len(a) < 2:
        return (float('nan'), float('nan'))
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    return (means[lo_idx], means[hi_idx])


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Full paired comparison: t-test + Wilcoxon + Cohen's d + bootstrap CI.

    Requires scipy for the t-test and Wilcoxon signed-rank test. The pure
    components (Cohen's d, bootstrap CI) are always computed. Raises
    ``ImportError`` with an install hint if scipy is not available.

    Returns a dict with: ``mean_a``, ``mean_b``, ``mean_diff``, ``n``,
    ``t_stat``, ``t_p``, ``wilcoxon_w``, ``wilcoxon_p``, ``cohens_d``,
    ``effect``, ``bootstrap_ci_lo``, ``bootstrap_ci_hi``. The Wilcoxon
    statistics are NaN if scipy raises (e.g. all differences zero).
    """
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "kazmorph.significance.paired_test requires scipy. "
            "Install it with: pip install scipy"
        ) from exc

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # scipy raises when all differences are zero or n is too small.
        w_stat, w_p = (float('nan'), float('nan'))
    d = cohens_d_paired(a, b)
    ci_lo, ci_hi = bootstrap_ci_diff(a, b)
    return {
        'mean_a': sum(a) / len(a), 'mean_b': sum(b) / len(b),
        'mean_diff': sum(a) / len(a) - sum(b) / len(b),
        'n': len(a), 't_stat': float(t_stat), 't_p': float(t_p),
        'wilcoxon_w': float(w_stat), 'wilcoxon_p': float(w_p),
        'cohens_d': d, 'effect': effect_label(d),
        'bootstrap_ci_lo': ci_lo, 'bootstrap_ci_hi': ci_hi,
    }


__all__ = [
    "cohens_d_paired",
    "effect_label",
    "holm_bonferroni",
    "bootstrap_ci_diff",
    "paired_test",
]
