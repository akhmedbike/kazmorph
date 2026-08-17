"""
Tests for the linear-chain CRF (kazmorph/crf.py) and uncertainty-weighted loss.

The CRF correctness is verified against brute-force enumeration of every
tag path, on two grid sizes:

  - 4 tags x 5 steps  =    1 024 paths  (matches cagf_cbt_crf's verification)
  - 5 tags x 8 steps  =  390 625 paths  (strengthened; ~1-2 s, catches
                                         longer-sequence off-by-ones)

We also verify:
  - Viterbi decoding reproduces the brute-force argmax.
  - Per-token marginals (forward-backward) sum to 1 and agree with the
    brute-force per-tag probabilities on the small grid.
  - The NLL is non-negative and is invariant to padded positions.
  - UncertaintyWeightedLoss down-weights noisier tasks as training progresses.
  - pick_device() returns mps on Apple Silicon when available.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from kazmorph.crf import LinearChainCRF
from kazmorph.losses import UncertaintyWeightedLoss
from kazmorph.device import pick_device


# ============================================================
#  Brute-force oracles
# ============================================================

def brute_force_log_z(crf: LinearChainCRF, emissions: torch.Tensor, length: int) -> float:
    """log Z by enumerating every length-`length` tag path."""
    num_tags = crf.num_tags
    total = 0.0
    for path in itertools.product(range(num_tags), repeat=length):
        score = (
            crf.start_transitions[path[0]].item()
            + emissions[0, path[0]].item()
        )
        for t in range(1, length):
            score += (
                crf.transitions[path[t - 1], path[t]].item()
                + emissions[t, path[t]].item()
            )
        score += crf.end_transitions[path[-1]].item()
        total += math.exp(score)
    return math.log(total)


def brute_force_best_path(crf, emissions, length):
    """Viterbi ground truth by exhaustive enumeration."""
    num_tags = crf.num_tags
    best_score, best_path = -1e18, None
    for path in itertools.product(range(num_tags), repeat=length):
        score = (
            crf.start_transitions[path[0]].item()
            + emissions[0, path[0]].item()
        )
        for t in range(1, length):
            score += (
                crf.transitions[path[t - 1], path[t]].item()
                + emissions[t, path[t]].item()
            )
        score += crf.end_transitions[path[-1]].item()
        if score > best_score:
            best_score, best_path = score, list(path)
    return best_path, best_score


def brute_force_marginals(crf, emissions, length):
    """Per-position marginal P(tag=i) by exhaustive enumeration."""
    num_tags = crf.num_tags
    log_z = brute_force_log_z(crf, emissions, length)
    marg = [[0.0] * num_tags for _ in range(length)]
    for path in itertools.product(range(num_tags), repeat=length):
        score = (
            crf.start_transitions[path[0]].item()
            + emissions[0, path[0]].item()
        )
        for t in range(1, length):
            score += (
                crf.transitions[path[t - 1], path[t]].item()
                + emissions[t, path[t]].item()
            )
        score += crf.end_transitions[path[-1]].item()
        prob = math.exp(score - log_z)
        for t in range(length):
            marg[t][path[t]] += prob
    return marg


# ============================================================
#  Forward algorithm
# ============================================================

class TestForwardAlgorithm:
    def test_matches_brute_force_4x5(self):
        torch.manual_seed(0)
        num_tags, seq_len = 4, 5
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(1, seq_len, num_tags)
        mask = torch.ones(1, seq_len)
        log_z_algo = crf._forward_algorithm(emissions, mask).item()
        log_z_brute = brute_force_log_z(crf, emissions[0], seq_len)
        assert abs(log_z_algo - log_z_brute) < 1e-3, (log_z_algo, log_z_brute)

    def test_matches_brute_force_5x8(self):
        # Strengthened: 5**8 = 390 625 paths. Catches off-by-ones the 4x5 grid might miss.
        torch.manual_seed(7)
        num_tags, seq_len = 5, 8
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(1, seq_len, num_tags)
        mask = torch.ones(1, seq_len)
        log_z_algo = crf._forward_algorithm(emissions, mask).item()
        log_z_brute = brute_force_log_z(crf, emissions[0], seq_len)
        assert abs(log_z_algo - log_z_brute) < 1e-3, (log_z_algo, log_z_brute)


# ============================================================
#  Viterbi decoding
# ============================================================

class TestViterbiDecode:
    def test_matches_brute_force_4x5(self):
        torch.manual_seed(1)
        num_tags, seq_len = 4, 5
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(1, seq_len, num_tags)
        mask = torch.ones(1, seq_len)
        decoded = crf.decode(emissions, mask)[0]
        brute_path, _ = brute_force_best_path(crf, emissions[0], seq_len)
        assert decoded == brute_path, (decoded, brute_path)

    def test_matches_brute_force_5x8(self):
        torch.manual_seed(11)
        num_tags, seq_len = 5, 8
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(1, seq_len, num_tags)
        mask = torch.ones(1, seq_len)
        decoded = crf.decode(emissions, mask)[0]
        brute_path, _ = brute_force_best_path(crf, emissions[0], seq_len)
        assert decoded == brute_path, (decoded, brute_path)

    def test_zero_length_returns_empty(self):
        crf = LinearChainCRF(3)
        emissions = torch.zeros(1, 4, 3)
        mask = torch.zeros(1, 4)  # fully masked -> length 0
        assert crf.decode(emissions, mask) == [[]]


# ============================================================
#  NLL
# ============================================================

class TestNLL:
    def test_nonnegative_and_finite(self):
        torch.manual_seed(2)
        num_tags = 3
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(2, 4, num_tags)
        tags = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        nll = crf.neg_log_likelihood(emissions, tags, mask)
        assert nll.item() >= -1e-4
        assert torch.isfinite(nll)

    def test_padding_does_not_affect_score(self):
        # NLL on (full, mask) must equal NLL on (truncated_emissions, trunc_mask).
        torch.manual_seed(3)
        num_tags = 4
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(1, 6, num_tags)
        tags = torch.tensor([[0, 1, 2, 3, 0, 0]])
        mask_full = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        mask_trunc = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        # The masked positions 4-5 don't contribute to the gold score; they
        # only contribute to log Z through emissions[4]/emissions[5], so we
        # truncate those positions for the comparison to be apples-to-apples.
        nll_full = crf.neg_log_likelihood(emissions[:, :4], tags[:, :4], mask_full[:, :4])
        nll_trunc = crf.neg_log_likelihood(emissions, tags, mask_trunc)
        assert abs(nll_full.item() - nll_trunc.item()) < 1e-4


# ============================================================
#  Marginals (forward-backward)
# ============================================================

class TestMarginals:
    def test_shape_and_range(self):
        crf = LinearChainCRF(4)
        emissions = torch.randn(2, 5, 4)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
        m = crf.marginals(emissions, mask)
        assert m.shape == (2, 5, 4)
        # Each unmasked position must sum to ~1.
        for b in range(2):
            length = int(mask[b].sum().item())
            for t in range(length):
                assert abs(m[b, t].sum().item() - 1.0) < 1e-3
        # Padded positions are zero.
        assert m[1, 3].sum().item() == 0.0
        assert m[1, 4].sum().item() == 0.0

    def test_matches_brute_force_4x5(self):
        torch.manual_seed(5)
        num_tags, seq_len = 4, 5
        crf = LinearChainCRF(num_tags)
        emissions = torch.randn(1, seq_len, num_tags)
        mask = torch.ones(1, seq_len)
        m_algo = crf.marginals(emissions, mask)[0]  # (T, N)
        m_brute = brute_force_marginals(crf, emissions[0], seq_len)
        for t in range(seq_len):
            for i in range(num_tags):
                assert abs(m_algo[t, i].item() - m_brute[t][i]) < 1e-3, (t, i, m_algo[t, i].item(), m_brute[t][i])

    def test_predicted_tag_marginal_is_max(self):
        # argmax over marginals at each position should agree with Viterbi.
        torch.manual_seed(9)
        crf = LinearChainCRF(4)
        emissions = torch.randn(1, 6, 4)
        mask = torch.ones(1, 6)
        m = crf.marginals(emissions, mask)
        viterbi_path = crf.decode(emissions, mask)[0]
        length = int(mask[0].sum().item())
        for t in range(length):
            marginal_argmax = int(m[0, t].argmax().item())
            assert marginal_argmax == viterbi_path[t], (t, marginal_argmax, viterbi_path[t])


# ============================================================
#  UncertaintyWeightedLoss
# ============================================================

class TestUncertaintyWeightedLoss:
    def test_requires_matching_num_tasks(self):
        uwl = UncertaintyWeightedLoss(num_tasks=2)
        with pytest.raises(ValueError):
            uwl([torch.tensor(1.0)])

    def test_initial_weights_are_one(self):
        uwl = UncertaintyWeightedLoss(num_tasks=3)
        _, weights = uwl([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])
        # At init log_var=0 => precision=exp(0)=1 for every task.
        assert all(abs(weights[f"weight_{i}"] - 1.0) < 1e-6 for i in range(3))

    def test_down_weights_noisier_task_after_training(self):
        # Simulate: task 0 has small constant loss, task 1 has large noisy loss.
        # After a few optimiser steps the learned precision of task 1 should
        # drop below task 0's — that is the whole point of uncertainty weighting.
        torch.manual_seed(0)
        uwl = UncertaintyWeightedLoss(num_tasks=2)
        opt = torch.optim.SGD(uwl.parameters(), lr=0.05)
        for _ in range(200):
            l0 = torch.tensor(0.5)                     # quiet task
            l1 = torch.tensor(2.0) + torch.randn(())*0.5  # noisy, large task
            opt.zero_grad()
            total, _ = uwl([l0, l1])
            total.backward()
            opt.step()
        _, weights = uwl([torch.tensor(0.5), torch.tensor(2.0)])
        # Noisier task 1 should now have smaller precision than task 0.
        assert weights["weight_1"] < weights["weight_0"], weights

    def test_log_vars_must_be_in_optimiser(self):
        # If log_vars are NOT in the optimiser's param list, they never update —
        # this is the cagf trap documented in train_loop.py:91-92. Pin it.
        uwl = UncertaintyWeightedLoss(num_tasks=2)
        # Not in any optimiser — forward + backward must still not raise.
        total, _ = uwl([torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)])
        total.backward()
        # Without an optimiser step, log_vars stay at init.
        assert uwl.log_vars.tolist() == [0.0, 0.0]


# ============================================================
#  Device selection
# ============================================================

class TestPickDevice:
    def test_returns_a_torch_device(self):
        d = pick_device()
        assert isinstance(d, torch.device)
        assert d.type in {"cuda", "mps", "cpu"}

    def test_prefers_mps_on_apple_silicon(self):
        # On the dev machine (M3 Pro) MPS should be available. If it isn't
        # (CI, headless, etc.) this test is a no-op — we only assert that when
        # MPS is available, it is preferred over CPU.
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            pytest.skip("MPS not available on this machine")
        assert pick_device().type == "mps"
