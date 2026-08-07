"""
Linear-chain CRF for sequence labelling.

Ported from cagf_cbt_crf (cagf/crf.py) with one addition: ``marginals()``, a
forward-backward routine that returns per-token marginal probabilities. The
original cagf module implements only the forward algorithm (for the NLL loss)
and Viterbi decoding; here we also need marginals so the joint POS+lemma model
can report an honest per-token confidence for the predicted POS tag (the CRF
equivalent of ``softmax(logits).max()`` in a non-structured classifier).

Why a CRF head on top of a BERT tagger
--------------------------------------
The standard neural POS tagger takes per-token logits from a transformer and
argmaxes them independently. That throws away the strong sequential prior that
exists in POS sequences (a NOUN is much more likely to follow a DET than
another NOUN, etc.). A linear-chain CRF adds trainable ``num_tags x num_tags``
transition scores and decodes the globally-best tag sequence with Viterbi,
which on morphologically-rich languages with sparse tags typically gains
0.5–1.5 points of POS F1 — especially on rare classes (SCONJ, INTJ, SYM) where
the per-token classifier is least reliable. This is the Lample et al. (2016)
recipe ("Neural Architectures for Named Entity Recognition") applied here to
POS over a KazBERT backbone.

What this module provides
------------------------
- ``neg_log_likelihood(emissions, tags, mask)`` — the training loss.
- ``decode(emissions, mask)`` — Viterbi decoding for inference.
- ``marginals(emissions, mask)`` — per-token marginal probability of the
  Viterbi-predicted tag, via forward-backward. Used as a calibrated
  confidence that is more honest than a post-hoc softmax temperature.

Tensor shape conventions (B = batch, T = seq_len, N = num_tags):
- ``emissions``: (B, T, N), real-valued scores (e.g. BERT logits).
- ``tags``:     (B, T), long, valid ids in [0, N) at every position. Use
                ``safe_tags[~mask] = 0`` to clamp padded positions (the score
                contribution is zeroed by the mask, but the index must be
                in-range or ``transitions[tags]`` will IndexError).
- ``mask``:     (B, T), float or bool, ``1``/True for real tokens. Padding
                MUST be right-aligned (real tokens first, then pads), because
                ``_score_sentence`` finds the last real tag via
                ``mask.sum(dim=1)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain conditional random field with start/end transitions.

    Parameters
    ----------
    num_tags
        Number of distinct tags. Determines the shape of ``transitions``.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] = score of moving from tag i to tag j.
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        # start_transitions[i] = score of starting the sequence at tag i.
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        # end_transitions[i] = score of ending the sequence at tag i.
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        # Small uniform init keeps the early gradients well-conditioned.
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    # ------------------------------------------------------------------
    # Forward and backward recursions
    # ------------------------------------------------------------------

    def _score_sentence(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Score of the gold tag sequence under the model. Returns (B,)."""
        batch_size, seq_len, _ = emissions.shape
        # First timestep: start transition + emission at position 0.
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for t in range(1, seq_len):
            emit = emissions[torch.arange(batch_size), t, tags[:, t]]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            # mask zeros out the contribution of padded positions.
            score = score + (trans + emit) * mask[:, t]
        # Add the end-transition score at the last real position.
        seq_lens = mask.sum(dim=1).long()
        last_tags = tags[torch.arange(batch_size), seq_lens - 1]
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log Z (partition function). Returns (B,).

        alpha[t, i] = log P(tags_0..t = some path ending at tag i, observations).
        Computed via the numerically-stable logsumexp recurrence.

        Note on the end transition: this returns
        ``logsumexp_i(alpha[L-1, i] + end_transitions[i])`` per batch element,
        i.e. the end transition is attached ONLY to the last real position.
        This convention is shared with ``_forward_algorithm_full`` and
        ``_backward_algorithm`` so the forward-backward product
        ``alpha[t,i] + beta[t,i] - log Z`` gives correct marginals.
        """
        alpha = self._forward_algorithm_full(emissions, mask.float())
        seq_lens = mask.sum(dim=1).long()
        log_z = torch.full(
            (alpha.shape[0],), float("-inf"),
            device=alpha.device, dtype=alpha.dtype,
        )
        for b in range(alpha.shape[0]):
            L = int(seq_lens[b].item())
            if L > 0:
                # end_transitions live only at the last real position; the
                # partition function sums that out over the final-step tag.
                last_step = alpha[b, L - 1, :] + self.end_transitions
                log_z[b] = torch.logsumexp(last_step, dim=0)
        return log_z

    def _backward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log beta. Returns (B, T, N).

        beta[t, i] = log P(observations_{t+1..T-1} | tag at position t is i).
        Computed right-to-left via logsumexp. This is the symmetric partner of
        ``_forward_algorithm`` and is what makes marginals tractable.
        """
        batch_size, seq_len, num_tags = emissions.shape
        seq_lens = mask.sum(dim=1).long()                  # (B,)
        # beta at the last real position = end_transitions (everything after is empty).
        beta = torch.full(
            (batch_size, seq_len, num_tags),
            float("-inf"),
            device=emissions.device,
            dtype=emissions.dtype,
        )
        # transitions[i, j] = score of moving from tag i to tag j.
        # For each "from" tag i we sum (over j) the contribution of going to
        # j, emitting at j, and continuing from j: that is a (N_from, N_to)
        # grid reduced along dim=1. We do this in a per-batch loop because the
        # sequence lengths vary within the batch.
        trans = self.transitions.detach()                  # (N_from, N_to)
        for b in range(batch_size):
            L = int(seq_lens[b].item())
            if L == 0:
                continue
            beta[b, L - 1, :] = self.end_transitions.detach()
            for t in range(L - 2, -1, -1):
                emit = emissions[b, t + 1]                 # (N_to,)
                next_beta_from = beta[b, t + 1, :]         # (N_to,)
                # (N_from, N_to): for each from-tag i, sum over to-tag j of
                # trans[i,j] + emit[j] + beta[t+1, j].
                grid = trans + (emit + next_beta_from).unsqueeze(0)
                beta[b, t, :] = torch.logsumexp(grid, dim=1)
        return beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def neg_log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean per-batch NLL = log Z - log P(gold). Returns a scalar."""
        gold_score = self._score_sentence(emissions, tags, mask.float())
        log_z = self._forward_algorithm(emissions, mask.float())
        return (log_z - gold_score).mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding. Returns a Python list of per-sequence tag-id lists.

        Each inner list has length equal to the unmasked prefix length of the
        corresponding batch element (so lengths may vary within the batch).
        Zero-length sequences yield an empty list.
        """
        batch_size, seq_len, num_tags = emissions.shape
        mask_bool = mask.bool()
        history: list[torch.Tensor] = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)            # (B, N_from, 1)
            trans = self.transitions.unsqueeze(0)           # (1, N_from, N_to)
            next_score = broadcast_score + trans
            best_score, best_idx = next_score.max(dim=1)    # (B, N_to), (B, N_to)
            best_score = best_score + emissions[:, t]
            step_mask = mask_bool[:, t].unsqueeze(1)
            score = torch.where(step_mask, best_score, score)
            history.append(best_idx)
        score = score + self.end_transitions.unsqueeze(0)
        seq_lens = mask_bool.sum(dim=1)
        best_paths: list[list[int]] = []
        for b in range(batch_size):
            length = int(seq_lens[b].item())
            if length == 0:
                best_paths.append([])
                continue
            _, best_last = score[b].max(dim=0)
            best_last = int(best_last.item())
            path = [best_last]
            for t in range(length - 2, -1, -1):
                best_last = int(history[t][b, best_last].item())
                path.append(best_last)
            path.reverse()
            best_paths.append(path)
        return best_paths

    @torch.no_grad()
    def marginals(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-token marginal probability of each tag. Returns (B, T, N).

        marginal[t, i] = P(tag_t = i | full observation sequence), computed by
        the forward-backward algorithm as
        ``log_marginal[t, i] = alpha[t, i] + beta[t, i] - log Z``,
        then exponentiated. Padded positions are returned as 0.

        This is the calibrated-replacement for ``softmax(logits/T).max()`` in
        the non-structured POS tagger: the joint tagger reads
        ``marginals(...)[b, t, predicted_tag]`` as the per-token confidence
        shown in the UI. Unlike a temperature-scaled softmax, this probability
        is consistent with the CRF's own global decoding.
        """
        mask_f = mask.float()
        alpha = self._forward_algorithm_full(emissions, mask_f)  # (B, T, N)
        beta = self._backward_algorithm(emissions, mask_f)        # (B, T, N)
        log_z = self._forward_algorithm(emissions, mask_f)        # (B,)
        log_marginal = alpha + beta - log_z.unsqueeze(1).unsqueeze(2)
        marginal = torch.exp(log_marginal)
        # Zero out padded positions (alpha is -inf there, so exp is already 0,
        # but the explicit mask makes the intent clear and guards against any
        # numerical noise leaking in from the beta init).
        marginal = marginal * mask_f.unsqueeze(2)
        return marginal

    def _forward_algorithm_full(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Full alpha tensor (B, T, N), not just log Z. Companion to marginals().

        Unlike ``_forward_algorithm`` (which returns only the final logsumexp
        over the last real position), this keeps every per-step alpha so the
        forward-backward product can be formed.

        Importantly, alpha as returned here does NOT include end_transitions:
        end_transitions are folded in only inside ``_forward_algorithm`` (for
        log Z) and ``_backward_algorithm`` (for beta at L-1). This split keeps
        the marginal formula ``alpha[t,i] + beta[t,i] - log Z`` from
        double-counting end_transitions at the final position.

        Padded positions are left as -inf.
        """
        batch_size, seq_len, num_tags = emissions.shape
        alpha = torch.full(
            (batch_size, seq_len, num_tags),
            float("-inf"),
            device=emissions.device,
            dtype=emissions.dtype,
        )
        alpha[:, 0, :] = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            broadcast_alpha = alpha[:, t - 1, :].unsqueeze(2)
            next_alpha = torch.logsumexp(broadcast_alpha + trans + emit, dim=1)
            step_mask = mask[:, t].unsqueeze(1)
            alpha[:, t, :] = torch.where(step_mask.bool(), next_alpha, alpha[:, t, :])
        return alpha
