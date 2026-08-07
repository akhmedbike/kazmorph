"""
Multi-task loss weighting by learned task uncertainty.

Ported from cagf_cbt_crf (cagf/losses.py), which implements Kendall, Gal, and
Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018).

The idea: instead of fixing the relative weights of two task losses by hand
(e.g. ``0.5 * pos_loss + 0.5 * lemma_loss``), give each task a learnable
log-variance ``s_i`` and minimise

    total = sum_i [ exp(-s_i) * loss_i + s_i ]

The first term precision-weights each task's loss; the second is a regulariser
that prevents the network from driving every ``s_i`` to +inf to zero out the
loss. At the optimum, noisier tasks end up with smaller precision
``exp(-s_i)`` — i.e. they are automatically down-weighted. The ``s_i`` are
updated by the same optimiser as the rest of the model, so they must be added
to the optimiser's parameter group (see the joint trainer).

We use this in the joint POS+lemma model (scripts/train_joint.py) to balance
the CRF negative-log-likelihood for POS against the cross-entropy for the
lemma prefix length, which live on very different scales.

Reference
---------
Kendall, A., Gal, Y., Cipolla, R. (2018). "Multi-Task Learning Using
Uncertainty to Weigh Losses." CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """Combine N scalar task losses with learnable per-task precision weights.

    Parameters
    ----------
    num_tasks
        Number of task losses that will be passed to ``forward``. Must match
        the length of the list passed to ``forward`` exactly.
    init_log_var
        Initial value of each ``log_var`` parameter. ``0.0`` (the default)
        means each task starts with precision ``exp(0) = 1.0`` — i.e. all
        tasks weighted equally at the start of training, matching the
        convention in cagf.

    Notes
    -----
    The ``log_vars`` parameter MUST be added to whatever optimiser trains the
    rest of the model, otherwise it never updates and the weighting is a
    no-op. The joint trainer wires this up as
    ``optim = AdamW(list(model.parameters()) + list(loss_weighter.parameters()), ...)``.
    """

    def __init__(self, num_tasks: int = 2, init_log_var: float = 0.0):
        super().__init__()
        self.num_tasks = num_tasks
        # One learnable scalar per task. Registered as a Parameter so it moves
        # with .to(device), .half(), etc. and is saved/loaded with state_dict.
        self.log_vars = nn.Parameter(torch.full((num_tasks,), float(init_log_var)))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Combine the per-task losses into a single scalar.

        Parameters
        ----------
        losses
            List of scalar loss tensors, length must equal ``num_tasks``.

        Returns
        -------
        total
            A scalar tensor: the precision-weighted sum plus the log-variance
            regulariser. This is what gets ``.backward()`` called on it.
        weights
            ``{"weight_0": float, "weight_1": float, ...}`` reporting the
            current precision ``exp(-log_var_i)`` of each task. Useful for
            logging which task is being up- or down-weighted as training
            progresses. The keys are positional (matching cagf) — map them to
            task names at the call site if you want readable logs.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"expected {self.num_tasks} losses, got {len(losses)}"
            )
        total = torch.zeros((), device=losses[0].device)
        weights: dict[str, float] = {}
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
            weights[f"weight_{i}"] = float(precision.item())
        return total, weights
