"""
Unified device selection for training and inference.

Single source of truth for "which torch device should we use on this machine".
The order is deliberate:

  1. CUDA — fastest, the obvious first choice where available.
  2. MPS  — Apple Silicon GPU. On an M3 Pro 18GB, KazBERT-base fine-tunes in
     ~30-60 min per fold at batch_size 8, vs hours on CPU. This is why MPS
     sits above CPU in the preference order, not below.
  3. CPU  — always available; the safe fallback.

The previous trainers in this repo hard-coded
``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` which silently
fell back to CPU on every Apple Silicon machine, including the development
MacBook this project targets. Replacing that line with ``pick_device()`` is the
one-line change that turns hours-long CPU runs into GPU runs locally.

CPU intra-op thread count is intentionally NOT capped here: the existing
trainers cap it only in the CPU branch (see scripts/train_neural.py:274-276),
and that policy stays where it is. This module only answers the device
question.
"""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Return the best available torch device: cuda > mps > cpu.

    Checks are runtime and re-evaluated on every call, so the result reflects
    the current process state (e.g. if a downstream caller disabled MPS via
    ``torch.backends.mps.is_available = False`` for debugging).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
