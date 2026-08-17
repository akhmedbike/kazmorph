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

import os
import random

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


def enable_determinism(seed: int = 42) -> None:
    """Seed every RNG source for reproducible runs.

    Sets ``PYTHONHASHSEED`` (for the process, via ``os.environ`` — best-effort,
    it only takes full effect for processes started with it set), Python's
    ``random``, NumPy, and PyTorch (CPU + CUDA). MPS has no deterministic-mode
    flag in current PyTorch, so MPS runs may still show minor non-determinism;
    call this at the very start of a script to minimise drift.

    Each backend is imported lazily inside a try/except, so this function works
    on a CPU-only box without numpy installed (it silently skips absent
    backends rather than raising). Call it once at program start, before any
    model construction or data loading.

    Adapted from the IHCONCS_2026_paper project (MIT).
    """
    # PYTHONHASHSEED must be set before interpreter start to fully take effect;
    # setting it here is best-effort (helps subprocesses spawned later).
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
