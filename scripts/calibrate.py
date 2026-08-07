#!/usr/bin/env python3
"""
Fit confidence calibration temperature for the neural POS tagger.

Reads a held-out CoNLL-U dev set, runs the model, and finds the scalar
temperature T that makes softmax confidence match empirical accuracy
(min Expected Calibration Error). Writes T into models/neural_pos/calibration.json,
which NeuralTagger loads automatically at startup.

Usage:
    python scripts/calibrate.py --model models/neural_pos --dev data/dev.conllu

IMPORTANT on the dev file: use ``data/dev.conllu`` (the clean post-decontamination
split from 2026-08-02), NOT the legacy ``data/dev_split.conllu``. The latter
predates the silver↔test leak fix and overlaps the training data, which makes
the model look ~3 pp more accurate than it is and yields a too-low temperature
that leaves confidence over-optimistic — the exact failure this calibration is
meant to correct.
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.calibration import calibrate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(PROJECT_ROOT, "models", "neural_pos"),
                   help="Path to the fine-tuned POS model directory")
    p.add_argument("--dev", default=os.path.join(PROJECT_ROOT, "data", "dev.conllu"),
                   help="Held-out CoNLL-U to fit T on (use data/dev.conllu, the clean "
                        "post-decontamination split — NOT legacy data/dev_split.conllu)")
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()

    if not os.path.exists(os.path.join(args.model, "config.json")):
        sys.exit(f"error: model not found: {args.model}")
    if not os.path.exists(args.dev):
        sys.exit(f"error: dev file not found: {args.dev}")

    print(f"Fitting temperature on {args.dev} ...")
    r = calibrate(args.model, args.dev, n_bins=args.bins)
    print(f"\nCalibration result:")
    print(f"  Temperature T = {r.temperature:.2f}")
    print(f"  ECE before (T=1): {r.ece_before:.4f}  ({r.ece_before*100:.1f}% gap)")
    print(f"  ECE after  (best): {r.ece_after:.4f}  ({r.ece_after*100:.1f}% gap)")
    print(f"\nSaved to {args.model}/calibration.json")
    print(f"NeuralTagger will load T automatically on next startup.")


if __name__ == "__main__":
    main()
