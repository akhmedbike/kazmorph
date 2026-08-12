#!/usr/bin/env python3
"""
Fit confidence calibration for the neural POS tagger.

Two modes:

  * ``scalar`` (default) — one global temperature T that minimizes ECE on the
    dev split (Guo et al., 2017). argmax-invariant, so POS accuracy is unchanged.
  * ``region``            — Region-Dependent TS: one T per confidence bin, which
    can reduce ECE on rare / low-confidence classes a little further.

Writes ``models/neural_pos/calibration.json``, which NeuralTagger loads
automatically at startup.

Usage:
    python scripts/calibrate.py --model models/neural_pos --dev data/dev.conllu
    python scripts/calibrate.py --model models/neural_pos --dev data/dev.conllu --mode region

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
    p.add_argument("--mode", choices=["scalar", "region"], default="scalar",
                   help="scalar = one global T (default, lowest risk); "
                        "region = Region-Dependent TS (one T per confidence bin)")
    p.add_argument("--n-regions", type=int, default=3,
                   help="Number of confidence bins for --mode region")
    args = p.parse_args()

    if not os.path.exists(os.path.join(args.model, "config.json")):
        sys.exit(f"error: model not found: {args.model}")
    if not os.path.exists(args.dev):
        sys.exit(f"error: dev file not found: {args.dev}")

    print(f"Fitting {args.mode} calibration on {args.dev} ...")
    payload = calibrate(args.model, args.dev, n_bins=args.bins,
                        mode=args.mode, n_regions=args.n_regions)

    print(f"\nCalibration result ({args.mode}):")
    print(f"  ECE before (T=1): {payload['ece_before']:.4f}  ({payload['ece_before']*100:.1f}% gap)")
    print(f"  ECE after       : {payload['ece_after']:.4f}  ({payload['ece_after']*100:.1f}% gap)")
    if args.mode == "scalar":
        print(f"  Temperature T   : {payload['temperature']:.2f}")
    else:
        print(f"  Regions:")
        for reg in payload["regions"]:
            print(f"    conf ∈ ({reg['lo']:.2f}, {reg['hi']:.2f}]  →  T = {reg['temperature']:.2f}")
    print(f"\nSaved to {args.model}/calibration.json")
    print(f"NeuralTagger will load it automatically on next startup.")


if __name__ == "__main__":
    main()
