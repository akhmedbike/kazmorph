#!/usr/bin/env python3
"""
Sensitivity analysis of the error-taxonomy threshold θ (paper Table 5).

Sweeps the diagnostic threshold θ ∈ [0.10, 0.50] and reports, for each value,
the resulting distribution of tokens across the five error classes
(BOR / AFC / SEG / AGR / NONE) on the dev and/or test splits. This is a
pure-Python computation (RMA + the deterministic taxonomy classifier) and
requires no trained models.

Why the distribution and not POS-accuracy? θ moves the *boundary* between the
specialised classes and the residual NONE class, so it changes the error-class
*labels* of tokens. POS accuracy itself is produced by the RMA/CRF/Neural
components and is independent of θ in the cascade and fallback modes; θ only
reaches POS through the (non-significant) routing mode. The meaningful,
always-applicable sensitivity question is therefore how stable the class
distribution is to θ — i.e. how robust the taxonomy's diagnostic picture is.

Usage:
    PYTHONPATH=. python scripts/sensitivity.py --dev data/dev.conllu --test data/test.conllu
    PYTHONPATH=. python scripts/sensitivity.py --grid 0.10,0.20,0.30,0.40,0.50 --out reports/sensitivity.md
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.rma import ReverseMorphAnalyzer
from kazmorph.taxonomy import classify_scored

CLASSES = ["BOR", "AFC", "SEG", "AGR", "NONE"]


def parse_conllu_tokens(path: str) -> list[str]:
    """Return the surface forms of all non-multiword tokens in a CoNLL-U file."""
    tokens: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            if "-" in parts[0] or "." in parts[0]:
                continue
            if len(parts) < 4:
                continue
            tokens.append(parts[1])
    return tokens


def distribution(tokens: list[str], rma: ReverseMorphAnalyzer, theta: float) -> dict[str, float]:
    """Class distribution (fractions summing to ~1) at a given θ."""
    counts = {c: 0 for c in CLASSES}
    for t in tokens:
        res = classify_scored(rma.analyze(t), threshold=theta)
        counts[res.difficulty_class] += 1
    n = max(1, len(tokens))
    return {c: counts[c] / n for c in CLASSES}


def render_table(rows: list[dict], split_name: str, default_theta: float) -> str:
    header = f"### {split_name}\n\n| θ | " + " | ".join(CLASSES) + " |"
    sep = "|---|" + "|".join(["---"] * len(CLASSES)) + "|"
    lines = [header, sep]
    for r in rows:
        mark = "\\*" if abs(r["theta"] - default_theta) < 1e-9 else ""
        cells = [f"{r['theta']:.2f}{mark}"] + [f"{r[c]*100:.1f}%" for c in CLASSES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", default=os.path.join(PROJECT_ROOT, "data", "dev.conllu"))
    p.add_argument("--test", default=os.path.join(PROJECT_ROOT, "data", "test.conllu"))
    p.add_argument("--grid", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
                   help="Comma-separated θ values to sweep")
    p.add_argument("--default", type=float, default=0.30, help="Default θ to highlight")
    p.add_argument("--out", default=None, help="Optional markdown file to write the tables to")
    args = p.parse_args()

    thetas = [float(x) for x in args.grid.split(",") if x.strip()]
    rma = ReverseMorphAnalyzer()

    blocks: list[str] = []
    for name, path in (("Development split", args.dev), ("Test split", args.test)):
        if not os.path.exists(path):
            print(f"(skip {name}: {path} not found)")
            continue
        tokens = parse_conllu_tokens(path)
        rows = [{"theta": th, **distribution(tokens, rma, th)} for th in thetas]
        blocks.append(render_table(rows, f"{name} (N = {len(tokens)} tokens)", args.default))

        # Variability summary: how much does each class move across the swept θ?
        print(f"\n{name} (N = {len(tokens)} tokens):")
        print(f"  {'θ':>5}  " + "  ".join(f"{c:>6}" for c in CLASSES))
        for r in rows:
            mark = "*" if abs(r["theta"] - args.default) < 1e-9 else " "
            print(f"  {r['theta']:>4.2f}{mark} " + "  ".join(f"{r[c]*100:5.1f}%" for c in CLASSES))
        for c in CLASSES:
            lo = min(r[c] for r in rows)
            hi = max(r[c] for r in rows)
            print(f"  → {c}: range {lo*100:.1f}% – {hi*100:.1f}% (Δ { (hi-lo)*100:.1f} pp)")

    out = "\n\n".join(blocks) + "\n"
    out += (f"\n\\* default θ = {args.default:.2f} used by the production pipeline. "
            "POS accuracy is independent of θ in the cascade and fallback modes; θ only "
            "reaches POS through the (non-significant) routing mode.\n")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Sensitivity of the error-taxonomy threshold θ\n\n" + out)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
