#!/usr/bin/env python3
"""Select the aligned-protocol winners (best dev.conllu weighted-F1 per encoder)
from reports/baseline_sweep_aligned/ and promote them to the canonical model
paths used by evaluate.py / benchmark.py / the Pipeline.

Old-protocol checkpoints are preserved under <name>_foldsplit_old/ so the
previous numbers remain reproducible.

Usage:  python3 scripts/promote_aligned.py
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "reports", "baseline_sweep_aligned")
MODELS = os.path.join(ROOT, "models")

PLAN = {  # sweep tag prefix -> canonical model dir
    "kazbert": "neural_pos",
    "xlmr": "neural_pos_xlmr",
    "kazr": "neural_pos_kazr",
}


def candidates(family: str) -> list[tuple[float, str]]:
    out = []
    for f in sorted(os.listdir(SWEEP)):
        if f.startswith(f"{family}_lr") and f.endswith("_fold.json"):
            lr = f[len(family) + 1:-len("_fold.json")]  # e.g. lr3e5
            out.append((json.load(open(os.path.join(SWEEP, f)))["best_f1"], lr))
    return sorted(out, reverse=True)


def main() -> None:
    for family, canonical in PLAN.items():
        cands = candidates(family)
        if not cands:
            sys.exit(f"error: no sweep results found for {family} in {SWEEP}")
        best_f1, best_lr = cands[0]
        print(f"{family:8s} winner {best_lr}  dev.conllu F1={best_f1:.4f}   "
              f"(all: {', '.join(f'{lr}={f1:.4f}' for f1, lr in cands)})")

        src = os.path.join(MODELS, "sweep_aligned", f"{family}_{best_lr}", "fold_1")
        if not os.path.exists(os.path.join(src, "config.json")):
            sys.exit(f"error: winner checkpoint missing: {src}")

        dst = os.path.join(MODELS, canonical)
        if os.path.islink(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            bak = os.path.join(MODELS, f"{canonical}_foldsplit_old")
            if os.path.exists(bak):
                shutil.rmtree(bak)
            shutil.move(dst, bak)
            print(f"         old checkpoint kept at models/{canonical}_foldsplit_old/")

        shutil.copytree(src, dst)
        model_root = os.path.dirname(src)
        for extra in ("tag_map.json", "cv_results.json"):
            s = os.path.join(model_root, extra)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, extra))

        json.dump({"winner": f"{family}_{best_lr}", "dev_f1": best_f1,
                   "candidates": [[lr, f1] for f1, lr in cands]},
                  open(os.path.join(SWEEP, f"{family}_winner.json"), "w"), indent=2)

    print("\nPromoted. Next: scripts/calibrate.py --model <path> for each of the "
          "three canonical dirs, then evaluate.py --config all.")


if __name__ == "__main__":
    main()
