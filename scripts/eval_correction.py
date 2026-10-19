#!/usr/bin/env python3
"""
Correction-subsystem evaluation: candidate reach + three-way end metrics.

Measures, on the held-out dev split with controlled corruption
(:mod:`kazmorph.correction.corrupt`, 9 operators, corrupt_prob=0.5):

  1. Candidate reach per mode (dl / dl_freq / dl_fst_filter / dl_freq_morph):
     Recall@1 / @5 / @10 and MRR — is the gold surface form among the
     generated candidates, and at which rank? Pipeline-independent.

  2. Three-way end metrics per mode (KEEP/CORRECT/REVIEW at the default
     thresholds): precision / recall / F0.5 on corrupted tokens, and the
     false-correction rate (FCR) on clean tokens. This is where detection
     gating (p_err = 1 - POS confidence) bites: a corrupted token that the
     tagger reads confidently is KEEPed before candidates are ever consulted.

  3. A small threshold grid (tau_c x tau_m at fixed tau_e) for the two
     morphology-aware modes, reporting the F0.5-best point.

GEC counting convention (as in correction_metrics.correction_prf): a wrong
CORRECT on a corrupted token counts as both FP and FN; any CORRECT on a clean
token is an FP. REVIEW and KEEP never change the surface form.

Protocol notes:
  * dev.conllu only — data/test.conllu stays untouched (grant POS metrics).
  * Seeds 11/22/33; corruption is applied AFTER loading, per seed.
  * Sentences whose tokenisation round-trip fails are skipped and counted.

Usage:
    PYTHONPATH=. python scripts/eval_correction.py --tag baseline
    PYTHONPATH=. python scripts/eval_correction.py --fast --tag quick   # CRF only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import parse_conllu                     # noqa: E402
from kazmorph.correction.corrupt import corrupt_fold         # noqa: E402
from kazmorph.correction.candidate_generator import (        # noqa: E402
    generate_candidates, candidate_probs,
)
from kazmorph.correction.corrector_3way import (        # noqa: E402
    TAU_E, TAU_C, TAU_M, ORIGINAL_EVIDENCE_ALPHA, DECISION_ENERGY_SCALE,
)
from kazmorph.correction.error_detector import estimate_p_err  # noqa: E402
from kazmorph.pipeline import Pipeline                       # noqa: E402

MODES = ["dl", "dl_freq", "dl_fst_filter", "dl_freq_morph"]
DEFAULT_SEEDS = [11, 22, 33]
GRID_TAU_E = [TAU_E, 0.25, 0.40]   # detection gate: flagged share grows with τ_e
GRID_TAU_C = [0.10, 0.15, 0.20, 0.30]
GRID_TAU_M = [0.00, 0.02, 0.05, 0.10]
GRID_SCALE = [1.0, 2.0, 4.0, 8.0]  # decision temperature (energy scaling)


def build_pipeline(fast: bool) -> Pipeline:
    """Full pipeline (CRF+Neural+Lemma) or CRF-only for quick iterations."""
    crf = "models/crf_model.joblib" if os.path.exists("models/crf_model.joblib") else None
    if fast or not os.path.exists("models/neural_pos/config.json"):
        return Pipeline(crf_model_path=crf, neural_model_path=None, lemma_model_path=None)
    lemma = "models/neural_lemma" if os.path.exists("models/neural_lemma/lemma_meta.json") else None
    return Pipeline(crf_model_path=crf, neural_model_path="models/neural_pos", lemma_model_path=lemma)


def decide(profile: dict, tau_e: float, tau_c: float, tau_m: float,
           energy_scale: float = 1.0) -> tuple[str, str]:
    """Faithful re-implementation of corrector_3way._classify_token over a
    cached per-token profile {token, pos, confidence, p_err, cands}. The
    softmax is recomputed per call so the energy scale is sweepable."""
    token = profile["token"]
    pos = profile.get("pos", "UNK")
    p_err = profile["p_err"]
    if p_err < tau_e or pos in ("PUNCT", "NUM", "SYM") or len(token) < 2:
        return "KEEP", token
    cands = profile["cands"]
    if not cands:
        # Gate crossed but nothing retrieved: suspicious with no options.
        return "REVIEW", token
    probs = candidate_probs(
        cands, include_original=True, original_token=token,
        original_energy=ORIGINAL_EVIDENCE_ALPHA * p_err,
        energy_scale=energy_scale,
    )
    original_lower = token.lower()
    correction_probs = [(w, p) for (w, p) in probs if w.lower() != original_lower]
    if not correction_probs:
        return "KEEP", token
    top_word, top_prob = correction_probs[0]
    remaining = [(w, p) for (w, p) in probs if w.lower() != top_word.lower()]
    second_prob = remaining[0][1] if remaining else 0.0
    margin = top_prob - second_prob
    if top_prob >= tau_c and margin >= tau_m:
        return "CORRECT", top_word
    return "REVIEW", token


def evaluate(args) -> dict:
    sentences = parse_conllu(args.dev)
    pipeline = build_pipeline(args.fast)

    # Per-mode accumulators.
    cand_ranks: dict[str, list[int]] = defaultdict(list)      # 1-based ranks, 0 = miss
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    fc = defaultdict(int); n_clean = defaultdict(int)         # clean-token FCR parts
    n_review = defaultdict(int); n_corrupted = defaultdict(int)
    skipped = 0
    # Cached profiles for the grid sweep: (mode -> list of profile dicts).
    grid_profiles: dict[str, list[dict]] = defaultdict(list)

    t0 = time.time()
    for seed in args.seeds:
        pairs = corrupt_fold(sentences, seed)
        for pair in pairs:
            text = " ".join(pair.corrupted_forms)
            results = pipeline.process(text)
            if len(results) != len(pair.corrupted_forms):
                skipped += 1
                continue
            cpos = pair.corruption.position if pair.corruption else None
            gold = pair.corruption.original.lower() if pair.corruption else None

            for i, res in enumerate(results):
                token = res["token"]
                pos = res.get("pos", "UNK")
                conf = float(res.get("confidence", 0.0))
                p_err = estimate_p_err(token, conf)
                is_clean = i != cpos
                # The real corrector consults candidates only after the τ_e
                # gate; mirror that (also the main perf lever — candidate
                # generation with morph gates costs ~ms per token).
                gate_open = not (p_err < TAU_E or pos in ("PUNCT", "NUM", "SYM")
                                 or len(token) < 2)
                # Candidate reach needs candidates on the corrupted position
                # regardless of the gate (reach measures generation, not
                # detection).
                need_cands = gate_open or (i == cpos)
                probs_by_mode: dict[str, list] = {}
                for mode in MODES:
                    cands = generate_candidates(token, mode=mode) if need_cands else []
                    profile = {"token": token, "pos": pos, "confidence": conf,
                               "p_err": p_err, "cands": cands}
                    # ── candidate reach (corrupted position only) ──
                    if i == cpos and gold is not None:
                        words = [w.lower() for w, _ in cands]
                        rank = words.index(gold) + 1 if gold in words else 0
                        cand_ranks[mode].append(rank)
                        n_corrupted[mode] += 1
                    # ── three-way decision at default thresholds ──
                    decision, predicted = decide(profile, TAU_E, TAU_C, TAU_M,
                                                  DECISION_ENERGY_SCALE)
                    if decision == "CORRECT":
                        if is_clean:
                            fp[mode] += 1; fc[mode] += 1
                        elif predicted.lower() == gold:
                            tp[mode] += 1
                        else:
                            fp[mode] += 1; fn[mode] += 1  # wrong correction = fp+fn
                    elif decision == "REVIEW":
                        n_review[mode] += 1
                        if not is_clean:
                            fn[mode] += 1
                    else:  # KEEP
                        if not is_clean:
                            fn[mode] += 1
                    if is_clean:
                        n_clean[mode] += 1
                    probs_by_mode[mode] = profile
                if args.grid and need_cands:
                    # Grid profiles include IDENTITY sentences too: false
                    # corrections on clean sentences are part of precision.
                    # (cpos None → every token counts as clean.)
                    for mode in ("dl_fst_filter", "dl_freq_morph"):
                        grid_profiles[mode].append(
                            {"i": i, "cpos": cpos, "gold": gold, **probs_by_mode[mode]})

    # ── aggregate ──
    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f05 = (1 + 0.25) * p * r / (0.25 * p + r) if p + r else 0.0
        return round(p, 4), round(r, 4), round(f05, 4)

    modes_json = {}
    for mode in MODES:
        ranks = cand_ranks[mode]
        r1 = sum(1 for x in ranks if 1 <= x <= 1) / len(ranks) if ranks else 0.0
        r5 = sum(1 for x in ranks if 1 <= x <= 5) / len(ranks) if ranks else 0.0
        r10 = sum(1 for x in ranks if 1 <= x <= 10) / len(ranks) if ranks else 0.0
        mrr = sum(1.0 / x for x in ranks if x) / len(ranks) if ranks else 0.0
        p, r, f05 = prf(tp[mode], fp[mode], fn[mode])
        modes_json[mode] = {
            "n_corrupted": n_corrupted[mode],
            "recall_at_1": round(r1, 4), "recall_at_5": round(r5, 4),
            "recall_at_10": round(r10, 4), "mrr": round(mrr, 4),
            "precision": p, "recall": r, "f05": f05,
            "fcr_clean": round(fc[mode] / n_clean[mode], 5) if n_clean[mode] else None,
            "review_rate": round(n_review[mode] / max(n_clean[mode] + n_corrupted[mode], 1), 4),
            "n_tp": tp[mode], "n_fp": fp[mode], "n_fn": fn[mode], "n_fc": fc[mode],
        }

    grid_json = {}
    if args.grid:
        for mode, profiles in grid_profiles.items():
            best = None
            rows = []
            for scale in GRID_SCALE:
                for tau_e in GRID_TAU_E:
                    for tau_c in GRID_TAU_C:
                        for tau_m in GRID_TAU_M:
                            g_tp = g_fp = g_fn = g_fc = 0
                            for pr in profiles:
                                d, pred = decide(pr, tau_e, tau_c, tau_m, scale)
                                clean = pr["i"] != pr["cpos"]
                                if d == "CORRECT":
                                    if clean:
                                        g_fp += 1; g_fc += 1
                                    elif pred.lower() == pr["gold"]:
                                        g_tp += 1
                                    else:
                                        g_fp += 1; g_fn += 1
                                elif not clean:
                                    g_fn += 1
                            p, r, f05 = prf(g_tp, g_fp, g_fn)
                            row = {"scale": scale, "tau_e": tau_e, "tau_c": tau_c,
                                   "tau_m": tau_m, "precision": p, "recall": r,
                                   "f05": f05, "n_fc": g_fc}
                            rows.append(row)
                            if best is None or f05 > best["f05"]:
                                best = row
            grid_json[mode] = {"best": best,
                               "rows": sorted(rows, key=lambda x: -x["f05"])[:5]}

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dev": args.dev, "seeds": args.seeds, "fast": args.fast,
            "n_sentences": len(sentences), "skipped_tokenization": skipped,
            "defaults": {"tau_e": TAU_E, "tau_c": TAU_C, "tau_m": TAU_M},
            "wall_seconds": round(time.time() - t0, 1),
        },
        "modes": modes_json,
        "grid_best": grid_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default="data/dev.conllu")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--fast", action="store_true", help="CRF-only pipeline (no neural)")
    ap.add_argument("--grid", action="store_true", help="tau_c x tau_m sweep")
    ap.add_argument("--tag", default="", help="filename tag: correction_eval_<tag>.json")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    report = evaluate(args)
    os.makedirs(args.out_dir, exist_ok=True)
    name = f"correction_eval{'_' + args.tag if args.tag else ''}.json"
    path = os.path.join(args.out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n== {path} ==")
    hdr = f"{'mode':16} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'P':>6} {'R':>6} {'F05':>6} {'FCR':>7} {'REV%':>6}"
    print(hdr); print("-" * len(hdr))
    for mode, m in report["modes"].items():
        fcr = f"{m['fcr_clean']*100:.2f}%" if m["fcr_clean"] is not None else "—"
        print(f"{mode:16} {m['recall_at_1']:6.3f} {m['recall_at_5']:6.3f} "
              f"{m['recall_at_10']:6.3f} {m['mrr']:6.3f} {m['precision']:6.3f} "
              f"{m['recall']:6.3f} {m['f05']:6.3f} {fcr:>7} {m['review_rate']*100:5.1f}%")
    for mode, res in (report.get("grid_best") or {}).items():
        b = res["best"]
        print(f"grid best {mode}: scale={b['scale']} tau_e={b['tau_e']} "
              f"tau_c={b['tau_c']} tau_m={b['tau_m']} "
              f"P={b['precision']} R={b['recall']} F05={b['f05']} fc={b['n_fc']}")
    print(f"n_corrupted/mode={report['modes']['dl']['n_corrupted']}, "
          f"skipped={report['meta']['skipped_tokenization']}, "
          f"wall={report['meta']['wall_seconds']}s")


if __name__ == "__main__":
    main()
