#!/usr/bin/env python3
"""
Train the surface error detector (kazmorph/correction/error_detector.py).

Data: gold train split only (data/train_split.conllu — no dev/test tokens).
Negatives: clean word tokens. Positives: the same tokens corrupted by the
same nine operators used at evaluation (corrupt.corrupt_token), 2 passes
with different seeds. Every corrupted variant keeps a matching clean twin in
the negative set, so the decision boundary is "is THIS surface form valid",
not "is this word frequent".

Features are extracted with the serving extractor (error_detector.
extract_features); the pipeline's CRF POS confidence is included as the
contextual feature by running the CRF pipeline over each sentence pair.

Output: models/error_detector.json (weights + bias + feature list + metrics)
and a console report: AUC, and the operating point that matters — detection
recall on corrupted tokens and false-flag rate on clean tokens at the τ_e
gate (0.15), overall and per corruption family.

Usage:
    PYTHONPATH=. python scripts/train_error_detector.py
    PYTHONPATH=. python scripts/train_error_detector.py --passes 3 --val-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import parse_conllu                       # noqa: E402
from kazmorph.correction.corrupt import corrupt_token, OP_IDENTITY  # noqa: E402
from kazmorph.correction.error_detector import extract_features, FEATURES  # noqa: E402
from kazmorph.pipeline import token_spans                      # noqa: E402
from kazmorph.pipeline import Pipeline                         # noqa: E402


def wordlike(form: str) -> bool:
    return len(form) >= 3 and any(c.isalpha() for c in form) and not any(c.isdigit() for c in form)


def build_dataset(args):
    sentences = parse_conllu(args.train)
    pipeline = Pipeline(crf_model_path="models/crf_model.joblib" if os.path.exists(
        "models/crf_model.joblib") else None)
    rng = random.Random(args.seed)

    rows = []  # (features, label, family)
    for pass_i in range(args.passes):
        for sent in sentences:
            forms = [t.form for t in sent.tokens]
            # One corrupted variant of the sentence per pass.
            idx_order = [i for i, t in enumerate(sent.tokens) if wordlike(t.form)]
            rng.shuffle(idx_order)
            corrupted = list(forms)
            target = None
            for i in idx_order:
                c = corrupt_token(forms[i], sent.tokens[i].upos, rng, pipeline.rma)
                if c is not None:
                    corrupted[i] = c.corrupted
                    target = (i, c.category)
                    break
            # Run the CRF pipeline once over the corrupted sentence for conf.
            text = " ".join(corrupted)
            spans = token_spans(text)
            results = pipeline.process(text) if spans else []
            conf_by_form = {}
            if len(results) == len(spans):
                for res, (_tok, s, e) in zip(results, spans):
                    conf_by_form[text[s:e]] = res.get("confidence", 1.0)
            # Negatives: clean wordlike tokens (downsampled for balance).
            for i, t in enumerate(sent.tokens):
                if not wordlike(t.form):
                    continue
                if i != (target[0] if target else -1) and rng.random() < args.clean_rate:
                    rows.append((extract_features(forms[i], conf_by_form.get(forms[i], 1.0)),
                                 0, "clean"))
            # Positive: the corrupted token.
            if target is not None:
                i, family = target
                rows.append((extract_features(corrupted[i], conf_by_form.get(corrupted[i], 1.0)),
                             1, family))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/train_split.conllu")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--clean-rate", type=float, default=0.25,
                    help="fraction of clean tokens sampled as negatives")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--tau-e", type=float, default=0.15)
    ap.add_argument("--out", default="models/error_detector.json")
    args = ap.parse_args()

    t0 = time.time()
    rows = build_dataset(args)
    n_pos = sum(1 for _, y, _ in rows if y == 1)
    print(f"dataset: {len(rows)} tokens ({n_pos} corrupted / {len(rows)-n_pos} clean), "
          f"{time.time()-t0:.0f}s")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X = [[r[0][f] for f in FEATURES] for r in rows]
    y = [r[1] for r in rows]
    X_tr, X_va, y_tr, y_va, rows_tr, rows_va = train_test_split(
        X, y, rows, test_size=args.val_frac, random_state=args.seed, stratify=y)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
    print(f"validation AUC: {auc:.4f}")

    # Operating point at the τ_e gate: detection recall / false-flag rate.
    def op_stats(rows_, probs):
        tp = fp = fn = tn = 0
        per_family = {}
        for (feats, label, family), p in zip(rows_, probs):
            flagged = p >= args.tau_e
            if label == 1:
                key = family
                d = per_family.setdefault(key, [0, 0])
                d[1] += 1
                if flagged:
                    tp += 1; d[0] += 1
                else:
                    fn += 1
            else:
                if flagged:
                    fp += 1
                else:
                    tn += 1
        rec = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        return rec, fpr, per_family

    probs_va = clf.predict_proba(X_va)[:, 1]
    rec, fpr, fam = op_stats(rows_va, probs_va)
    print(f"at tau_e={args.tau_e}: detection recall={rec:.3f}, clean false-flag={fpr:.3f}")
    for k in sorted(fam, key=lambda k: -fam[k][1]):
        hit, n = fam[k]
        print(f"    {k:32} {hit}/{n} = {hit/n:.2f}")

    weights = {f: float(w) for f, w in zip(FEATURES, clf.coef_[0])}
    payload = {
        "model": "logistic_regression_v1",
        "features": list(FEATURES),
        "weights": weights,
        "bias": float(clf.intercept_[0]),
        "metrics": {
            "val_auc": round(float(auc), 4),
            "val_detection_recall_at_tau_e": round(rec, 4),
            "val_clean_false_flag_at_tau_e": round(fpr, 4),
            "tau_e": args.tau_e,
            "n_train": len(y_tr), "n_val": len(y_va),
            "passes": args.passes, "clean_rate": args.clean_rate, "seed": args.seed,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")
    print("weights:", {k: round(v, 3) for k, v in weights.items()})


if __name__ == "__main__":
    main()
