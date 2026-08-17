#!/usr/bin/env python3
"""
Train CRF model for Kazakh POS tagging on UD Kazakh-KTB.

Architecture (from paper): RMA features → CRF disambiguation.
Uses the full feature set (~15 features) from kazmorph.crf_tagger.

Usage:
    python3 scripts/train_crf.py [--dev-split 0.15 --seed 42]
"""

import os
import sys
import json
import random
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from kazmorph.rma import ReverseMorphAnalyzer, RMAResult
from kazmorph.crf_tagger import CRFTagger


# ---------------------------------------------------------------------------
# CoNLL-U parser
# ---------------------------------------------------------------------------

def parse_conllu(filepath: str) -> list[list[dict]]:
    """Parse CoNLL-U file into list of sentences, each a list of token dicts."""
    sentences = []
    current = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments
            if line.startswith("#"):
                continue

            # Empty line = sentence boundary
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            # Skip multiword tokens (e.g. "12-13")
            parts = line.split("\t")
            if "-" in parts[0]:
                continue

            if len(parts) >= 4:
                current.append({
                    "id": parts[0],
                    "form": parts[1],
                    "lemma": parts[2],
                    "upos": parts[3],
                    "xpos": parts[4] if len(parts) > 4 else "_",
                })

    if current:
        sentences.append(current)

    return sentences


# ---------------------------------------------------------------------------
# Feature extraction: RMA → features
# ---------------------------------------------------------------------------

def sentence_to_features(
    sentence: list[dict],
    rma: ReverseMorphAnalyzer,
) -> tuple[list[dict], list[str]]:
    """
    Run RMA on each token in a sentence, then extract CRF features.
    Returns (features_list, labels_list).
    """
    # Stage 1: Run RMA on all tokens
    rma_results = [rma.analyze(tok["form"]) for tok in sentence]

    # Stage 2: Extract features using CRFTagger's feature extractor
    features = []
    labels = []

    for i, (tok, rma_res) in enumerate(zip(sentence, rma_results)):
        feat = CRFTagger.extract_features(rma_results, i)
        features.append(feat)
        labels.append(tok["upos"])

    return features, labels


def prepare_data(
    sentences: list[list[dict]],
    rma: ReverseMorphAnalyzer,
) -> tuple[list[list[dict]], list[list[str]]]:
    """Prepare full dataset for CRF training."""
    X, y = [], []
    for sent in sentences:
        feats, labels = sentence_to_features(sent, rma)
        X.append(feats)
        y.append(labels)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

UPOS_LABELS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "NUM",
    "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "AUX",
    "PROPN", "PUNCT", "SYM", "X",
]


def print_report(y_true, y_pred, label_set=None):
    """Print classification report."""
    if label_set is None:
        label_set = sorted(set(l for seq in y_true for l in seq)
                           | set(l for seq in y_pred for l in seq))

    # Filter out labels with no support
    label_set = [l for l in label_set if l in set(ll for seq in y_true for ll in seq)]

    from sklearn.metrics import precision_recall_fscore_support
    from collections import Counter

    # Flatten for sklearn metrics
    y_true_flat = [l for seq in y_true for l in seq]
    y_pred_flat = [l for seq in y_pred for l in seq]

    print("\n=== CRF POS Tagging Report ===")
    print(f"Accuracy:  {metrics.flat_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:  {metrics.flat_f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {metrics.flat_f1_score(y_true, y_pred, average='weighted'):.4f}")

    support = Counter(y_true_flat)

    # Per-label metrics via sklearn
    labels_sorted = sorted(label_set)
    p, r, f, s = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, labels=labels_sorted, zero_division=0
    )

    # Sort by F1 descending
    per_label = list(zip(labels_sorted, p, r, f, s))
    per_label.sort(key=lambda x: x[3], reverse=True)

    print("\nPer-label P/R/F1:")
    print(f"{'Label':>8s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>5s}  {'Support':>7s}")
    for label, prec, rec, f1, sup in per_label:
        print(f"{label:>8s}  {prec:9.4f}  {rec:7.4f}  {f1:5.4f}  {int(sup):7d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CRF POS tagger for Kazakh")
    parser.add_argument("--dev-split", type=float, default=0.15,
                        help="Dev split ratio from train (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--train-file", default=os.path.join(PROJECT_ROOT, "data", "train.conllu"),
        help="Path to train CoNLL-U (default: data/train.conllu — the canonical "
             "70/15/15 split produced by scripts/make_splits.py)",
    )
    parser.add_argument(
        "--dev-file", default=os.path.join(PROJECT_ROOT, "data", "dev.conllu"),
        help="Path to dev CoNLL-U (default: data/dev.conllu)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Paths
    train_path = args.train_file
    dev_path = args.dev_file
    model_dir = os.path.join(PROJECT_ROOT, "models")
    model_path = os.path.join(model_dir, "crf_model.joblib")

    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print(f"Loading corpus: {train_path}")
    train_sents = parse_conllu(train_path)
    dev_sents = parse_conllu(dev_path)
    print(f"  Train: {len(train_sents)} sentences")
    print(f"  Dev:   {len(dev_sents)} sentences")

    # Initialize RMA
    print("\nInitializing RMA...")
    rma = ReverseMorphAnalyzer()

    # Extract features
    print("Extracting features...")
    X_train, y_train = prepare_data(train_sents, rma)
    X_dev, y_dev = prepare_data(dev_sents, rma)

    print(f"  Train tokens: {sum(len(s) for s in X_train)}")
    print(f"  Dev tokens:   {sum(len(s) for s in X_dev)}")

    # Train CRF with hyperparameter search
    print("\nTraining CRF with hyperparameter search...")

    best_f1 = 0.0
    best_params = {}
    best_model = None

    param_grid = [
        {"c1": 0.01, "c2": 0.01},
        {"c1": 0.01, "c2": 0.1},
        {"c1": 0.05, "c2": 0.01},
        {"c1": 0.05, "c2": 0.05},
        {"c1": 0.1,  "c2": 0.01},
        {"c1": 0.1,  "c2": 0.1},
        {"c1": 0.2,  "c2": 0.01},
        {"c1": 0.5,  "c2": 0.01},
    ]

    for params in param_grid:
        crf_trial = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=params["c1"],
            c2=params["c2"],
            max_iterations=150,
            all_possible_transitions=True,
            all_possible_states=True,
        )
        crf_trial.fit(X_train, y_train)

        y_pred = crf_trial.predict(X_dev)
        f1 = metrics.flat_f1_score(y_dev, y_pred, average="weighted")
        acc = metrics.flat_accuracy_score(y_dev, y_pred)

        marker = " *best*" if f1 > best_f1 else ""
        print(f"  c1={params['c1']:<4} c2={params['c2']:<4} → "
              f"acc={acc:.4f}  w-F1={f1:.4f}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = crf_trial

    crf = best_model
    print(f"\n  Best: c1={best_params['c1']}, c2={best_params['c2']}, "
          f"weighted-F1={best_f1:.4f}")

    # Persist the selected hyperparameters so the paper's "Materials and Methods"
    # section can cite the exact configuration that produced the released model.
    hyper_path = os.path.join(model_dir, "crf_hyperparams.json")
    with open(hyper_path, "w", encoding="utf-8") as f:
        json.dump({
            "algorithm": "lbfgs",
            "c1": best_params["c1"],
            "c2": best_params["c2"],
            "max_iterations": 150,
            "all_possible_transitions": True,
            "all_possible_states": True,
            "dev_weighted_f1": best_f1,
            "grid": param_grid,
            "train_file": os.path.relpath(args.train_file, PROJECT_ROOT),
            "dev_file": os.path.relpath(args.dev_file, PROJECT_ROOT),
            "seed": args.seed,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Hyperparameters written to {hyper_path}")

    # Evaluate on dev
    print("\n--- Dev set evaluation ---")
    y_dev_pred = crf.predict(X_dev)
    print_report(y_dev, y_dev_pred)

    # Save model
    print(f"\nSaving model to {model_path}...")
    joblib.dump(crf, model_path)
    print(f"Model saved ({os.path.getsize(model_path)} bytes)")

    # Quick sanity check: load and predict
    print("\nSanity check: loading model and predicting...")
    loaded = joblib.load(model_path)
    sample_pred = loaded.predict(X_dev[:1])
    print(f"  Sample prediction: {sample_pred[0][:5]}")
    print(f"  Expected:          {y_dev[0][:5]}")

    print("\nDone! Use the model with:")
    print(f"  pipeline = Pipeline(crf_model_path='{model_path}')")


if __name__ == "__main__":
    main()
