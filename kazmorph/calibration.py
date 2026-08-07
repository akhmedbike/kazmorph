"""
Temperature scaling for neural POS confidence calibration.

The raw softmax of a fine-tuned BERT is systematically over-confident: it
assigns probability ~1.0 to most predictions even when it is wrong ~15% of
the time. Temperature scaling divides the logits by a single scalar T > 1
before softmax, which flattens the distribution so that the reported
confidence matches the empirical accuracy (Guo et al., 2017, "On Calibration
of Modern Neural Networks").

This module fits T on a held-out set by minimizing Expected Calibration Error
(ECE): bin predictions by confidence, and in each bin compare the average
confidence to the observed accuracy. T is a monotone scalar — it never changes
the argmax, so POS/lemma accuracy is unaffected; only the confidence number
becomes honest.

No extra model is trained; only one parameter is fitted, so this stays stable
on small data (the 1607-token dev split is plenty for one scalar).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from kazmorph.corpus import POS_TAGS, parse_conllu, stratified_kfold


# BIO tagging, mirroring scripts/train_neural.py
TAG2ID = {"O": 0}
for _i, _t in enumerate(POS_TAGS):
    TAG2ID[f"B-{_t}"] = _i + 1


@dataclass
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float
    n_bins: int


def _ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted avg of |acc(conf) - avg conf| per bin."""
    if not confidences:
        return 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin is inclusive of 1.0
        in_bin = [j for j, c in enumerate(confidences)
                  if (lo < c <= hi) or (i == 0 and c == lo)]
        if not in_bin:
            continue
        bin_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        bin_acc = sum(1 for j in in_bin if correct[j]) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)
    return ece


def collect_logits(
    model_path: str,
    conllu_path: str,
    *,
    max_len: int = 128,
):
    """
    Run the model over a CoNLL-U file and collect, for every gold token, the
    (max_logit, predicted_index, correct?) triple. Returns lists aligned to
    tokens. Used by fit_temperature.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    sentences = parse_conllu(conllu_path)
    # per-token, the predicted-subword logits vector + the gold tag index
    all_logits: list[torch.Tensor] = []
    all_gold: list[int] = []
    all_pred: list[int] = []

    for sent in sentences:
        words = [t.form for t in sent.tokens]
        gold_tags = [t.upos for t in sent.tokens]
        if not words:
            continue
        enc = tokenizer(words, is_split_into_words=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        word_ids = enc.word_ids(batch_index=0)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

        seen: set[int] = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            gold_tag = gold_tags[wid]
            gold_bio = f"B-{gold_tag}" if gold_tag in POS_TAGS else "O"
            all_logits.append(logits[idx].detach().cpu())
            all_gold.append(TAG2ID.get(gold_bio, 0))
            all_pred.append(int(logits[idx].argmax().item()))

    return all_logits, all_gold, all_pred


def fit_temperature(
    logits_list: list[torch.Tensor],
    gold: list[int],
    pred: list[int],
    *,
    n_bins: int = 10,
    t_grid: list[float] | None = None,
) -> CalibrationResult:
    """
    Find the temperature T that minimizes ECE.

    argmax is invariant to T (softmax(logits/T).argmax == softmax(logits).argmax),
    so `pred` doesn't change — only the confidence does. We grid-search T over
    a reasonable range because ECE is non-convex but smooth and one-dimensional.
    """
    if t_grid is None:
        t_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    logits = torch.stack(logits_list)  # (N, num_labels)
    pred_t = torch.tensor(pred)
    gold_t = torch.tensor(gold)
    correct = (pred_t == gold_t).tolist()

    # baseline ECE at T=1
    conf_t1 = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
    ece_before = _ece(conf_t1, correct, n_bins)

    best_t, best_ece = 1.0, ece_before
    for t in t_grid:
        conf = torch.softmax(logits / t, dim=-1).max(dim=-1).values.tolist()
        ece = _ece(conf, correct, n_bins)
        if ece < best_ece:
            best_t, best_ece = t, ece

    return CalibrationResult(temperature=best_t, ece_before=ece_before,
                             ece_after=best_ece, n_bins=n_bins)


def calibrate(model_path: str, conllu_path: str, *, n_bins: int = 10) -> CalibrationResult:
    """Fit T on a CoNLL-U dev set and save it into the model directory."""
    logits, gold, pred = collect_logits(model_path, conllu_path)
    result = fit_temperature(logits, gold, pred, n_bins=n_bins)

    out_path = os.path.join(model_path, "calibration.json")
    with open(out_path, "w") as f:
        json.dump({"temperature": result.temperature,
                   "ece_before": result.ece_before,
                   "ece_after": result.ece_after}, f, indent=2)
    return result
