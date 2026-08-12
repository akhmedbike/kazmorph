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


@dataclass
class RegionCalibrationResult:
    """Region-Dependent Temperature Scaling (RD-TS).

    Instead of a single global T, the confidence range [0, 1] is split into
    ``n_regions`` equal-width bins and one temperature is fitted per bin by
    ECE minimization. At inference the temperature applied to a token is
    chosen by its *uncalibrated* max-probability. RD-TS can reduce ECE on
    rare/low-confidence classes slightly beyond scalar TS (the research
    review's recommended refinement), at no accuracy cost — argmax is still
    invariant within each region. Scalar TS remains the default.
    """
    regions: list[dict]   # [{"lo": float, "hi": float, "temperature": float}]
    ece_before: float
    ece_after: float
    n_bins: int
    n_regions: int


# Default scalar-T grid: fine 0.8–5.0 step 0.1. ECE is a smooth 1-D objective,
# so a fine grid + the argmin is both robust and cheap on the dev split.
DEFAULT_T_GRID: list[float] = [round(0.8 + 0.1 * i, 1) for i in range(43)]


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


def _best_temperature(logits_t: torch.Tensor, correct: list[bool], n_bins: int,
                      t_grid: list[float]) -> tuple[float, float, float]:
    """Return (best_T, best_ECE, ECE_at_T=1) for a subset of tokens."""
    conf_t1 = torch.softmax(logits_t, dim=-1).max(dim=-1).values.tolist()
    ece_t1 = _ece(conf_t1, correct, n_bins)
    best_t, best_ece = 1.0, ece_t1
    for t in t_grid:
        conf = torch.softmax(logits_t / t, dim=-1).max(dim=-1).values.tolist()
        ece = _ece(conf, correct, n_bins)
        if ece < best_ece:
            best_t, best_ece = t, ece
    return best_t, best_ece, ece_t1


def fit_temperature(
    logits_list: list[torch.Tensor],
    gold: list[int],
    pred: list[int],
    *,
    n_bins: int = 10,
    t_grid: list[float] | None = None,
) -> CalibrationResult:
    """
    Find the single temperature T that minimizes ECE.

    argmax is invariant to T (softmax(logits/T).argmax == softmax(logits).argmax),
    so `pred` doesn't change — only the confidence does. We grid-search T over
    a fine range because ECE is non-convex but smooth and one-dimensional.
    """
    if t_grid is None:
        t_grid = DEFAULT_T_GRID

    logits = torch.stack(logits_list)  # (N, num_labels)
    pred_t = torch.tensor(pred)
    gold_t = torch.tensor(gold)
    correct = (pred_t == gold_t).tolist()

    best_t, best_ece, ece_before = _best_temperature(logits, correct, n_bins, t_grid)
    return CalibrationResult(temperature=best_t, ece_before=ece_before,
                             ece_after=best_ece, n_bins=n_bins)


def fit_region_temperature(
    logits_list: list[torch.Tensor],
    gold: list[int],
    pred: list[int],
    *,
    n_regions: int = 3,
    n_bins: int = 10,
    t_grid: list[float] | None = None,
) -> RegionCalibrationResult:
    """
    Region-Dependent Temperature Scaling (RD-TS).

    Split the tokens into ``n_regions`` equal-width bins by their *uncalibrated*
    max softmax probability, fit one temperature per bin by ECE minimization on
    that bin's tokens, and report the resulting global ECE after per-region
    rescaling. Membership is decided by the uncalibrated confidence, so the
    region selection at inference matches what was fitted here.
    """
    if t_grid is None:
        t_grid = DEFAULT_T_GRID

    logits = torch.stack(logits_list)
    correct = (torch.tensor(pred) == torch.tensor(gold)).tolist()
    raw_conf = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
    ece_before = _ece(raw_conf, correct, n_bins)

    edges = [i / n_regions for i in range(n_regions + 1)]
    regions: list[dict] = []
    calibrated_conf = [0.0] * len(correct)
    for ri in range(n_regions):
        lo, hi = edges[ri], edges[ri + 1]
        idx = [j for j, c in enumerate(raw_conf)
               if (lo < c <= hi) or (ri == 0 and c <= lo + 1e-12)]
        if not idx:
            regions.append({"lo": lo, "hi": hi, "temperature": 1.0})
            continue
        sub_logits = logits[idx]
        sub_correct = [correct[j] for j in idx]
        best_t, _, _ = _best_temperature(sub_logits, sub_correct, n_bins, t_grid)
        sub_conf = torch.softmax(sub_logits / best_t, dim=-1).max(dim=-1).values.tolist()
        for k, j in enumerate(idx):
            calibrated_conf[j] = sub_conf[k]
        regions.append({"lo": lo, "hi": hi, "temperature": best_t})

    ece_after = _ece(calibrated_conf, correct, n_bins)
    return RegionCalibrationResult(regions=regions, ece_before=ece_before,
                                   ece_after=ece_after, n_bins=n_bins,
                                   n_regions=n_regions)


def calibrate(
    model_path: str,
    conllu_path: str,
    *,
    n_bins: int = 10,
    mode: str = "scalar",
    n_regions: int = 3,
) -> dict:
    """
    Fit calibration on a CoNLL-U dev set and save it into the model directory.

    ``mode="scalar"`` (default) fits one global temperature T. ``mode="region"``
    fits Region-Dependent TS (one T per confidence bin). Returns the dict that
    was written to ``calibration.json``; ``NeuralTagger`` loads it automatically.
    """
    logits, gold, pred = collect_logits(model_path, conllu_path)
    out_path = os.path.join(model_path, "calibration.json")

    if mode == "region":
        r = fit_region_temperature(logits, gold, pred, n_regions=n_regions, n_bins=n_bins)
        payload = {
            "mode": "region",
            "regions": r.regions,
            "ece_before": r.ece_before,
            "ece_after": r.ece_after,
            "n_regions": r.n_regions,
        }
    else:
        r = fit_temperature(logits, gold, pred, n_bins=n_bins)
        payload = {
            "mode": "scalar",
            "temperature": r.temperature,
            "ece_before": r.ece_before,
            "ece_after": r.ece_after,
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
