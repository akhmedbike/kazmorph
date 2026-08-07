#!/usr/bin/env python3
"""
Train a KazBERT + linear-chain-CRF POS tagger (Stage 3, CRF variant).

This is the A/B control for the joint multi-task model (scripts/train_joint.py).
The joint model adds a CRF head AND a lemma head; this script adds the CRF head
alone. Comparing the two against the plain softmax POS tagger
(scripts/train_neural.py) on the same k-fold splits is what tells you whether
the CRF head alone (this script), the joint multitask objective alone, or the
combination is responsible for any POS-F1 gain.

Architecture
------------
KazBERT (Eraly-ml/KazBERT) produces per-token emissions (B, T, num_tags).
A linear-chain CRF head (kazmorph.crf.LinearChainCRF) layers trainable
start/end/transition scores on top and decodes globally with Viterbi. Training
minimises the CRF negative log-likelihood instead of per-token cross-entropy.
The Lample et al. (2016) recipe applied to POS over a KazBERT backbone.

What is shared with scripts/train_neural.py
-------------------------------------------
- The KazBERT backbone choice and the BIO POS label set (O + 17 B-<UPOS>).
- The k-fold CV protocol via kazmorph.corpus.stratified_kfold.
- The silver-data plumbing (POSTokenDataset, _SilverAwareCollator) — but
  silver-weighting interacts with CRF NLL, not CE, so this script applies the
  weight directly to per-token NLL contributions (see _weighted_crf_nll).
- pick_device() — MPS on Apple Silicon, CUDA elsewhere, CPU fallback.

Output
------
models/neural_pos_crf/  (config.json, model.safetensors, tokenizer files,
tag_map.json, cv_results.json). The pipeline does NOT load this directory by
default; it stays a separate experimental artifact until you point
Pipeline(..., joint_model_path=...) or a future NeuralTagger at it.

Usage
-----
    # 5-fold CV, KazBERT + CRF, MPS auto-selected on Apple Silicon
    python scripts/train_neural_crf.py --folds 5 --epochs 12

    # Quick smoke test (does not save anything meaningful)
    python scripts/train_neural_crf.py --folds 1 --epochs 2 --limit 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import POS_TAGS, Sentence, load_pool, stratified_kfold
from kazmorph.crf import LinearChainCRF
from kazmorph.device import pick_device

# Reuse the dataset / collator / silver-weighted CE helpers from the softmax
# trainer. The dataset's job (subword alignment, silver_mask) is identical
# regardless of the head; only the loss differs here.
from scripts.train_neural import (
    POSTokenDataset,
    _SilverAwareCollator,
    TAG2ID,
    ID2TAG,
    NUM_LABELS,
)

BASE_MODEL = "Eraly-ml/KazBERT"
MAX_LEN = 128


# ============================================================
#  Model: KazBERT encoder + CRF head
# ============================================================
class BertCrfTagger(nn.Module):
    """KazBERT backbone producing per-token emissions, plus a CRF head.

    The BERT logits serve as CRF emissions; the CRF adds trainable transition
    scores and decodes with Viterbi. We don't use
    AutoModelForTokenClassification because we want our own CRF on top of the
    backbone's per-token hidden states, not HF's internal classifier head.
    """

    def __init__(self, base_model: str, num_tags: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.bert.config, "hidden_dropout_prob", 0.1))
        self.emission_head = nn.Linear(hidden, num_tags)
        self.crf = LinearChainCRF(num_tags)

    def emissions(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.emission_head(self.dropout(out.last_hidden_state))

    def loss(self, input_ids, attention_mask, labels, silver_mask, silver_weight):
        emissions = self.emissions(input_ids, attention_mask)
        return _weighted_crf_nll(
            self.crf, emissions, labels, silver_mask, silver_weight
        )

    @torch.no_grad()
    def decode(self, input_ids, attention_mask) -> list[list[int]]:
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask)


def _weighted_crf_nll(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
    labels: torch.Tensor,
    silver_mask: torch.Tensor,
    silver_weight: float,
) -> torch.Tensor:
    """CRF NLL with per-token silver/gold weighting.

    Standard CRF NLL = log Z - log P(gold sequence), reduced as a single
    scalar per batch element. That scalar form is inherently sequence-level —
    it can't be trivially split into per-token contributions the way CE can.
    So we approximate silver down-weighting at the SEQUENCE level: a batch
    element's full NLL is scaled by the silver-weighted MEAN of its token
    weights, then we average across the batch. Sequences that are mostly
    silver contribute proportionally less.

    This is the right granularity for the silver problem (Apertium sentences
    are entirely silver, gold sentences entirely gold — token-level weighting
    within a mixed sentence basically never happens in our data). It is NOT
    identical to per-token CE silver-weighting in scripts/train_neural.py;
    the two trainers operate at different structural levels and that's fine.
    """
    # CRF mask: real-token = 1.0, pad/ignored = 0.0. Derived from labels.
    crf_mask = (labels != -100).float()
    # Clamp labels into [0, num_tags) at masked positions — required by the
    # CRF (otherwise transitions[labels] IndexErrors). Mirrors the cagf
    # pattern at cagf/model.py:103-104.
    safe_labels = labels.clone()
    safe_labels[crf_mask == 0] = 0
    # Per-batch NLL is a scalar each.
    nll_per_seq = crf._forward_algorithm(emissions, crf_mask) - \
                  crf._score_sentence(emissions, safe_labels, crf_mask)
    # Per-sequence silver fraction = mean silver_mask over real tokens.
    seq_real = crf_mask.sum(dim=1).clamp(min=1.0)
    seq_silver_frac = (silver_mask * crf_mask).sum(dim=1) / seq_real
    seq_weight = (1.0 - seq_silver_frac) * 1.0 + seq_silver_frac * silver_weight
    return (nll_per_seq * seq_weight).sum() / seq_weight.sum().clamp(min=1e-6)


# ============================================================
#  Single fold train + eval
# ============================================================
def run_fold(
    base_model: str,
    train_sents: Sequence[Sentence],
    dev_sents: Sequence[Sentence],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
    out_dir: str | None,
    fold_label: str,
    n_gold_train: int | None = None,
    silver_weight: float = 1.0,
) -> tuple[float, dict]:
    print(f"\n[{fold_label}] Loading {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = BertCrfTagger(base_model, num_tags=NUM_LABELS).to(device)

    if n_gold_train is None:
        n_gold_train = len(train_sents)
    train_ds = POSTokenDataset(train_sents, tokenizer, TAG2ID, n_gold=n_gold_train)
    dev_ds = POSTokenDataset(dev_sents, tokenizer, TAG2ID)
    collator = _SilverAwareCollator(tokenizer=tokenizer, label_pad_token_id=-100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0
    history = []
    print(f"[{fold_label}] Train: {len(train_sents)} sents "
          f"(gold={n_gold_train}, silver={len(train_sents) - n_gold_train}, "
          f"silver_weight={silver_weight}) | Dev: {len(dev_sents)} sents | epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            silver_mask = batch["silver_mask"].to(device)
            loss = model.loss(input_ids, attention_mask, labels, silver_mask, silver_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        # eval: Viterbi-decode and compare to gold BIO tags via seqeval.
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                pred_ids = model.decode(input_ids, attention_mask)
                for i in range(labels.shape[0]):
                    pred_seq, label_seq = [], []
                    pred_i = pred_ids[i]
                    pi = 0
                    for j in range(labels.shape[1]):
                        if labels[i, j].item() != -100:
                            label_seq.append(ID2TAG[labels[i, j].item()])
                            if pi < len(pred_i):
                                pred_seq.append(ID2TAG[pred_i[pi]])
                                pi += 1
                            else:
                                pred_seq.append("O")
                    all_preds.append(pred_seq)
                    all_labels.append(label_seq)
        dev_f1 = f1_score(all_labels, all_preds, average="weighted")
        marker = "  *best*" if dev_f1 > best_f1 else ""
        print(f"[{fold_label}] Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"dev_f1={dev_f1:.4f}{marker}")
        history.append({"epoch": epoch + 1, "loss": avg_loss, "dev_f1": dev_f1})
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                # Save backbone + heads as a single module state_dict, plus
                # tag_map for the consumer. We can't use save_pretrained on the
                # BertCrfTagger wrapper, so we serialize BERT via its own API
                # and our CRF/head weights separately.
                model.bert.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
                torch.save(
                    {"emission_head": model.emission_head.state_dict(),
                     "crf": model.crf.state_dict()},
                    os.path.join(out_dir, "crf_head.pt"),
                )

    return best_f1, {"history": history, "last_report": (all_labels, all_preds)}


# ============================================================
#  Main
# ============================================================
def train(args):
    device = pick_device()
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        print(f"CPU intra-op threads: {torch.get_num_threads()}")
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")

    pool = load_pool(args.data if args.data else None, dedup=True)
    if args.limit:
        pool = pool[: args.limit]
    print(f"Gold pool (deduplicated): {len(pool)} sentences")

    silver_sents: list[Sentence] = []
    if args.silver:
        from kazmorph.corpus import parse_conllu
        for sf in args.silver:
            silver_sents += parse_conllu(sf)
        print(f"Silver data: {len(silver_sents)} sentences "
              f"(silver_weight={args.silver_weight}) "
              f"from {len(args.silver)} file(s)")

    folds = stratified_kfold(pool, k=args.folds, seed=args.seed)
    print(f"Cross-validation: {len(folds)} folds")

    fold_f1s: list[float] = []
    last_fold_report = None
    for i, (train_gold, dev) in enumerate(folds):
        fold_label = f"fold {i+1}/{len(folds)}"
        train_sents = list(train_gold) + silver_sents
        n_gold_train = len(train_gold)
        out_dir = args.model_dir if i == len(folds) - 1 else None
        best_f1, details = run_fold(
            args.base_model,
            train_sents,
            dev,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            device=device,
            out_dir=out_dir,
            fold_label=fold_label,
            n_gold_train=n_gold_train,
            silver_weight=args.silver_weight,
        )
        fold_f1s.append(best_f1)
        if i == len(folds) - 1:
            last_fold_report = details["last_report"]

    # Summary
    print("\n" + "=" * 60)
    print("  CRF POS TAGGER — CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, f1 in enumerate(fold_f1s):
        print(f"  Fold {i+1}: best dev weighted-F1 = {f1:.4f}")
    mean_f1 = sum(fold_f1s) / len(fold_f1s)
    if len(fold_f1s) > 1:
        std = (sum((f - mean_f1) ** 2 for f in fold_f1s) / len(fold_f1s)) ** 0.5
        print(f"  Mean ± std: {mean_f1:.4f} ± {std:.4f}")
    else:
        std = 0.0
        print(f"  Mean: {mean_f1:.4f}")
    print("=" * 60)

    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "tag_map.json"), "w") as f:
        json.dump({"tag2id": TAG2ID,
                   "id2tag": {str(k): v for k, v in ID2TAG.items()},
                   "pos_tags": POS_TAGS, "base_model": args.base_model}, f, indent=2)
    with open(os.path.join(args.model_dir, "cv_results.json"), "w") as f:
        json.dump({"base_model": args.base_model, "folds": args.folds,
                   "epochs": args.epochs, "fold_f1": fold_f1s,
                   "mean_f1": mean_f1, "std_f1": std,
                   "head": "linear_chain_crf"}, f, indent=2)
    print(f"\nSaved to {args.model_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train KazBERT + CRF POS tagger")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--silver", nargs="*", default=None)
    p.add_argument("--silver-weight", type=float, default=0.3,
                   help="Per-sequence loss weight applied to silver sentences")
    p.add_argument("--data", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="Use only the first N gold sentences (smoke testing)")
    p.add_argument("--model-dir",
                   default=os.path.join(PROJECT_ROOT, "models", "neural_pos_crf"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    train(p.parse_args())
