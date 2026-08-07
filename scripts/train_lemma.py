#!/usr/bin/env python3
"""
Train a neural lemmatizer for Kazakh (Stage 3b).

Task formulation
----------------
Lemmatization is reframed as token classification: for each WORD, predict the
LENGTH OF THE LEMMATIC PREFIX (how many leading characters to keep). Given a
word "шабуылынан" whose lemma is "шабуыл" (6 chars), the target label is 6.
This turns a generative problem into a one-forward-pass sequence-labeling one,
identical in structure to POS tagging — so it stays fast on CPU (critical for
the real-time web UI) and reuses the KazBERT checkpoint we already fine-tuned
for POS.

Stem alternations (қ↔ғ, к↔г devoicing/voicing, ~6% of errors) are NOT learned
by the network alone — the head only sees prefix length. They are handled by a
small deterministic post-processing step in the NeuralLemmatizer (voice/devoice
the final character of the predicted prefix). This split (network learns depth,
rules handle alternation) matches where each method is strong.

Data
----
- Gold: UD Kazakh-KTB (deduplicated pool, 1078 sentences) via kazmorph.corpus.
- Silver: data/silver_lemmas.tsv produced by kazmorph.lemma_silver (~9000
  wordform→lemma pairs from Apertium). Silver triples the training vocabulary,
  which is what lets the model generalize beyond the tiny UD corpus.

Output: models/neural_lemma/  (config.json + model + tokenizer + meta JSON).

Usage
-----
    python scripts/train_lemma.py --folds 5 --epochs 12
    python scripts/train_lemma.py --folds 1 --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import Sentence, load_pool, stratified_kfold

# Reuse the PRODUCTION lemma decoder so that the dev metrics computed here match
# exactly what the pipeline produces at inference time. Previously this script
# carried its own _apply_alternation with an UNCONDITIONAL devoicing rule — the
# same regression that was fixed in neural_lemma._repair_lemma (commit c4b93db:
# өзі→өс, көзі→көс). Training/eval under the buggy decoder selected "best"
# checkpoints against a metric that did not match production behaviour. Importing
# _repair_lemma removes that divergence.
from kazmorph.neural_lemma import _repair_lemma


BASE_MODEL = "Eraly-ml/KazBERT"
MAX_LEN = 128
# Label set: prefix lengths 0..MAX_PREFIX. A label k means "keep first k chars".
# 0 is a sentinel for "no change / unknown" (used for punctuation, numbers, and
# words whose lemma is the whole word — those are handled by the fallback that
# when predicted prefix >= len(word), lemma == word). We cap at 24 since 99% of
# Kazakh lemmas are shorter (the corpus max lemma length is ~18).
MAX_PREFIX = 24
NUM_LABELS = MAX_PREFIX + 1  # labels 0..24

# Kazakh final-consonant voicing/devoicing pairs (voiced → voiceless). Used by
# prefix_label to detect a stem-alternation pair at the divergence point and
# bump the cut by +1 so the voiced consonant is included in the kept prefix
# (the production decoder _repair_lemma, imported above, then devoices it back).
# NOTE: the actual lemma decoding is delegated to neural_lemma._repair_lemma so
# that training/eval metrics match production exactly. _VOWELS is no longer
# needed here (it lived only in the removed _apply_alternation).
_DEVOICE = {"б": "п", "г": "к", "ғ": "қ", "д": "т", "ж": "ш", "з": "с"}


# ============================================================
#  Data
# ============================================================
def gold_lemma_pairs(sentences: Sequence[Sentence]) -> dict[str, str]:
    """{form_lower: lemma_lower} from gold UD sentences."""
    out: {}
    out = {}
    for s in sentences:
        for tok in s.tokens:
            out.setdefault(tok.form.lower(), tok.lemma.lower())
    return out


def load_silver_lemmas(path: str | None) -> dict[str, str]:
    """{form_lower: lemma_lower} from a silver TSV (form<TAB>lemma<TAB>upos)."""
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                out.setdefault(parts[0], parts[1])
    return out


def prefix_label(form: str, lemma: str) -> int:
    """
    Target label = number of leading chars of `form` to KEEP as the lemma,
    before the devoicing repair in _apply_alternation fires. Capped at MAX_PREFIX.

    Default is the longest common prefix (LCP) of (form, lemma). For non-
    alternating words this is exactly right: адамның→адам keeps 4 chars.

    STEM ALTERNATION BUMP (+1): when form and lemma differ ONLY by a
    voicing/devoicing pair at position LCP (form[LCP] voiced, lemma[LCP]
    its voiceless counterpart — e.g. бастығы/бастық where ғ↔қ), we return
    LCP+1 instead of LCP. The reason: the production decoder _repair_lemma
    only devoices the FINAL kept consonant, so the voiced consonant must be
    INCLUDED in the kept prefix for the repair to fire. Without the bump the
    model learns to cut *before* the alternating consonant (LCP), leaving it
    out of the prefix — and then no devoicing happens, producing басты
    instead of бастық.

    The bump is CONDITIONAL and conservative: it fires only on a true
    voicing/voiceless pair at the divergence point, never unconditionally.
    An unconditional +1 would break consonant-suffix words (адамның→адамн,
    қыздар→қызт). See tests/test_lemmatizer.py.
    """
    if not form:
        return 0
    # longest common prefix
    k = 0
    for a, b in zip(form, lemma):
        if a != b:
            break
        k += 1
    # Alternation bump: include the voiced consonant so devoicing can repair
    # it. Only when there IS a voiced consonant at form[k] AND lemma[k] is
    # exactly its voiceless pair (not just any divergence).
    if k < len(form) and k < len(lemma):
        if form[k] in _DEVOICE and _DEVOICE[form[k]] == lemma[k]:
            k += 1
    return min(k, MAX_PREFIX)


class LemmaDataset(Dataset):
    """Per-word prefix-length classification dataset."""

    def __init__(self, sentences: Sequence[Sentence], tokenizer, gold_lemmas,
                 max_len=MAX_LEN, n_gold: int | None = None):
        """
        Parameters
        ----------
        n_gold
            Number of leading entries in ``sentences`` that are GOLD; the
            remainder are silver sentences (silver lemma *dict* entries are
            already merged into ``gold_lemmas`` by the caller). Token positions
            inside gold sentences get ``silver_mask = 0.0``, inside silver
            sentences get ``silver_mask = 1.0``. ``None`` treats every sentence
            as gold (the legacy behaviour).
        """
        self.sentences = list(sentences)
        self.tokenizer = tokenizer
        self.gold = gold_lemmas
        self.max_len = max_len
        self.n_gold = n_gold if n_gold is not None else len(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = [t.form for t in self.sentences[idx].tokens]
        is_silver = 1.0 if idx >= self.n_gold else 0.0
        enc = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            # no padding — collator pads per batch
        )
        word_ids = enc.word_ids(batch_index=0)
        labels = []
        silver_mask = []
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
                silver_mask.append(0.0)
            else:
                form = tokens[wid].lower()
                lemma = self.gold.get(form, form)
                labels.append(prefix_label(form, lemma))
                silver_mask.append(is_silver)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "silver_mask": silver_mask,
        }


# ============================================================
#  Train + eval one fold
# ============================================================
class _LemmaSilverCollator:
    """Per-batch padding collator that also pads ``silver_mask``.

    Mirrors scripts/train_neural.py's ``_SilverAwareCollator``: wraps the HF
    DataCollatorForTokenClassification for input_ids/attention_mask/labels and
    adds the same dynamic padding to ``silver_mask`` (pad value 0.0).
    """

    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self._inner = DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=label_pad_token_id
        )

    def __call__(self, features: list[dict]) -> dict:
        silver_masks = [f.pop("silver_mask") for f in features]
        batch = self._inner(features)
        max_len = batch["labels"].shape[1]
        padded = torch.zeros(len(silver_masks), max_len, dtype=torch.float)
        for i, m in enumerate(silver_masks):
            padded[i, : len(m)] = torch.tensor(m, dtype=torch.float)
        batch["silver_mask"] = padded
        return batch


def _weighted_lemma_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    silver_mask: torch.Tensor,
    silver_weight: float,
) -> torch.Tensor:
    """Per-token prefix-length CE with per-token silver/gold weighting.

    At ``silver_weight=1.0`` this reduces exactly to the legacy
    ``out.loss = CrossEntropyLoss(ignore_index=-100).mean()`` — verified by
    test_silver_weight.test_silver_weight_1_equals_legacy_lemma_ce.
    """
    import torch.nn.functional as F
    flat_logits = logits.reshape(-1, NUM_LABELS)
    flat_labels = labels.reshape(-1)
    flat_silver = silver_mask.reshape(-1)
    valid = flat_labels != -100
    ce = F.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="none")
    per_token_w = (1.0 - flat_silver[valid]) * 1.0 + flat_silver[valid] * silver_weight
    return (ce * per_token_w).sum() / per_token_w.sum().clamp(min=1e-6)


def run_fold(
    base_model: str,
    train_sents: Sequence[Sentence],
    dev_sents: Sequence[Sentence],
    silver_lemmas: dict[str, str],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: str | None,
    fold_label: str,
    n_gold_train: int | None = None,
    silver_weight: float = 1.0,
) -> tuple[float, float]:
    """Return (best_exact_match_acc, best_prefix_acc).

    ``n_gold_train`` / ``silver_weight`` are the lemma-side analogues of the
    POS trainer's silver-weighting: silver sentences (the tail of
    ``train_sents`` past the first ``n_gold_train``) contribute their CE loss
    multiplied by ``silver_weight``; gold sentences contribute with weight 1.0.
    ``silver_weight=1.0`` (the default) reproduces the legacy behaviour
    bit-for-bit.
    """
    print(f"\n[{fold_label}] Loading {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=NUM_LABELS)
    model.to(device)

    # Build gold lemma maps. Training uses gold(train)+silver; eval uses gold(dev).
    train_gold = gold_lemma_pairs(train_sents)
    train_lemmas = {**silver_lemmas, **train_gold}  # gold wins on conflict
    dev_gold = gold_lemma_pairs(dev_sents)

    if n_gold_train is None:
        n_gold_train = len(train_sents)
    train_ds = LemmaDataset(train_sents, tokenizer, train_lemmas, n_gold=n_gold_train)
    dev_ds = LemmaDataset(dev_sents, tokenizer, dev_gold)
    collator = _LemmaSilverCollator(tokenizer=tokenizer, label_pad_token_id=-100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_exact = 0.0
    print(f"[{fold_label}] Train: {len(train_sents)} sents "
          f"(gold={n_gold_train}, silver={len(train_sents) - n_gold_train}, "
          f"silver_weight={silver_weight}) | Dev: {len(dev_sents)} sents")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            silver_mask = batch["silver_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = _weighted_lemma_loss(logits, labels, silver_mask, silver_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        # eval: exact-match lemma accuracy on dev (computed per-sentence so we
        # can decode the predicted lemma from each word's surface form)
        model.eval()
        exact_correct, prefix_correct, n = _eval_dev(model, tokenizer, dev_sents, dev_gold, device)

        exact_acc = exact_correct / n if n else 0.0
        prefix_acc = prefix_correct / n if n else 0.0
        marker = "  *best*" if exact_acc > best_exact else ""
        print(f"[{fold_label}] Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"lemma_acc={exact_acc:.4f}  prefix_acc={prefix_acc:.4f}{marker}")
        if exact_acc > best_exact:
            best_exact = exact_acc
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
    return best_exact, 0.0


def _eval_dev(model, tokenizer, dev_sents, dev_gold, device) -> tuple[int, int, int]:
    """Exact-match lemma accuracy + prefix-length accuracy on dev sentences.

    Uses the PRODUCTION decoder ``_repair_lemma`` (imported from
    kazmorph.neural_lemma) with the token's UPOS, so the exact-match accuracy
    here is computed exactly the way the pipeline decodes lemmas at inference.
    This is what makes the reported dev ``lemma_acc`` comparable to the held-out
    test numbers in reports/EVALUATION_REPORT.md.
    """
    exact_correct = 0
    prefix_correct = 0
    n = 0
    for s in dev_sents:
        tokens = s.tokens
        forms = [t.form for t in tokens]
        enc = tokenizer(list(forms), is_split_into_words=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        word_ids = enc.word_ids(batch_index=0)
        with torch.no_grad():
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(-1)[0].tolist()
        seen = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            form = forms[wid]
            upos = tokens[wid].upos
            gold_lemma = dev_gold.get(form.lower(), form.lower())
            pred_len = int(preds[idx])
            # Decode the predicted lemma exactly as the pipeline does: prefix +
            # POS-aware, lexicon-gated devoicing repair.
            pred_lemma = _repair_lemma(form.lower(), pred_len, pos=upos)
            # target prefix length for this (form, lemma) pair
            target_len = prefix_label(form.lower(), gold_lemma)
            n += 1
            if pred_lemma == gold_lemma:
                exact_correct += 1
            # prefix accuracy = did we predict the correct cut length?
            if pred_len == target_len:
                prefix_correct += 1
    return exact_correct, prefix_correct, n



# ============================================================
#  Main
# ============================================================
def train(args):
    from kazmorph.device import pick_device
    device = pick_device()
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        print(f"CPU intra-op threads: {torch.get_num_threads()}")
    print(f"Device: {device}  Base: {args.base_model}")

    pool = load_pool()
    print(f"Gold pool: {len(pool)} sentences")
    silver_lemmas = load_silver_lemmas(args.silver)
    print(f"Silver lemma dict: {len(silver_lemmas)} forms")

    # Silver SENTENCES (full CoNLL-U with lemmas) — contextual training data.
    # This roughly doubles the corpus and lets the model learn strip depths from
    # classical-literature wordforms not present in gold UD.
    silver_sents = []
    if args.silver_conllu:
        from kazmorph.corpus import parse_conllu as _parse
        for sf in args.silver_conllu:
            if os.path.exists(sf):
                silver_sents += _parse(sf)
        print(f"Silver sentences: {len(silver_sents)} from {len(args.silver_conllu)} file(s) "
              f"(silver_weight={args.silver_weight})")

    folds = stratified_kfold(pool, k=args.folds, seed=args.seed)
    accs: list[float] = []
    for i, (train_sents, dev_sents) in enumerate(folds):
        out_dir = args.model_dir if i == len(folds) - 1 else None
        # Silver sentences join the training fold; gold lemma dict still wins on
        # conflict for gold dev-eval consistency. n_gold_train tells the dataset
        # which sentences are gold so silver tokens can be loss-down-weighted.
        train_with_silver = list(train_sents) + silver_sents
        best, _ = run_fold(
            args.base_model, train_with_silver, dev_sents, silver_lemmas,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, out_dir=out_dir, fold_label=f"fold {i+1}/{len(folds)}",
            n_gold_train=len(train_sents), silver_weight=args.silver_weight,
        )
        accs.append(best)

    print("\n" + "=" * 60)
    print("  LEMMATIZER CROSS-VALIDATION SUMMARY (exact-match lemma acc)")
    print("=" * 60)
    for i, a in enumerate(accs):
        print(f"  Fold {i+1}: {a:.4f}")
    mean = sum(accs) / len(accs)
    if len(accs) > 1:
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"  Mean ± std: {mean:.4f} ± {std:.4f}")
    else:
        print(f"  Mean: {mean:.4f}")
    print("=" * 60)

    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "lemma_meta.json"), "w") as f:
        json.dump({"base_model": args.base_model, "folds": args.folds,
                   "fold_acc": accs, "mean_acc": mean,
                   "max_prefix": MAX_PREFIX}, f, indent=2)
    print(f"\nSaved to {args.model_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train neural Kazakh lemmatizer")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--silver", default=os.path.join(PROJECT_ROOT, "data", "silver_lemmas.tsv"),
                   help="Silver lemma TSV (form<TAB>lemma<TAB>upos)")
    p.add_argument("--silver-conllu", nargs="*", default=None,
                   help="Silver CoNLL-U sentence file(s) to mix into training "
                        "(e.g. data/silver_sentences.conllu)")
    p.add_argument("--silver-weight", type=float, default=0.5,
                   help="Per-token loss weight applied to silver sentences "
                        "(1.0 = legacy behaviour, <1.0 down-weights silver)")
    p.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "models", "neural_lemma"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    args = p.parse_args()
    train(args)
