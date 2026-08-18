#!/usr/bin/env python3
"""
Train neural POS tagger (Stage 3) for KazMorphCorpus-2026.

Fine-tunes a pretrained transformer for token classification (UPOS tagging)
on the UD Kazakh-KTB corpus.

Improvements over the original trainer:
  1. Honest data handling via kazmorph.corpus — the full UD Kazakh-KTB pool
     (1078 unique sentences) is deduplicated, and evaluation uses k-fold
     cross-validation so reported metrics are not inflated by train/dev/test
     overlap (test.conllu is a superset of train_split.conllu).
  2. Pluggable base model — default is the monolingual KazBERT, which the
     literature reports at 90–92% Kazakh morphology accuracy vs ~84% for the
     multilingual XLM-RoBERTa-base used previously. XLM-R and Kaz-RoBERTa are
     selectable via --base-model.
  3. Class weighting to counter the severe tag imbalance (SCONJ/INTJ/SYM have
     3–23 tokens vs thousands for NOUN/VERB) that drags Macro-F1 down.
  4. Optional silver-data files (CoNLL-U produced by the Apertium tagger)
     can be mixed in with a down-weighted loss. The weight is controlled by
     --silver-weight (default 0.3): silver tokens contribute their per-token
     CE multiplied by this weight, gold tokens by 1.0. This prevents the
     residual ~1.9% Apertium-vs-UD tag disagreements (concentrated on the
     VERB↔AUX distinction) from being learned as if they were gold. A weight
     of 1.0 reproduces the legacy "list(train_gold) + silver_sents" behaviour
     bit-for-bit.

The final fold's best model is saved to --model-dir (default models/neural_pos)
so the pipeline's NeuralTagger loads it unchanged.

Usage:
    # KazBERT, 5-fold CV (recommended)
    python scripts/train_neural.py --base-model Eraly-ml/KazBERT --folds 5

    # Quick single-split smoke test
    python scripts/train_neural.py --folds 1 --epochs 3

    # Add silver CoNLL-U from the Apertium tagger, weighted 0.3
    python scripts/train_neural.py --silver data/silver_apertium.conllu --silver-weight 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
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
from seqeval.metrics import classification_report, f1_score

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import POS_TAGS, Sentence, load_pool, stratified_kfold, tag_counts
from kazmorph.augment import AugmentConfig, augment_sentence


# ============================================================
#  Defaults
# ============================================================
BASE_MODEL = "Eraly-ml/KazBERT"   # monolingual Kazakh BERT (was: xlm-roberta-base)
MAX_LEN = 128

# BIO tagging for seqeval: B-TAG per POS, plus O for non-UD tokens.
TAG2ID = {"O": 0}
for _i, _t in enumerate(POS_TAGS):
    TAG2ID[f"B-{_t}"] = _i + 1
ID2TAG = {v: k for k, v in TAG2ID.items()}
NUM_LABELS = len(TAG2ID)


# ============================================================
#  Dataset
# ============================================================
class POSTokenDataset(Dataset):
    """
    Dataset for POS tagging with subword alignment.

    Returns UNPADDED per-sentence examples (lists, not padded tensors); dynamic
    padding to the longest sentence *in each batch* is handled by
    DataCollatorForTokenClassification. This matters a lot on CPU: the corpus
    median sentence is 8 tokens, so padding every sentence to 128 wasted ~15×
    compute. Dynamic padding cuts epoch time by roughly an order of magnitude.
    """

    def __init__(self, sentences: Sequence[Sentence], tokenizer, tag2id, max_len=MAX_LEN,
                 n_gold: int | None = None, augment: AugmentConfig | None = None,
                 augment_seed: int = 0):
        """
        Parameters
        ----------
        n_gold
            Number of leading entries in ``sentences`` that are GOLD; the
            remainder are silver. Token positions inside gold sentences get
            ``silver_mask = 0.0`` (loss weight 1.0 at silver_weight=1); tokens
            inside silver sentences get ``silver_mask = 1.0``. ``None`` treats
            every sentence as gold (the legacy behaviour), which keeps the
            dataset backward-compatible with callers that never pass silver
            data.
        augment
            Optional :class:`~kazmorph.augment.AugmentConfig`. When set and
            active, GOLD sentences are augmented on the fly in ``__getitem__``
            (silver sentences are never augmented — they already carry label
            noise). Each epoch therefore sees fresh variations. ``None``
            (default) disables augmentation and reproduces legacy behaviour.
        augment_seed
            Base seed for the per-item RNG. Combined with the item index so
            different items get independent streams and results are
            reproducible.
        """
        self.sentences = list(sentences)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        self.n_gold = n_gold if n_gold is not None else len(self.sentences)
        self.augment = augment
        # One RNG per dataset, seeded deterministically; __getitem__ advances it.
        # We do NOT seed per-item with the index because augmentation is meant
        # to vary across epochs (DataLoader reshuffles), and a stable base seed
        # keeps a full run reproducible while still differing epoch-to-epoch.
        self._aug_rng = random.Random(augment_seed)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        is_silver = 1.0 if idx >= self.n_gold else 0.0
        # Augment gold sentences only (silver already carries label noise).
        if self.augment is not None and self.augment.active and not is_silver:
            sentence = augment_sentence(sentence, self._aug_rng, self.augment)

        tokens, tags = sentence.forms, sentence.upos_tags

        encoding = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            # NOTE: no padding here — the data collator pads per batch.
        )

        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        silver_mask = []
        for wid in word_ids:
            if wid is None:
                labels.append(-100)  # ignored in loss
                silver_mask.append(0.0)
            else:
                tag = tags[wid]
                bio_tag = f"B-{tag}" if tag in POS_TAGS else "O"
                labels.append(self.tag2id.get(bio_tag, 0))
                silver_mask.append(is_silver)

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "silver_mask": silver_mask,
        }


# ============================================================
#  Class-weight computation
# ============================================================
def compute_class_weights(sentences: Sequence[Sentence], smooth: float = 1.0) -> torch.Tensor:
    """
    Inverse-frequency class weights for the BIO label set.

    Weight = total / (num_classes * (count + smooth)). The +smooth avoids
    division by zero for near-absent classes and dampens the blow-up for
    extremely rare tags.

    Labels that never occur in ``sentences`` (count == 0) get a NEUTRAL weight
    of 1.0. This matters because the tag inventory is fixed (NUM_LABELS, built
    from the full POS_TAGS set) but any given training pool only uses a subset:
    e.g. the ``O`` label (non-UD tokens) and some rare tags (SYM, PART) are
    often entirely absent. Without this guard, an absent label received weight
    ``total / (num_classes * smooth)`` — for the ~10k-token pool that is ~410x,
    which then dominated the loss and collapsed training (loss blew up to ~8,
    dev_f1 -> 0). Neutral 1.0 is correct: a class the model never sees should
    not be up-weighted.
    """
    counts = Counter()
    for s in sentences:
        for tag in s.upos_tags:
            bio = f"B-{tag}" if tag in POS_TAGS else "O"
            counts[bio] += 1

    total = sum(counts.values())
    weights = torch.zeros(NUM_LABELS, dtype=torch.float)
    for i in range(NUM_LABELS):
        label = ID2TAG[i]
        c = counts.get(label, 0)
        if c == 0:
            # Absent label: neutral weight, not inverse-frequency-of-zero
            # (which would be a huge number and destabilise the loss).
            weights[i] = 1.0
        else:
            weights[i] = total / (NUM_LABELS * (c + smooth))
    return weights


# ============================================================
#  Silver-aware data plumbing
# ============================================================
class _SilverAwareCollator:
    """Per-batch padding collator that also pads ``silver_mask``.

    Wraps ``DataCollatorForTokenClassification`` (which already handles
    input_ids, attention_mask, and labels with the right pad token) and adds
    the same per-batch dynamic padding to the ``silver_mask`` field produced
    by ``POSTokenDataset``. We can't use the HF collator directly because it
    only knows about the three canonical token-classification fields; any
    extra key in the batch dict is left un-padded and breaks the DataLoader.

    The mask is padded with 0.0 — silver loss weight contributes nothing on
    pad positions, matching how labels are padded with -100 (ignored in loss).
    """

    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self._inner = DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=label_pad_token_id
        )

    def __call__(self, features: list[dict]) -> dict:
        # Pop silver_mask before handing the rest to the HF collator, which
        # would otherwise try to pad it as if it were labels.
        silver_masks = [f.pop("silver_mask") for f in features]
        batch = self._inner(features)
        max_len = batch["labels"].shape[1]
        padded = torch.zeros(len(silver_masks), max_len, dtype=torch.float)
        for i, m in enumerate(silver_masks):
            padded[i, : len(m)] = torch.tensor(m, dtype=torch.float)
        batch["silver_mask"] = padded
        return batch


def _weighted_pos_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    silver_mask: torch.Tensor,
    silver_weight: float,
    class_weights: torch.Tensor | None,
    device: torch.device,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Per-token CE with class weights AND a per-token silver/gold weight.

    Per-token loss weight = ``silver_weight`` where silver_mask=1 (silver
    token) and ``1.0`` where silver_mask=0 (gold token). Both class weights
    (rare-tag up-weighting) and the silver/gold weight are applied
    multiplicatively on each token's CE contribution, then normalised by the
    total weight so the loss stays on the same scale regardless of
    silver_weight.

    Two optional regularizers (default off → legacy behaviour preserved):

      * ``label_smoothing`` ∈ [0, 1) — passed through to
        ``F.cross_entropy(label_smoothing=...)``. Mixes a uniform prior into
        the target distribution, which reduces overconfidence and is
        particularly helpful for rare classes on small datasets.
      * ``focal_gamma`` > 0 — multiplies each token's (already
        class-weighted, optionally smoothed) CE by a focusing factor
        ``(1 - p_t) ** gamma``, where ``p_t`` is the model's predicted
        probability of the true class (Lin et al., 2017). gamma=2 is the
        standard value; it down-weights easy, confident-correct tokens so the
        optimizer spends more capacity on hard/rare cases.

    The legacy ``silver_weight=1.0`` with ``class_weights=None`` and both
    regularizers at 0 reduces exactly to
    ``CrossEntropyLoss(ignore_index=-100).mean()`` — verified by
    test_silver_weight.test_silver_weight_1_equals_legacy_ce.
    """
    import torch.nn.functional as F

    flat_logits = logits.view(-1, NUM_LABELS)        # (B*T, C)
    flat_labels = labels.view(-1)                    # (B*T,)
    flat_silver = silver_mask.reshape(-1)            # (B*T,)
    valid = flat_labels != -100

    v_logits = flat_logits[valid]
    v_labels = flat_labels[valid]

    # Per-token unweighted CE; ignore_index handling is manual since we need
    # the per-token values for the weighted reduction. label_smoothing is
    # passed straight through (0 = hard target, legacy behaviour).
    ce = F.cross_entropy(
        v_logits,
        v_labels,
        weight=class_weights.to(device) if class_weights is not None else None,
        reduction="none",
        label_smoothing=label_smoothing,
    )                                                # (n_valid,)

    # Focal modulation: scale each token's loss by (1 - p_true)^gamma so that
    # tokens the model already gets right (high p_true) contribute less.
    if focal_gamma > 0.0:
        with torch.no_grad():
            p_true = torch.softmax(v_logits, dim=-1).gather(1, v_labels.unsqueeze(1)).squeeze(1)
            focal = (1.0 - p_true).clamp(min=1e-6) ** focal_gamma
        ce = ce * focal

    per_token_w = (1.0 - flat_silver[valid]) * 1.0 + flat_silver[valid] * silver_weight
    return (ce * per_token_w).sum() / per_token_w.sum().clamp(min=1e-6)


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
    class_weights: torch.Tensor | None,
    device: torch.device,
    out_dir: str | None = None,
    fold_label: str = "",
    n_gold_train: int | None = None,
    silver_weight: float = 1.0,
    patience: int | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    augment: AugmentConfig | None = None,
    warmup_frac: float = 0.1,
) -> tuple[float, dict]:
    """Train one fold; return (best_dev_weighted_f1, last_epoch_details).

    Parameters
    ----------
    n_gold_train
        Number of leading entries in ``train_sents`` that are GOLD. The
        remainder are silver (see ``POSTokenDataset.__init__``). If ``None``
        or equal to ``len(train_sents)`` there is no silver data and the loss
        is identical to the legacy behaviour.
    silver_weight
        Per-token loss weight applied to silver tokens. ``1.0`` (the default)
        is exactly the legacy ``list(train_gold) + silver_sents`` behaviour,
        so existing runs are reproducible bit-for-bit by leaving this at 1.0.
        Values < 1.0 down-weight silver so its residual errors (the ~1.9% of
        Apertium tags that disagree with gold, concentrated on VERB↔AUX) hurt
        the gold-trained model less.
    patience
        Early stopping: if the dev weighted-F1 does not improve for this many
        consecutive epochs, training stops early and the best checkpoint is
        kept. ``None`` disables early stopping (run all ``epochs``).
    """
    print(f"\n[{fold_label}] Loading {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=NUM_LABELS)
    model.to(device)

    if not train_sents:
        raise ValueError(f"[{fold_label}] empty train set — check --folds/--data")
    if not dev_sents:
        raise ValueError(f"[{fold_label}] empty dev set — check --folds/--data")

    # If the caller didn't tell us how many gold sentences there are, every
    # sentence is gold and silver-weighting is a no-op (legacy behaviour).
    if n_gold_train is None:
        n_gold_train = len(train_sents)
    train_ds = POSTokenDataset(
        train_sents, tokenizer, TAG2ID, n_gold=n_gold_train,
        augment=augment,
    )
    dev_ds = POSTokenDataset(dev_sents, tokenizer, TAG2ID)
    # DataCollatorForTokenClassification pads input_ids, attention_mask and
    # labels automatically; our extra silver_mask needs the same per-batch
    # padding (with 0.0 — silver loss weight contributes nothing on pads).
    collator = _SilverAwareCollator(tokenizer=tokenizer, label_pad_token_id=-100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_frac * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    history = []
    print(f"[{fold_label}] Train: {len(train_sents)} sents "
          f"(gold={n_gold_train}, silver={len(train_sents) - n_gold_train}, "
          f"silver_weight={silver_weight}) | "
          f"Dev: {len(dev_sents)} sents | epochs={epochs}")

    # Early-stopping state: number of epochs since dev-F1 last improved.
    epochs_since_best = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            silver_mask = batch["silver_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, C)
            loss = _weighted_pos_loss(
                logits, labels, silver_mask, silver_weight, class_weights, device,
                label_smoothing=label_smoothing, focal_gamma=focal_gamma,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        # --- eval ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                for i in range(labels.shape[0]):
                    pred_seq, label_seq = [], []
                    for j in range(labels.shape[1]):
                        if labels[i, j].item() != -100:
                            pred_seq.append(ID2TAG[preds[i, j].item()])
                            label_seq.append(ID2TAG[labels[i, j].item()])
                    all_preds.append(pred_seq)
                    all_labels.append(label_seq)

        dev_f1 = f1_score(all_labels, all_preds, average="weighted")
        marker = "  *best*" if dev_f1 > best_f1 else ""
        print(f"[{fold_label}] Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"dev_f1={dev_f1:.4f}{marker}")
        history.append({"epoch": epoch + 1, "loss": avg_loss, "dev_f1": dev_f1})

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            epochs_since_best = 0
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                print(f"[{fold_label}] Early stopping at epoch {epoch+1} "
                      f"(no dev_f1 improvement for {patience} epochs).")
                break

    return best_f1, {"history": history, "last_report": (all_labels, all_preds)}


# ============================================================
#  Main
# ============================================================

def _write_cv_results(model_dir: str, base_model: str, folds: int,
                      epochs: int, fold_f1s: list[float], *,
                      partial: bool) -> None:
    """Write cv_results.json (and a partial variant) so progress survives
    interruptions. ``partial=True`` also writes cv_results_partial.json so a
    reader can tell the CV run is not yet complete."""
    os.makedirs(model_dir, exist_ok=True)
    mean_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0
    std = (sum((f - mean_f1) ** 2 for f in fold_f1s) / len(fold_f1s)) ** 0.5 \
        if len(fold_f1s) > 1 else 0.0
    payload = {
        "base_model": base_model, "folds": folds, "epochs": epochs,
        "folds_completed": len(fold_f1s),
        "fold_f1": fold_f1s, "mean_f1": mean_f1, "std_f1": std,
        "partial": partial,
    }
    out = os.path.join(model_dir, "cv_results_partial.json" if partial
                       else "cv_results.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    if not partial and os.path.exists(os.path.join(model_dir, "cv_results_partial.json")):
        os.remove(os.path.join(model_dir, "cv_results_partial.json"))


def _promote_fold(model_dir: str, fold_dir: str) -> None:
    """Copy a completed fold's checkpoint into the top-level model dir so the
    inference Pipeline always has a loadable model after every fold. Uses
    shutil.copytree with dirs_exist_ok to overwrite in place."""
    import shutil
    os.makedirs(model_dir, exist_ok=True)
    for name in os.listdir(fold_dir):
        src = os.path.join(fold_dir, name)
        dst = os.path.join(model_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def train(args):
    from kazmorph.device import pick_device
    device = pick_device()
    # On CPU, cap intra-op parallelism at a moderate thread count. Setting it
    # too high relative to physical cores causes thread oversubscription and
    # catastrophic slowdown (observed: 79 threads, ~0 epochs in 8 min). We do
    # NOT raise interop threads — BERT forward is intra-op bound, and extra
    # interop threads only add contention. (MPS/CUDA ignore this knob.)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        print(f"CPU intra-op threads: {torch.get_num_threads()}")
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")

    # --- load gold pool ---
    pool = load_pool(args.data if args.data else None, dedup=True)
    print(f"Gold pool (deduplicated): {len(pool)} sentences")

    silver_sents: list[Sentence] = []
    if args.silver:
        from kazmorph.corpus import parse_conllu
        for sf in args.silver:
            silver_sents += parse_conllu(sf)
        print(f"Silver data: {len(silver_sents)} sentences "
              f"(silver_weight={args.silver_weight}) "
              f"from {len(args.silver)} file(s)")

    # tag distribution on gold
    tc = tag_counts(pool)
    print(f"Gold tag distribution ({sum(tc.values())} tokens): "
          f"{', '.join(f'{t}={c}' for t, c in sorted(tc.items(), key=lambda kv: -kv[1])[:6])} ...")

    if args.dev_file:
        dev_pool = load_pool([args.dev_file], dedup=True)
        folds = [(pool, dev_pool)]
        print(f"External dev file: {args.dev_file} ({len(dev_pool)} sentences); "
              f"training uses the FULL gold pool ({len(pool)} sentences), "
              f"k-fold splitting bypassed")
    else:
        folds = stratified_kfold(pool, k=args.folds, seed=args.seed)
        print(f"Cross-validation: {len(folds)} folds")

    fold_f1s: list[float] = []
    last_fold_report = None

    for i, (train_gold, dev) in enumerate(folds):
        fold_label = f"fold {i+1}/{len(folds)}"
        fold_dir = os.path.join(args.model_dir, f"fold_{i+1}")
        fold_metric_path = os.path.join(fold_dir, "fold_metric.json")

        # ── Resume support ──────────────────────────────────────
        # A fold counts as DONE when its checkpoint dir holds a fold_metric.json.
        # On --resume we read back the stored F1 and skip retraining that fold,
        # so an interrupted 5-fold run only redoes the folds it never finished.
        if args.resume and os.path.exists(fold_metric_path):
            with open(fold_metric_path) as fm:
                best_f1 = json.load(fm)["best_f1"]
            print(f"\n[{fold_label}] RESUMED — skipping (cached best_f1={best_f1:.4f})")
            fold_f1s.append(best_f1)
            continue

        # Combine gold train + (optionally down-weighted) silver train. The
        # gold count is passed through so the dataset knows which tokens to
        # mark as silver; silver_weight=1.0 reproduces the legacy behaviour.
        train_sents = list(train_gold) + silver_sents
        n_gold_train = len(train_gold)
        # Class weights are computed over the FULL training pool (gold + silver),
        # not gold alone. Computing on gold-only made rare-class weights (PART,
        # INTJ, SYM) explode to 17-100x; when silver/synth then introduces many
        # such tokens at silver_weight, the weighted loss blew up to ~8 and the
        # model collapsed (dev_f1 ≈ 0). Using the actual training distribution
        # keeps the weights moderate and the loss stable.
        cw = compute_class_weights(train_sents, smooth=args.class_weight_smooth) \
            if args.class_weights else None
        # Build the augmentation config. --no-augment disables it entirely.
        aug_cfg = None if args.no_augment else AugmentConfig(
            casing_dropout=args.augment_casing,
            token_dropout=args.augment_token,
            punct_jitter=args.augment_punct,
        )

        # Each fold persists to its own fold_K/ checkpoint so the run is
        # resumable. After the loop the last fold is promoted to the top-level
        # model dir (the path the inference Pipeline loads from).
        best_f1, details = run_fold(
            args.base_model,
            train_sents,
            dev,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            class_weights=cw,
            device=device,
            out_dir=fold_dir,
            fold_label=fold_label,
            n_gold_train=n_gold_train,
            silver_weight=args.silver_weight,
            patience=args.patience or None,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            augment=aug_cfg,
            warmup_frac=args.warmup_frac,
        )

        # Persist this fold's metric + tag_map so --resume can detect it and
        # the inference code can load the tokenizer from the fold dir.
        os.makedirs(fold_dir, exist_ok=True)
        with open(fold_metric_path, "w") as fm:
            json.dump({"best_f1": best_f1, "fold": i + 1,
                       "epochs_run": len(details.get("history", []))}, fm, indent=2)
        with open(os.path.join(fold_dir, "tag_map.json"), "w") as f:
            json.dump({"tag2id": TAG2ID,
                       "id2tag": {str(k): v for k, v in ID2TAG.items()},
                       "pos_tags": POS_TAGS, "base_model": args.base_model}, f, indent=2)

        fold_f1s.append(best_f1)
        if i == len(folds) - 1:
            last_fold_report = details["last_report"]

        # Incremental cv_results so partial progress is never lost.
        _write_cv_results(args.model_dir, args.base_model, args.folds,
                          args.epochs, fold_f1s, partial=(i < len(folds) - 1))

        # Promote the just-finished fold's checkpoint to the top-level model
        # dir immediately, so the runnable inference model is always the most
        # recent completed fold (and is overwritten by each subsequent fold).
        _promote_fold(args.model_dir, fold_dir)

    # --- summary across folds ---
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, f1 in enumerate(fold_f1s):
        print(f"  Fold {i+1}: best dev weighted-F1 = {f1:.4f}")
    mean_f1 = sum(fold_f1s) / len(fold_f1s)
    if len(fold_f1s) > 1:
        var = sum((f - mean_f1) ** 2 for f in fold_f1s) / len(fold_f1s)
        std = var ** 0.5
        print(f"  Mean ± std: {mean_f1:.4f} ± {std:.4f}")
    else:
        print(f"  Mean: {mean_f1:.4f}")
    print("=" * 60)

    # --- persist final artifacts for the promoted (last fold) checkpoint ---
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "tag_map.json"), "w") as f:
        json.dump({"tag2id": TAG2ID, "id2tag": {str(k): v for k, v in ID2TAG.items()},
                   "pos_tags": POS_TAGS, "base_model": args.base_model}, f, indent=2)
    _write_cv_results(args.model_dir, args.base_model, args.folds,
                      args.epochs, fold_f1s, partial=False)

    print(f"\nFinal checkpoint + tag_map.json saved to: {args.model_dir}/")

    if last_fold_report is not None:
        labels, preds = last_fold_report
        print("\nClassification report (last fold, dev):")
        try:
            print(classification_report(labels, preds, digits=4))
        except Exception as e:  # seqeval can be finicky on tiny folds
            print(f"  (classification_report unavailable: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train neural POS tagger for Kazakh")
    p.add_argument("--base-model", default=BASE_MODEL,
                   help="HuggingFace model id (default: Eraly-ml/KazBERT)")
    p.add_argument("--folds", type=int, default=5,
                   help="Number of CV folds (default: 5; use 1 for single split)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping: stop after this many epochs without "
                        "dev-F1 improvement (default: 3; 0 disables).")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--class-weights", action="store_true", default=True,
                   help="Use inverse-frequency class weights (default: on)")
    p.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    p.add_argument("--class-weight-smooth", type=float, default=1.0,
                   help="Smoothing for inverse-frequency class weights (default 1.0). "
                        "Lower → more aggressive up-weighting of rare tags (X, INTJ).")
    # ── Loss regularizers ────────────────────────────────────────────────────
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing for the POS loss (default 0.0 = off). "
                        "0.1 is a typical value; reduces overconfidence on rare classes.")
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="Focal-loss gamma (default 0.0 = off). 2.0 focuses the loss "
                        "on hard/rare examples (Lin et al. 2017).")
    # ── Augmentation (gold-only; see kazmorph/augment.py) ────────────────────
    p.add_argument("--no-augment", action="store_true",
                   help="Disable on-the-fly data augmentation entirely.")
    p.add_argument("--augment-casing", type=float, default=0.10,
                   help="Casing-dropout probability per gold token (default 0.10).")
    p.add_argument("--augment-token", type=float, default=0.05,
                   help="Token→[MASK] dropout probability per gold token (default 0.05).")
    p.add_argument("--augment-punct", type=float, default=0.10,
                   help="Punctuation-duplication probability per gold sentence (default 0.10).")
    # ── LR schedule ──────────────────────────────────────────────────────────
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of total optimizer steps used for linear warmup (default 0.1).")
    p.add_argument("--silver", nargs="*", default=None,
                   help="Path(s) to silver CoNLL-U files to mix into training")
    p.add_argument("--silver-weight", type=float, default=0.3,
                   help="Per-token loss weight applied to silver data (1.0 = "
                        "legacy behaviour, <1.0 down-weights silver)")
    p.add_argument("--data", nargs="*", default=None,
                   help="Override gold CoNLL-U files (default: all data/*.conllu)")
    p.add_argument("--dev-file", default=None,
                   help="External dev CoNLL-U (e.g. data/dev.conllu). When set, ALL "
                        "sentences from --data are used for training and this file "
                        "becomes the dev set (early stopping + model selection); "
                        "stratified k-fold splitting is bypassed. This makes the "
                        "training data identical to what train_crf.py consumes.")
    p.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "models", "neural_pos"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4),
                   help="torch intra-op threads on CPU (default: min(8, cores))")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted CV run: skip folds whose checkpoint "
                        "already exists under <model-dir>/fold_K/. On completion, "
                        "the last fold's checkpoint is promoted to <model-dir>/.")
    args = p.parse_args()
    train(args)
