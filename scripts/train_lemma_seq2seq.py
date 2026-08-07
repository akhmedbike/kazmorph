#!/usr/bin/env python3
"""
Train a seq2seq neural lemmatizer for Kazakh (Stage 3b, alternative to the
prefix-length classifier in scripts/train_lemma.py).

Task formulation
----------------
Lemmatization is cast as character-level sequence-to-sequence translation:
  source: "<UPOS> wordform"
  target: "lemma"

The POS tag is prepended as a soft hint about paradigm (жүр as VERB = "run",
жүр as AUX = "is going"). A seq2seq model can learn stem alternations (қ↔ғ,
к↔г) directly that the length-only classifier cannot encode and had to patch
with a hand-written devoice rule.

Why seq2seq here, given we just spent effort making the pipeline fast:
the classifier caps at ~87% lemma accuracy on the held-out split (61.5% of
its errors are systematic off-by-one over-truncation that the length label
cannot express). The seq2seq head removes that ceiling. Speed is gated by
an A/B test (scripts/ab_lemma.py) — seq2seq only replaces the classifier
if it wins on accuracy AND stays within 3x latency.

Backbone: google/mt5-small (300M params, multilingual SentencePiece vocab of
250k covering Cyrillic well).

Tokenizer note
--------------
transformers 5.10.2 has a regression: AutoTokenizer mis-detects mt5's
spiece.model as a tiktoken file and crashes. We bypass that by wrapping
sentencepiece directly in MTSPTokenizer below, which exposes the minimal
encode/decode interface that MT5ForConditionalGeneration needs (it only
consumes input_ids / attention_mask / labels tensors, never tokenizer
methods at train time). The tokenizer object itself is saved alongside
the model and reloaded the same way at inference.

Data
----
- Gold: UD Kazakh-KTB (deduplicated pool, 1078 sentences) via kazmorph.corpus.
- Silver: data/silver_lemmas.tsv (form→lemma pairs) and optionally
  data/silver_sentences.conllu (contextual sentences).

Output: models/neural_lemma_seq2seq/  (config + safetensors + spiece.model
+ tokenizer_meta.json + lemma_meta.json).

Usage
-----
    # smoke test (a few minutes)
    python scripts/train_lemma_seq2seq.py --folds 1 --epochs 3

    # full training (overnight, ~6-9h on 16-core CPU)
    systemd-inhibit --what=sleep --who="kazmorph" --why="seq2seq lemma" \\
        .venv/bin/python -u scripts/train_lemma_seq2seq.py \\
        --folds 3 --epochs 12 --threads 8 \\
        --silver data/silver_lemmas.tsv \\
        --silver-conllu data/silver_sentences.conllu \\
        --model-dir models/neural_lemma_seq2seq
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Sequence

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    MT5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.corpus import Sentence, load_pool, stratified_kfold


BASE_MODEL = "google/mt5-small"
MAX_SOURCE_LEN = 64   # "<UPOS> wordform" — Kazakh words rarely exceed ~20 chars
MAX_TARGET_LEN = 32   # lemmas are ≤ form length, cap at 32 for headroom
SP_EPSILON = 0.1      # label_smoothing for seq2seq LM loss (helps small data)


# ============================================================
#  SentencePiece tokenizer wrapper
# ============================================================
class MTSPTokenizer:
    """
    Thin wrapper over sentencepiece exposing the interface MT5 training/
    inference needs. transformers 5.10.2's AutoTokenizer has a regression that
    mis-detects mt5's spiece.model as tiktoken and crashes; sentencepiece
    itself works perfectly (250100-piece vocab, correct Cyrillic segmentation).

    Saved alongside the model so inference reloads the exact same vocab.
    """

    def __init__(self, spiece_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spiece_path)
        self.spiece_path = spiece_path
        self.vocab_size = self.sp.GetPieceSize()
        self.pad_token_id = self.sp.PieceToId("<pad>")      # 0
        self.eos_token_id = self.sp.PieceToId("</s>")       # 1
        self.bos_token_id = None                            # mt5 has no bos
        self.unk_token_id = self.sp.PieceToId("<unk>")      # 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def encode(self, text: str, *, max_length: int | None = None,
               add_eos: bool = True) -> list[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids
                   if i not in (self.pad_token_id, self.eos_token_id, self.unk_token_id)
                   and i >= 0]
        return self.sp.DecodeIds(ids)

    def save(self, dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "spiece.model")
        shutil.copyfile(self.spiece_path, dest)
        return dest


def _resolve_spiece(model_id: str) -> str:
    """Locate spiece.model for an mt5 checkpoint, downloading if needed."""
    # When loading from a local fine-tuned dir, spiece.model sits next to config.
    local = os.path.join(model_id, "spiece.model")
    if os.path.exists(local):
        return local
    # Otherwise resolve from HF cache via snapshot_download (no tokenizer load).
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=model_id, allow_patterns=["spiece.model"])
    return os.path.join(path, "spiece.model")


# ============================================================
#  Data
# ============================================================
def gold_lemma_pairs(sentences: Sequence[Sentence]) -> dict[str, tuple[str, str]]:
    """{form_lower: (lemma_lower, upos)} from gold UD sentences.

    upos is included so the seq2seq source "<UPOS> word" can be built during
    training. If a form appears with multiple POS tags, the first wins (rare).
    """
    out: dict[str, tuple[str, str]] = {}
    for s in sentences:
        for tok in s.tokens:
            key = tok.form.lower()
            if key not in out:
                out[key] = (tok.lemma.lower(), tok.upos)
    return out


def load_silver_lemmas(path: str | None) -> dict[str, tuple[str, str]]:
    """{form_lower: (lemma_lower, upos)} from silver TSV (form<TAB>lemma<TAB>upos).

    The upos column lets us build the "<UPOS> word" source for silver pairs.
    """
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
                out.setdefault(parts[0], (parts[1], parts[2]))
            elif len(parts) >= 2 and parts[0] and parts[1]:
                # Fall back to NOUN tag if upos column missing (shouldn't happen
                # for silver_lemmas.tsv, but be defensive).
                out.setdefault(parts[0], (parts[1], "NOUN"))
    return out


class LemmaSeq2SeqDataset(Dataset):
    """
    Per-word seq2seq examples: source "<UPOS> form" → target "lemma".

    Word-level (not sentence-level) batching is the deliberate choice: Kazakh
    lemmatization is nearly context-free (suffixes attach based on form, not
    neighbors), so a per-word model matches a per-sentence one on accuracy
    while generating much shorter decoder sequences — critical for keeping
    CPU inference within the 3x latency budget set by the A/B gate.

    Each sentence contributes one example per non-PUNCT token. The form's POS
    comes from the gold (train) or silver annotation, prepended to the source
    string.
    """

    def __init__(
        self,
        sentences: Sequence[Sentence],
        tokenizer: MTSPTokenizer,
        lemma_lookup: dict[str, tuple[str, str]],
        *,
        max_source_len: int = MAX_SOURCE_LEN,
        max_target_len: int = MAX_TARGET_LEN,
    ):
        self.examples: list[tuple[str, str]] = []
        # De-dup on (form, upos): a repeated wordform in the same role adds no
        # signal and inflates epoch size.
        seen: set[tuple[str, str]] = set()
        for s in sentences:
            for tok in s.tokens:
                if tok.upos == "PUNCT":
                    continue
                form = tok.form.lower()
                # Prefer annotated lemma+upos from lookup (gold/silver merge);
                # fall back to the sentence's own annotation.
                if form in lemma_lookup:
                    lemma, upos = lemma_lookup[form]
                else:
                    lemma, upos = tok.lemma.lower(), tok.upos
                key = (form, upos)
                if key in seen:
                    continue
                seen.add(key)
                self.examples.append((f"{upos} {form}", lemma))
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        source, target = self.examples[idx]
        return {
            "input_ids": self.tokenizer.encode(source, max_length=self.max_source_len),
            "labels": self.tokenizer.encode(target, max_length=self.max_target_len),
        }


def collate_fn_factory(tokenizer: MTSPTokenizer):
    """Dynamic padding collator. Pads source with pad_id, target labels with -100
    so PyTorch CE ignores them. Also builds attention_mask from source length."""
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        src_max = max(len(b["input_ids"]) for b in batch)
        tgt_max = max(len(b["labels"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            src = b["input_ids"]
            tgt = b["labels"]
            input_ids.append(src + [pad_id] * (src_max - len(src)))
            attn.append([1] * len(src) + [0] * (src_max - len(src)))
            # labels padded with -100 (ignored in loss); decoder_input_ids is
            # built internally by MT5 by shifting labels right.
            labels.append(tgt + [-100] * (tgt_max - len(tgt)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


# ============================================================
#  Train + eval one fold
# ============================================================
def run_fold(
    base_model: str,
    train_sents: Sequence[Sentence],
    dev_sents: Sequence[Sentence],
    silver_lemmas: dict[str, tuple[str, str]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: str | None,
    fold_label: str,
    spiece_path: str,
) -> float:
    """Return best exact-match lemma accuracy on dev."""
    print(f"\n[{fold_label}] Loading {base_model} ...")
    tokenizer = MTSPTokenizer(spiece_path)
    model = MT5ForConditionalGeneration.from_pretrained(base_model)
    model.to(device)

    train_gold = gold_lemma_pairs(train_sents)
    train_lookup = {**silver_lemmas, **train_gold}  # gold wins on conflict
    dev_gold = gold_lemma_pairs(dev_sents)

    train_ds = LemmaSeq2SeqDataset(train_sents, tokenizer, train_lookup)
    dev_lookup_for_eval = dev_gold  # only forms present in dev get scored
    collate = collate_fn_factory(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate)

    # AdamW with weight decay. mt5 tolerates higher LR than BERT-classification
    # heads; 1e-4 is a common starting point in the mt5 fine-tuning literature.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # label_smoothing is critical for mt5 convergence on small data: the
            # pretrained span-corruption head is overconfident, and without
            # smoothing it takes many epochs before the model starts emitting
            # the target distribution. 0.1 is the value from the original T5
            # paper and a common default for low-resource fine-tuning.
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, label_smoothing=0.1)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        # Eval: exact-match lemma accuracy on dev, decoded with greedy generate.
        model.eval()
        acc, n = _eval_dev(model, tokenizer, dev_sents, dev_lookup_for_eval, device)
        marker = "  *best*" if acc > best_acc else ""
        print(f"[{fold_label}] Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"lemma_acc={acc:.4f}  ({n} tokens){marker}")
        if acc > best_acc:
            best_acc = acc
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                model.save_pretrained(out_dir, safe_serialization=True)
                tokenizer.save(out_dir)
    return best_acc


def _eval_dev(model, tokenizer: MTSPTokenizer, dev_sents, dev_gold, device) -> tuple[float, int]:
    """Greedy-decode lemmas for each dev word and compute exact-match accuracy.

    Batched generation: all wordforms of a sentence are encoded together into
    one batch and decoded in a single generate() call. This is much faster than
    per-word generation on CPU (the dominant cost of seq2seq inference).
    """
    correct = 0
    total = 0
    for s in dev_sents:
        # Build source strings for non-PUNCT tokens only — PUNCT is identity
        # and excluded from lemma accuracy (matches the classifier eval).
        items = [(t.form, t.upos) for t in s.tokens if t.upos != "PUNCT"]
        if not items:
            continue
        sources = [f"{upos} {form.lower()}" for form, upos in items]
        # Encode + pad to max len in this batch.
        encoded = [tokenizer.encode(src, max_length=MAX_SOURCE_LEN) for src in sources]
        max_len = max(len(e) for e in encoded)
        input_ids = torch.tensor(
            [e + [tokenizer.pad_token_id] * (max_len - len(e)) for e in encoded],
            dtype=torch.long, device=device,
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=1,           # greedy — fastest on CPU
                do_sample=False,
            )
        # generated is (batch, seq_len); decode each row.
        for i, (form, upos) in enumerate(items):
            pred_ids = generated[i].tolist()
            pred_lemma = tokenizer.decode(pred_ids, skip_special=True).strip()
            gold_lemma = dev_gold.get(form.lower(), (form.lower(), upos))[0]
            total += 1
            if pred_lemma == gold_lemma:
                correct += 1
    return (correct / total if total else 0.0), total


# ============================================================
#  Main
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        # CRITICAL: set_num_interop_threads BREAKS performance on this CPU
        # (documented in WORKLOG §8). Do NOT add it.
        print(f"CPU intra-op threads: {torch.get_num_threads()}")
    print(f"Device: {device}  Base: {args.base_model}")

    # Resolve tokenizer BEFORE loading pool so we fail fast on tokenizer issues.
    spiece_path = _resolve_spiece(args.base_model)
    print(f"Tokenizer: sentencepiece @ {spiece_path}")

    pool = load_pool()
    print(f"Gold pool: {len(pool)} sentences")
    silver_lemmas = load_silver_lemmas(args.silver)
    print(f"Silver lemma dict: {len(silver_lemmas)} forms")

    silver_sents: list[Sentence] = []
    if args.silver_conllu:
        from kazmorph.corpus import parse_conllu as _parse
        for sf in args.silver_conllu:
            if os.path.exists(sf):
                silver_sents += _parse(sf)
        print(f"Silver sentences: {len(silver_sents)} from {len(args.silver_conllu)} file(s)")

    folds = stratified_kfold(pool, k=args.folds, seed=args.seed)
    accs: list[float] = []
    for i, (train_sents, dev_sents) in enumerate(folds):
        out_dir = args.model_dir if i == len(folds) - 1 else None
        train_with_silver = list(train_sents) + silver_sents
        best = run_fold(
            args.base_model, train_with_silver, dev_sents, silver_lemmas,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, out_dir=out_dir, fold_label=f"fold {i+1}/{len(folds)}",
            spiece_path=spiece_path,
        )
        accs.append(best)

    print("\n" + "=" * 60)
    print("  SEQ2SEQ LEMMATIZER CROSS-VALIDATION SUMMARY (exact-match lemma acc)")
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
    meta = {
        "base_model": args.base_model,
        "folds": args.folds,
        "fold_acc": accs,
        "mean_acc": mean,
        "max_source_len": MAX_SOURCE_LEN,
        "max_target_len": MAX_TARGET_LEN,
        "task": "seq2seq",
        "format": "source='<UPOS> word' target='lemma'",
    }
    with open(os.path.join(args.model_dir, "lemma_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.model_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train seq2seq Kazakh lemmatizer (mt5-small)")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="mt5 needs higher LR than BERT to overcome span-corruption "
                        "pretraining; 3e-4 is a common low-resource default")
    p.add_argument("--silver", default=os.path.join(PROJECT_ROOT, "data", "silver_lemmas.tsv"),
                   help="Silver lemma TSV (form<TAB>lemma<TAB>upos)")
    p.add_argument("--silver-conllu", nargs="*", default=None,
                   help="Silver CoNLL-U sentence file(s) to mix into training")
    p.add_argument("--model-dir",
                   default=os.path.join(PROJECT_ROOT, "models", "neural_lemma_seq2seq"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    args = p.parse_args()
    train(args)
