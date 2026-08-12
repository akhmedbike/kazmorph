"""
Stage 3: Neural POS tagger (XLM-RoBERTa-base fine-tuned on UD Kazakh-KTB)

Fine-tuned for token classification with 17 Universal POS tags.
Achieves ~92% F1 on dev split (standalone).

When model is not available, passes tokens through unchanged (stub mode).
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from kazmorph.rma import RMAResult


# 17 Universal POS tags
POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]


class NeuralTagger:
    """Stage 3: Neural POS tagger based on fine-tuned XLM-RoBERTa."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.id2tag: dict[int, str] = {}
        self.device = torch.device("cpu")
        # Temperature for softmax calibration (T=1 = uncalibrated). Fitted by
        # scripts/calibrate.py to make confidence reflect real accuracy.
        self.temperature = 1.0
        # Optional Region-Dependent TS: one temperature per confidence bin,
        # chosen at inference by the uncalibrated max prob. When set, this
        # overrides the scalar ``temperature``. See scripts/calibrate.py --mode.
        self.region_temps: list[dict] | None = None

        if model_path and os.path.exists(os.path.join(model_path, "config.json")):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load a fine-tuned transformer model and tag mapping."""
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load tag mapping
        tag_map_path = os.path.join(path, "tag_map.json")
        if os.path.exists(tag_map_path):
            with open(tag_map_path) as f:
                tag_map = json.load(f)
            self.id2tag = {int(k): v for k, v in tag_map["id2tag"].items()}
        else:
            # Fallback: build from model config
            self.id2tag = {i: f"B-{t}" for i, t in enumerate(POS_TAGS, 1)}
            self.id2tag[0] = "O"

        # Load calibration temperature if fitted (see kazmorph.calibration).
        calib_path = os.path.join(path, "calibration.json")
        if os.path.exists(calib_path):
            try:
                with open(calib_path) as f:
                    payload = json.load(f)
                if payload.get("mode") == "region":
                    # Region-Dependent TS: list of {lo, hi, temperature}.
                    self.region_temps = payload.get("regions")
                    self.temperature = 1.0
                else:
                    self.temperature = float(payload.get("temperature", 1.0))
                    self.region_temps = None
            except (ValueError, OSError, TypeError):
                self.temperature = 1.0
                self.region_temps = None

    def tag(self, tokens: list[RMAResult]) -> list[RMAResult]:
        """Apply neural POS tagging or pass through (stub)."""
        if self.model is None:
            return tokens
        return self._model_predict(tokens)

    def _model_predict(self, tokens: list[RMAResult]) -> list[RMAResult]:
        """
        Run inference: tokenize -> forward pass -> argmax -> update POS.

        Long inputs are processed in word-level chunks of CHUNK_WORDS tokens so
        nothing falls outside the model's max_length=128 window. Without this,
        pasting many sentences together (e.g. a 500-word document) silently
        dropped the POS/confidence of every word past position ~100, leaving
        them at the RMA fallback (often UNK / 0.2) — the "low confidence on
        long text" failure mode. Chunks are split on sentence-final punctuation
        when possible to keep context boundaries natural.
        """
        CHUNK_WORDS = 100  # <= 128 with headroom for [CLS]/[SEP] and subwords

        # Build word indices of sentence boundaries (last token of a sentence =
        # a punctuation token) so we can cut between sentences, not mid-word.
        _SENT_END = {".", "!", "?", "»", "…"}
        boundaries = [i for i, t in enumerate(tokens)
                      if t.token in _SENT_END or any(c in t.token for c in _SENT_END)]
        boundaries.append(len(tokens) - 1)

        # Walk the token list in chunks of at most CHUNK_WORDS, preferring to
        # cut at the nearest sentence boundary at or before the chunk-size limit.
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < len(tokens):
            ideal_end = start + CHUNK_WORDS
            if ideal_end >= len(tokens):
                chunks.append((start, len(tokens)))
                break
            # nearest boundary at or before ideal_end, within a lookback window
            best_end = ideal_end
            for b in boundaries:
                if start < b <= ideal_end:
                    best_end = b + 1  # include the punctuation token
            if best_end <= start + 1:  # avoid pathological tiny chunks
                best_end = ideal_end
            chunks.append((start, best_end))
            start = best_end

        for c_start, c_end in chunks:
            self._predict_chunk(tokens, c_start, c_end)
        return tokens

    def _calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax(logits / T), with T either a global scalar or per-token
        (Region-Dependent TS, chosen by the uncalibrated max prob). Both leave
        argmax unchanged. ``logits`` has shape (1, seq, num_labels)."""
        if self.region_temps:
            seq_logits = logits[0]                      # (seq, C)
            raw = torch.softmax(seq_logits, dim=-1).max(dim=-1).values  # (seq,)
            T = torch.ones_like(raw)
            for reg in self.region_temps:
                lo, hi, t = reg["lo"], reg["hi"], reg["temperature"]
                mask = (raw > lo) & (raw <= hi)
                T = torch.where(mask, torch.full_like(T, float(t)), T)
            return torch.softmax(seq_logits / T.unsqueeze(-1), dim=-1).unsqueeze(0)
        return torch.softmax(logits / self.temperature, dim=-1)

    def _predict_chunk(self, tokens: list[RMAResult], start: int, end: int) -> None:
        """Predict POS + confidence for tokens[start:end] in one forward pass."""
        chunk = tokens[start:end]
        words = [t.token for t in chunk]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Softmax with optional calibration. The max softmax value is the
            # model's real probability for the predicted tag — replacing the
            # old heuristic (RMA confidence + 0.15) which left 77% of tokens
            # stuck at 1.0 while real accuracy was 82%. Two modes: scalar T
            # (Guo 2017) or per-region T (RD-TS); both leave argmax unchanged.
            probs = self._calibrated_probs(logits)
            predictions = probs.argmax(dim=-1)[0].tolist()
            max_probs = probs.max(dim=-1).values[0].tolist()

        # Align predictions to original words (first subword token)
        seen_words = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen_words:
                continue
            seen_words.add(wid)

            pred_tag = self.id2tag.get(predictions[idx], "O")
            # Always overwrite confidence with the real softmax probability of
            # the predicted tag — even when the POS doesn't change. This is what
            # makes confidence honest: a confirmed NOUN at softmax 0.55 is a
            # genuinely uncertain prediction, not a 0.95 one.
            chunk[wid].confidence = round(float(max_probs[idx]), 4)
            # Update POS when the model produced a valid BIO tag.
            if pred_tag.startswith("B-"):
                pos = pred_tag[2:]
                if pos in POS_TAGS:
                    chunk[wid].pos = pos
