"""
Stage 3b (alternative): seq2seq neural lemmatizer based on mt5-small.

A character-level encoder-decoder that maps "<UPOS> wordform" → "lemma".
Unlike the prefix-length classifier in kazmorph/neural_lemma.py, a seq2seq
model can express stem alternations (қ↔ғ, к↔г) directly without a hand-written
devoice rule. This module is the inference side; training lives in
scripts/train_lemma_seq2seq.py.

The class mirrors the classifier's interface (tag(list[RMAResult]) ->
list[RMAResult], mutating t.lemma in place) so it can drop into kazmorph/
pipeline.py as a one-line swap. Soft-failure contract is preserved: if the
model directory is absent or unreadable, tag() returns tokens unchanged and
the pipeline falls back to the RMA lemma.

Performance note
----------------
Seq2seq generation is autoregressive (one forward pass per output token),
so per-word it is several times slower than the classifier's single argmax.
To stay within the latency budget set by the A/B gate (≤3x classifier), we
batch all words of a chunk into one model.generate() call and use greedy
decoding (num_beams=1) with a short max_target_len. The chunking here is
identical to the POS tagger's so the two stages align naturally.

When this module is the active lemmatizer (A/B gate passed), pipeline.py
points self.lemmatizer at NeuralLemmatizerSeq2Seq; otherwise the classifier
in kazmorph/neural_lemma.py is used and this module stays unloaded.
"""

from __future__ import annotations

import os

import torch
from transformers import MT5ForConditionalGeneration

# Import the same SentencePiece wrapper used at training time. It lives in the
# training script to keep the tokenizer-bypass fix for transformers 5.10.2 in
# exactly one place.
import scripts.train_lemma_seq2seq as _train_mod
from kazmorph.rma import RMAResult


# Decoder length cap. Matches MAX_TARGET_LEN in scripts/train_lemma_seq2seq.py.
# Kazakh lemmas rarely exceed ~18 chars; 32 gives headroom for compound PROPNs.
MAX_TARGET_LEN = 32
# Chunk size for long inputs. Mirrors the POS NeuralTagger (kazmorph/neural.py)
# and the classifier lemmatizer so chunks line up and nothing exceeds mt5's
# encoder max_position_embeddings.
CHUNK_WORDS = 100


class NeuralLemmatizerSeq2Seq:
    """Stage 3b: predict lemma via mt5-small seq2seq generation."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        if model_path and os.path.exists(os.path.join(model_path, "config.json")):
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load the fine-tuned mt5 checkpoint and its SentencePiece tokenizer.

        On CPU the model is dynamically quantized to int8 (linear layers only).
        This is critical for staying within the latency budget: a float32 mt5
        is ~7x slower than the classifier lemmatizer on the same text, which
        blows the 3x A/B gate. int8 quantization brings it to ~2.5x (measured
        on dev_split) with no measurable accuracy loss — the lemma output is
        byte-identical on spot checks (жұмысының → жұмыс). Quantization is a
        no-op on GPU (dynamic int8 is CPU-only and ignored otherwise).
        """
        self.model = MT5ForConditionalGeneration.from_pretrained(path)
        spiece = os.path.join(path, "spiece.model")
        self.tokenizer = _train_mod.MTSPTokenizer(spiece)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )

    def tag(self, tokens: list[RMAResult]) -> list[RMAResult]:
        """Predict and rewrite each token's lemma in place, or pass through.

        Soft-failure: if the model failed to load, tokens are returned
        unchanged and the pipeline falls back to the RMA lemma.
        """
        if self.model is None:
            return tokens
        if not tokens:
            return tokens

        # Chunk long inputs on sentence boundaries so the batch size per
        # generate() call stays bounded (large batches blow up mt5 decode time
        # super-linearly on CPU). Same boundary set as the POS tagger.
        _SENT_END = {".", "!", "?", "»", "…"}
        boundaries = [i for i, t in enumerate(tokens)
                      if t.token in _SENT_END or any(c in t.token for c in _SENT_END)]
        boundaries.append(len(tokens) - 1)

        start = 0
        while start < len(tokens):
            ideal_end = start + CHUNK_WORDS
            if ideal_end >= len(tokens):
                self._predict_chunk(tokens, start, len(tokens))
                break
            best_end = ideal_end
            for b in boundaries:
                if start < b <= ideal_end:
                    best_end = b + 1
            if best_end <= start + 1:
                best_end = ideal_end
            self._predict_chunk(tokens, start, best_end)
            start = best_end
        return tokens

    def _predict_chunk(self, tokens: list[RMAResult], start: int, end: int) -> None:
        """Decode lemmas for tokens[start:end] in a single batched generate() call.

        Only tokens containing alpha chars are sent through the model; everything
        else (punctuation, numbers) keeps its RMA lemma. This mirrors the
        classifier's overwrite guard and keeps generation focused on real words.
        """
        chunk = tokens[start:end]

        # Collect indices of word-tokens we actually want to lemmatize.
        word_positions: list[int] = []
        sources: list[str] = []
        for i, t in enumerate(chunk):
            if not any(c.isalpha() for c in t.token):
                continue
            word_positions.append(i)
            sources.append(f"{t.pos} {t.token.lower()}")

        if not word_positions:
            return

        # Encode + pad to the longest source in this batch.
        tok = self.tokenizer
        encoded = [tok.encode(s, max_length=64) for s in sources]
        max_src = max(len(e) for e in encoded)
        input_ids = torch.tensor(
            [e + [tok.pad_token_id] * (max_src - len(e)) for e in encoded],
            dtype=torch.long, device=self.device,
        )
        attention_mask = (input_ids != tok.pad_token_id).long()

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=1,        # greedy — fastest on CPU, near-identical acc
                do_sample=False,
            )

        # Overwrite lemma only for plausible generations. Same guard as the
        # classifier: non-empty, ≥2 chars, alpha-only output (a stray eos/pad
        # leak would otherwise corrupt good RMA lemmas).
        for i, pos in enumerate(word_positions):
            pred_ids = generated[i].tolist()
            pred_lemma = tok.decode(pred_ids, skip_special=True).strip()
            if pred_lemma and len(pred_lemma) >= 2 and any(c.isalpha() for c in pred_lemma):
                chunk[pos].lemma = pred_lemma
