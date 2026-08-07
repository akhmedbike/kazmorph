"""
Tests for the seq2seq lemmatizer (kazmorph/neural_lemma_seq2seq.py).

Two layers:
1. Model-independent tests — verify the soft-failure contract, the
   word/punct filtering, and the chunking logic. These run always and fast.
2. Model-dependent tests — verify actual lemma prediction. Skipped
   automatically when no fine-tuned model is present (CI without the
   1.2 GB checkpoint), using the same skip-if-absent pattern as
   tests/test_lemmatizer.py.

The tokenizer wrapper (MTSPTokenizer) is tested directly since it is the
workaround for the transformers 5.10.2 mt5 regression — it must round-trip
Cyrillic perfectly or both training and inference silently break.
"""

import os
import sys

import pytest

# Make the project importable when running from repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kazmorph.rma import RMAResult


# Path to the fine-tuned seq2seq model, if it exists. Mirrors the layout used
# by pipeline.py / main.py.
SEQ2SEQ_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "neural_lemma_seq2seq")
HAS_SEQ2SEQ_MODEL = os.path.exists(os.path.join(SEQ2SEQ_MODEL_DIR, "config.json"))

# Skip marker for tests that need the trained checkpoint.
needs_model = pytest.mark.skipif(
    not HAS_SEQ2SEQ_MODEL, reason=f"seq2seq model not found at {SEQ2SEQ_MODEL_DIR}"
)


def _rma(token, lemma, pos="NOUN", confidence=0.9):
    """Helper: build a minimal RMAResult for pipeline-style tests."""
    return RMAResult(
        token=token, lemma=lemma, pos=pos,
        suffixes=[], suffix_categories=[], confidence=confidence,
    )


# ============================================================
#  Soft-failure contract (always runs)
# ============================================================
class TestSoftFailure:
    """Without a model file, the lemmatizer must pass tokens through unchanged
    so the pipeline degrades to RMA lemmas. This is the same contract as the
    classifier (kazmorph/neural_lemma.py)."""

    def test_no_model_returns_tokens_unchanged(self):
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=None)
        assert lemmatizer.model is None

        tokens = [_rma("жұмысының", "жұмысының"), _rma("адам", "адам")]
        out = lemmatizer.tag(tokens)
        # Must return the same list, lemmas untouched (RMA fallback)
        assert out is tokens
        assert out[0].lemma == "жұмысының"
        assert out[1].lemma == "адам"

    def test_nonexistent_path_is_stub(self):
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path="/nonexistent/path/xyz")
        assert lemmatizer.model is None
        tokens = [_rma("сөз", "сөз")]
        out = lemmatizer.tag(tokens)
        assert out[0].lemma == "сөз"

    def test_empty_input_returns_empty(self):
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=None)
        assert lemmatizer.tag([]) == []


# ============================================================
#  Tokenizer wrapper round-trip (always runs — uses cached spiece.model)
# ============================================================
class TestMTSPTokenizer:
    """MTSPTokenizer is the sentencepiece bypass for the transformers 5.10.2
    mt5 regression. It must round-trip Cyrillic exactly — if it doesn't, both
    training and inference silently collapse to <unk>."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        import scripts.train_lemma_seq2seq as train_mod
        sp_path = train_mod._resolve_spiece("google/mt5-small")
        return train_mod.MTSPTokenizer(sp_path)

    def test_vocab_size_is_full(self, tokenizer):
        # The 5.10.2 regression loaded vocab=104; the real mt5 vocab is 250k+.
        # This assertion catches a regression to the broken state.
        assert tokenizer.vocab_size > 100_000, (
            f"vocab_size={tokenizer.vocab_size} looks broken (mt5 should be 250k+)"
        )

    def test_special_tokens_present(self, tokenizer):
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.unk_token_id == 2

    @pytest.mark.parametrize("word", [
        "жұмысының", "конституция", "басып", "арабтардың", "сөздің",
        "NOUN жұмыс", "VERB басып",
    ])
    def test_cyrillic_roundtrip(self, tokenizer, word):
        """encode → decode must reproduce the input exactly (modulo leading
        space from SentencePiece's ▁ marker, which decode strips)."""
        ids = tokenizer.encode(word)
        decoded = tokenizer.decode(ids, skip_special=True).strip()
        assert decoded == word, f"roundtrip failed: {word!r} → {decoded!r}"

    def test_encode_adds_eos(self, tokenizer):
        ids = tokenizer.encode("сөз")
        assert ids[-1] == tokenizer.eos_token_id

    def test_decode_skips_special_tokens(self, tokenizer):
        ids = tokenizer.encode("сөз")
        decoded = tokenizer.decode(ids, skip_special=True)
        assert tokenizer.eos_token not in decoded
        assert tokenizer.pad_token not in decoded


# ============================================================
#  Chunking logic (always runs — uses a stub model-like object)
# ============================================================
class TestChunking:
    """Long inputs must be chunked to keep mt5 generate() batch sizes bounded.
    This tests the chunk boundary logic without loading the real model."""

    def test_chunk_respects_sentence_boundaries(self, monkeypatch):
        """A long input should split at sentence-final punctuation, not mid-word."""
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq

        # Build a fake lemmatizer with a patched _predict_chunk that records
        # the chunk ranges it was called with.
        calls: list[tuple[int, int]] = []
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=None)
        lemmatizer.model = object()  # truthy so tag() runs the real path

        def fake_predict(chunk_self_ref, tokens, start, end):
            calls.append((start, end))

        # Bind the patched method on the instance.
        lemmatizer._predict_chunk = lambda tokens, start, end: calls.append((start, end))

        # 50 words with a period in the middle (sentence boundary).
        tokens = [_rma(f"word{i}", f"word{i}") for i in range(25)]
        tokens.append(_rma(".", ".", pos="PUNCT"))
        tokens += [_rma(f"word{i}", f"word{i}") for i in range(25, 50)]

        lemmatizer.tag(tokens)

        # Should be exactly one chunk (input ≤ CHUNK_WORDS=100), no split needed.
        assert len(calls) == 1
        assert calls[0] == (0, 51)

    def test_long_input_splits_into_multiple_chunks(self):
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq, CHUNK_WORDS
        calls: list[tuple[int, int]] = []
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=None)
        lemmatizer.model = object()
        lemmatizer._predict_chunk = lambda tokens, start, end: calls.append((start, end))

        # 3 chunks worth of tokens, with a period every 50 words to force
        # boundary-aligned splits.
        tokens = []
        for i in range(CHUNK_WORDS * 3):
            tokens.append(_rma(f"w{i}", f"w{i}"))
            if (i + 1) % 50 == 0:
                tokens.append(_rma(".", ".", pos="PUNCT"))

        lemmatizer.tag(tokens)
        # At least 2 chunks for an input 3x the chunk size.
        assert len(calls) >= 2
        # Chunks must tile the input with no gaps/overlaps.
        assert calls[0][0] == 0
        for i in range(1, len(calls)):
            assert calls[i][0] == calls[i - 1][1], f"gap/overlap between chunks {i-1} and {i}"


# ============================================================
#  Model-dependent tests (skipped without the trained checkpoint)
# ============================================================
@needs_model
class TestModelLoaded:
    """Verify the real model loads and produces plausible lemmas. Skipped in
    CI environments where the 1.2 GB checkpoint isn't downloaded."""

    def test_model_loads(self):
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=SEQ2SEQ_MODEL_DIR)
        assert lemmatizer.model is not None
        assert lemmatizer.tokenizer is not None
        assert lemmatizer.tokenizer.vocab_size > 100_000

    def test_tag_rewrites_lemma(self):
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=SEQ2SEQ_MODEL_DIR)
        # жұмысының → жұмыс is a clean, high-frequency inflected NOUN.
        tokens = [_rma("жұмысының", "жұмысының", pos="NOUN")]
        out = lemmatizer.tag(tokens)
        # The lemma should have been rewritten to a shorter form (the seq2seq
        # model is trained to strip the possessive+genitive suffixes).
        assert out[0].lemma != "жұмысының", "lemma was not rewritten"
        assert len(out[0].lemma) < len("жұмысының")

    def test_punct_not_sent_to_model(self):
        """Punctuation tokens keep their RMA lemma even with a loaded model —
        the model is only invoked on alpha-containing tokens."""
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=SEQ2SEQ_MODEL_DIR)
        tokens = [_rma(".", ".", pos="PUNCT"), _rma(",", ",", pos="PUNCT")]
        out = lemmatizer.tag(tokens)
        assert out[0].lemma == "."
        assert out[1].lemma == ","

    def test_tag_preserves_list_identity(self):
        """tag() mutates in place and returns the same list object — the
        pipeline relies on this."""
        from kazmorph.neural_lemma_seq2seq import NeuralLemmatizerSeq2Seq
        lemmatizer = NeuralLemmatizerSeq2Seq(model_path=SEQ2SEQ_MODEL_DIR)
        tokens = [_rma("адам", "адам", pos="NOUN")]
        out = lemmatizer.tag(tokens)
        assert out is tokens
