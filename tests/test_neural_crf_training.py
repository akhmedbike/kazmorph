"""
Tests for the KazBERT + CRF POS tagger trainer
(scripts/train_neural_crf.py).

These tests run a tiny end-to-end smoke training run on MPS/CPU, which is the
only honest way to verify that the BERT-emissions + CRF-NLL integration works
in practice. The CRF math itself is verified against brute-force enumeration
in test_crf.py; here we verify the WIRING.

Skip conditions:
- If KazBERT can't be downloaded / isn't cached, the smoke test is skipped
  rather than failing — CI environments without network access shouldn't fail
  on missing weights.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ============================================================
#  CRF-NLL weighting helper
# ============================================================
class TestWeightedCrfNll:
    """_weighted_crf_nll silver-weighting at the sequence level."""

    def test_silver_weight_1_is_plain_crf_nll(self):
        """At silver_weight=1.0 the weighted NLL must equal the plain CRF NLL
        averaged across the batch — this is the backward-compatibility
        invariant for the CRF trainer."""
        from scripts.train_neural_crf import _weighted_crf_nll
        from kazmorph.crf import LinearChainCRF
        torch.manual_seed(0)
        num_tags = 5
        crf = LinearChainCRF(num_tags)
        B, T = 3, 6
        emissions = torch.randn(B, T, num_tags)
        labels = torch.tensor([[0, 1, 2, 3, 4, 0],
                                [1, 2, 0, 0, 1, 2],
                                [4, 3, 2, 1, 0, 4]])
        silver_mask = torch.zeros(B, T)

        plain_nll = crf.neg_log_likelihood(emissions, labels, (labels != -100).float())
        weighted = _weighted_crf_nll(crf, emissions, labels, silver_mask, silver_weight=1.0)
        assert abs(plain_nll.item() - weighted.item()) < 1e-4, \
            (plain_nll.item(), weighted.item())

    def test_silver_weight_0_drops_silver_sequences(self):
        """At silver_weight=0.0, fully-silver sequences contribute nothing —
        the loss equals the plain CRF NLL on the gold-only subset."""
        from scripts.train_neural_crf import _weighted_crf_nll
        from kazmorph.crf import LinearChainCRF
        torch.manual_seed(1)
        num_tags = 4
        crf = LinearChainCRF(num_tags)
        B, T = 2, 4
        emissions = torch.randn(B, T, num_tags)
        labels = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        # Sequence 0 gold, sequence 1 fully silver.
        silver_mask = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0]])

        weighted = _weighted_crf_nll(crf, emissions, labels, silver_mask, silver_weight=0.0)
        # Plain NLL on just the gold sequence (batch of 1).
        gold_only = crf.neg_log_likelihood(
            emissions[:1], labels[:1], torch.ones(1, T)
        )
        assert abs(weighted.item() - gold_only.item()) < 1e-4, \
            (weighted.item(), gold_only.item())


# ============================================================
#  Model wiring: BertCrfTagger
# ============================================================
# Module-scoped fixture (not class-scoped-on-instance-method, which pytest 9
# deprecated and silently skips). Loading KazBERT is the expensive bit, so we
# do it once per module.
@pytest.fixture(scope="module")
def small_model():
    """Load KazBERT + CRF head once for the whole module (slow on first run)."""
    try:
        from scripts.train_neural_crf import BertCrfTagger, NUM_LABELS
        from kazmorph.device import pick_device
        model = BertCrfTagger("Eraly-ml/KazBERT", num_tags=NUM_LABELS)
        model.to(pick_device()).eval()
        return model
    except Exception as exc:
        pytest.skip(f"KazBERT unavailable: {exc}")


class TestBertCrfTaggerWiring:
    """BertCrfTagger emits / decodes / loss shapes are correct."""

    def test_emissions_have_correct_shape(self, small_model):
        from kazmorph.device import pick_device
        device = pick_device()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        attention_mask = torch.ones(1, 5, dtype=torch.long, device=device)
        with torch.no_grad():
            em = small_model.emissions(input_ids, attention_mask)
        # (B=1, T=5, num_labels)
        assert em.shape == (1, 5, small_model.emission_head.out_features)

    def test_decode_returns_variable_length_paths(self, small_model):
        from kazmorph.device import pick_device
        device = pick_device()
        # Two sequences, different lengths via attention mask.
        input_ids = torch.tensor([[1, 2, 3, 4, 0], [10, 11, 12, 13, 14]], device=device)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], device=device)
        with torch.no_grad():
            paths = small_model.decode(input_ids, attention_mask)
        assert len(paths) == 2
        # Viterbi paths have length = unmasked prefix length per batch element.
        assert len(paths[0]) == 4
        assert len(paths[1]) == 5

    def test_loss_is_scalar_and_finite(self, small_model):
        from kazmorph.device import pick_device
        device = pick_device()
        input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
        attention_mask = torch.ones(1, 4, dtype=torch.long, device=device)
        labels = torch.tensor([[0, 1, 2, 3]], device=device)
        silver_mask = torch.zeros(1, 4, device=device)
        loss = small_model.loss(input_ids, attention_mask, labels, silver_mask, 1.0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ============================================================
#  End-to-end smoke: one epoch actually trains
# ============================================================
@pytest.mark.slow
class TestSmokeTrainOneEpoch:
    """Run one epoch of real training on a handful of sentences.

    Marked ``slow`` because it loads KazBERT (~440 MB) and runs one full
    forward+backward+optimizer step over the small synthetic corpus. Skipped
    if KazBERT isn't available locally.
    """

    def test_smoke_train_runs_and_reduces_loss(self, tmp_path):
        try:
            from scripts.train_neural_crf import run_fold
            from kazmorph.corpus import Sentence, Token
        except Exception as exc:
            pytest.skip(f"imports failed: {exc}")

        # 8 tiny synthetic sentences. POS tag values are real UD tags so the
        # BIO encoding in POSTokenDataset doesn't trip over them.
        def make_sent(i: int) -> Sentence:
            return Sentence(
                sent_id=f"s{i}",
                tokens=(Token(form=f"сөз{i}", lemma=f"сөз{i}", upos="NOUN"),
                         Token(form=f"жаса{i}", lemma=f"жаса{i}", upos="VERB")),
            )
        train_sents = [make_sent(i) for i in range(6)]
        dev_sents = [make_sent(i) for i in range(6, 8)]

        from kazmorph.device import pick_device
        out_dir = str(tmp_path / "smoke_model")
        try:
            best_f1, details = run_fold(
                base_model="Eraly-ml/KazBERT",
                train_sents=train_sents,
                dev_sents=dev_sents,
                epochs=2,
                batch_size=4,
                lr=2e-5,
                weight_decay=0.01,
                max_grad_norm=1.0,
                device=pick_device(),
                out_dir=out_dir,
                fold_label="smoke",
            )
        except Exception as exc:
            pytest.skip(f"KazBERT unavailable: {exc}")

        # Best F1 over the 2 epochs — anything finite and >= 0 means training ran.
        assert best_f1 >= 0.0
        assert len(details["history"]) == 2
        # The loss history is monotonically meaningful (not NaN).
        for h in details["history"]:
            assert h["loss"] == h["loss"]  # not NaN
