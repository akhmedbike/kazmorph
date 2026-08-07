"""
Tests for the silver-weighted loss and the silver-aware data plumbing
introduced into scripts/train_neural.py and scripts/train_lemma.py.

The core invariant: at silver_weight=1.0 the new per-token weighted loss MUST
reproduce the legacy loss exactly (legacy = ``CrossEntropyLoss(ignore_index=
-100).mean()`` for POS, model-internal CE for lemma). This is what makes the
change backward-compatible — a re-run with --silver-weight 1.0 should give
bit-for-bit the same training dynamics as before.

These tests exercise the loss functions directly (not full training runs) so
they are fast and deterministic. Smoke tests of the actual trainers live in
test_neural_crf_training.py and test_neural_joint.py.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

# Make scripts/ importable as a package-equivalent.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# POS has 18 label ids (O + 17 B-<UPOS>); lemma has 25 (prefix lengths 0..24).
# Import them once so tests don't hard-code constants that drift from the trainer.
from scripts.train_neural import NUM_LABELS as POS_NUM_LABELS
from scripts.train_lemma import NUM_LABELS as LEMMA_NUM_LABELS


# ============================================================
#  POS-side loss: silver-weighting vs legacy CE
# ============================================================
class TestPosSilverWeight:
    """_weighted_pos_loss in scripts/train_neural.py."""

    def test_silver_weight_1_equals_legacy_ce(self):
        """At silver_weight=1.0 with no silver and no class weights, the new
        per-token weighted loss must equal CrossEntropyLoss(ignore_index=-100,
        reduction='mean') bit-for-bit. This pins backward compatibility."""
        from scripts.train_neural import _weighted_pos_loss
        torch.manual_seed(0)
        C = POS_NUM_LABELS
        B, T = 2, 5
        logits = torch.randn(B, T, C, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 3, -100], [1, 2, 0, -100, -100]])
        silver_mask = torch.zeros(B, T)  # all gold

        # Reference: standard CE with ignore_index=-100.
        legacy = torch.nn.functional.cross_entropy(
            logits.reshape(-1, C), labels.reshape(-1), ignore_index=-100
        )
        # New: per-token weighted at silver_weight=1.0, no class weights.
        new = _weighted_pos_loss(logits, labels, silver_mask,
                                  silver_weight=1.0, class_weights=None,
                                  device=torch.device("cpu"))
        assert abs(legacy.item() - new.item()) < 1e-5, (legacy.item(), new.item())

    def test_silver_weight_lt_1_reduces_silver_contribution(self):
        """With a MIX of gold + silver tokens, lowering silver_weight pulls the
        blended loss toward the gold-only mean.

        Implementation note: the loss is ``(ce * w).sum() / w.sum()`` — so with
        ALL tokens silver, scaling w by silver_weight cancels between numerator
        and denominator and the loss is invariant. The down-weighting only
        manifests when gold and silver coexist in a batch (which is the only
        case that matters in practice, since real batches always mix them).
        """
        from scripts.train_neural import _weighted_pos_loss
        torch.manual_seed(1)
        C = POS_NUM_LABELS
        B, T = 1, 4
        logits = torch.randn(B, T, C, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 0]])
        # Half gold (tokens 0,1), half silver (tokens 2,3).
        silver_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        loss_full = _weighted_pos_loss(logits, labels, silver_mask,
                                        silver_weight=1.0, class_weights=None,
                                        device=torch.device("cpu"))
        loss_quarter = _weighted_pos_loss(logits, labels, silver_mask,
                                           silver_weight=0.25, class_weights=None,
                                           device=torch.device("cpu"))
        # The blended loss MUST move (silver_weight is not inert when gold+silver
        # coexist). We don't pin the exact value — test_silver_mask_mixed does —
        # we only assert that the knob actually does something.
        assert abs(loss_full.item() - loss_quarter.item()) > 1e-4, \
            "silver_weight had no effect on a mixed gold+silver batch"

    def test_gold_loss_unaffected_by_silver_weight(self):
        """Silver weight should NOT affect gold tokens' loss contribution."""
        from scripts.train_neural import _weighted_pos_loss
        torch.manual_seed(2)
        C = POS_NUM_LABELS
        B, T = 1, 4
        logits = torch.randn(B, T, C, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 0]])
        silver_mask = torch.zeros(B, T)  # all gold

        l_low = _weighted_pos_loss(logits, labels, silver_mask,
                                    silver_weight=0.1, class_weights=None,
                                    device=torch.device("cpu"))
        l_high = _weighted_pos_loss(logits, labels, silver_mask,
                                     silver_weight=1.0, class_weights=None,
                                     device=torch.device("cpu"))
        # With no silver tokens, the silver_weight knob is inert.
        assert abs(l_low.item() - l_high.item()) < 1e-6

    def test_silver_mask_mixed(self):
        """Mix of gold + silver tokens: silver_weight applies per-token."""
        from scripts.train_neural import _weighted_pos_loss
        torch.manual_seed(3)
        C = POS_NUM_LABELS
        B, T = 1, 4
        logits = torch.randn(B, T, C, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 0]])
        # Tokens 0,1 gold; tokens 2,3 silver.
        silver_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        # Per-token CE first.
        ce_per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, C), labels.reshape(-1), reduction="none"
        )  # shape (4,)
        # At weight 0.5: gold contributes ce[0]+ce[1] (w=1), silver contributes
        # (ce[2]+ce[3]) * 0.5. Denominator = 2*1 + 2*0.5 = 3.
        expected = (ce_per_token[0] + ce_per_token[1] +
                    (ce_per_token[2] + ce_per_token[3]) * 0.5) / 3.0
        actual = _weighted_pos_loss(logits, labels, silver_mask,
                                     silver_weight=0.5, class_weights=None,
                                     device=torch.device("cpu"))
        assert abs(expected.item() - actual.item()) < 1e-5, (expected.item(), actual.item())


# ============================================================
#  Lemma-side loss: silver-weighting vs legacy CE
# ============================================================
class TestLemmaSilverWeight:
    """_weighted_lemma_loss in scripts/train_lemma.py."""

    def test_silver_weight_1_equals_legacy_lemma_ce(self):
        """At silver_weight=1.0 with no silver, must equal CE(ignore_index=-100,
        reduction='mean'). The lemma task used to call model.loss directly,
        which is exactly that CE — so this is the legacy-fidelity test."""
        from scripts.train_lemma import _weighted_lemma_loss, NUM_LABELS
        torch.manual_seed(0)
        B, T = 2, 5
        logits = torch.randn(B, T, NUM_LABELS, requires_grad=True)
        labels = torch.tensor([[0, 5, 3, 12, -100], [2, 8, 0, -100, -100]])
        silver_mask = torch.zeros(B, T)

        legacy = torch.nn.functional.cross_entropy(
            logits.reshape(-1, NUM_LABELS), labels.reshape(-1), ignore_index=-100
        )
        new = _weighted_lemma_loss(logits, labels, silver_mask, silver_weight=1.0)
        assert abs(legacy.item() - new.item()) < 1e-5, (legacy.item(), new.item())

    def test_silver_weight_zero_kills_silver_loss(self):
        """At silver_weight=0.0, silver tokens contribute nothing — but gold
        tokens still do. This is the limiting case and confirms the per-token
        weighting is multiplicative."""
        from scripts.train_lemma import _weighted_lemma_loss, NUM_LABELS
        torch.manual_seed(4)
        B, T = 1, 4
        logits = torch.randn(B, T, NUM_LABELS, requires_grad=True)
        labels = torch.tensor([[3, 7, 1, 9]])
        # First two gold, last two silver.
        silver_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        loss = _weighted_lemma_loss(logits, labels, silver_mask, silver_weight=0.0)
        # Only gold tokens 0 and 1 contribute, both with weight 1.0.
        gold_ce = torch.nn.functional.cross_entropy(
            logits[0, :2], labels[0, :2], reduction="none"
        )
        expected = gold_ce.sum() / 2.0
        assert abs(loss.item() - expected.item()) < 1e-5, (loss.item(), expected.item())


# ============================================================
#  Dataset silver_mask plumbing
# ============================================================
class TestDatasetSilverMask:
    """POSTokenDataset/LemmaDataset tag silver positions correctly."""

    def _make_sentences(self, n_gold: int, n_silver: int):
        """Tiny fake gold/silver sentences for the dataset constructors."""
        from kazmorph.corpus import Sentence, Token
        sents = []
        for i in range(n_gold + n_silver):
            sents.append(Sentence(
                sent_id=f"s{i}",
                tokens=(Token(form=f"a{i}", lemma=f"a{i}", upos="NOUN"),
                         Token(form=f"b{i}", lemma=f"b{i}", upos="VERB")),
            ))
        return sents

    def test_pos_dataset_silver_mask_gold_is_zero_silver_is_one(self):
        from scripts.train_neural import POSTokenDataset
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("Eraly-ml/KazBERT")
        except Exception:
            pytest.skip("KazBERT tokenizer not available offline")
        sents = self._make_sentences(n_gold=2, n_silver=1)
        ds = POSTokenDataset(sents, tok, tag2id={"O": 0, "B-NOUN": 1, "B-VERB": 2},
                              n_gold=2)
        # Index 0 is gold → mask should be all 0 on real tokens.
        item0 = ds[0]
        real0 = [m for m, l in zip(item0["silver_mask"], item0["labels"]) if l != -100]
        assert all(m == 0.0 for m in real0)
        # Index 2 is silver → mask should be all 1 on real tokens.
        item2 = ds[2]
        real2 = [m for m, l in zip(item2["silver_mask"], item2["labels"]) if l != -100]
        assert all(m == 1.0 for m in real2)

    def test_pos_dataset_no_n_gold_treats_all_as_gold(self):
        """n_gold=None (legacy default) must mark everything as gold."""
        from scripts.train_neural import POSTokenDataset
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("Eraly-ml/KazBERT")
        except Exception:
            pytest.skip("KazBERT tokenizer not available offline")
        sents = self._make_sentences(n_gold=0, n_silver=3)
        ds = POSTokenDataset(sents, tok, tag2id={"O": 0, "B-NOUN": 1, "B-VERB": 2})
        for i in range(len(ds)):
            item = ds[i]
            real = [m for m, l in zip(item["silver_mask"], item["labels"]) if l != -100]
            assert all(m == 0.0 for m in real), f"idx {i} marked silver in legacy mode"


# ============================================================
#  Collator
# ============================================================
class TestSilverAwareCollator:
    """The custom collator must pad silver_mask in lockstep with labels."""

    def test_collator_pads_silver_mask_to_batch_max(self):
        from scripts.train_neural import _SilverAwareCollator
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("Eraly-ml/KazBERT")
        except Exception:
            pytest.skip("KazBERT tokenizer not available offline")
        collator = _SilverAwareCollator(tokenizer=tok, label_pad_token_id=-100)
        # Two sentences of different lengths.
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
             "labels": [0, 1, 0], "silver_mask": [0.0, 1.0, 0.0]},
            {"input_ids": [4, 5], "attention_mask": [1, 1],
             "labels": [2, 0], "silver_mask": [1.0, 1.0]},
        ]
        batch = collator(features)
        # silver_mask must have shape (2, max_len) where max_len matches labels.
        assert batch["silver_mask"].shape == batch["labels"].shape
        # Padded positions must be 0.0 (silver loss weight contributes nothing).
        assert batch["silver_mask"][1, 2].item() == 0.0
        # Real positions preserved.
        assert batch["silver_mask"][0, 1].item() == 1.0
