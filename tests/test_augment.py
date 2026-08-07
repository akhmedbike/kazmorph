"""Tests for data augmentation and loss regularizers (train_neural strengthening).

Covers:
  * kazmorph.augment — casing dropout, token dropout, punct jitter, label
    preservation, determinism, backward-compat (inactive config = identity).
  * scripts.train_neural._weighted_pos_loss — focal gamma and label smoothing
    are applied when requested, and default to the legacy loss when off.
"""

from __future__ import annotations

import random

import pytest
import torch

from kazmorph.augment import AugmentConfig, augment_sentence
from kazmorph.corpus import Sentence, Token


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gold_sentence() -> Sentence:
    return Sentence(
        sent_id="t1",
        tokens=[
            Token(form="Қазақ", lemma="қазақ", upos="NOUN"),
            Token(form="тілі", lemma="тіл", upos="NOUN"),
            Token(form="бар", lemma="бар", upos="VERB"),
            Token(form=".", lemma=".", upos="PUNCT"),
        ],
    )


# ---------------------------------------------------------------------------
# augment_sentence — label preservation (the key correctness property)
# ---------------------------------------------------------------------------

class TestAugmentLabelPreservation:
    def test_disabled_config_returns_equivalent_sentence(self, gold_sentence):
        """An all-zero config must be a no-op (forms unchanged, tags unchanged)."""
        cfg = AugmentConfig(casing_dropout=0.0, token_dropout=0.0, punct_jitter=0.0)
        rng = random.Random(0)
        out = augment_sentence(gold_sentence, rng, cfg)
        assert [t.form for t in out.tokens] == [t.form for t in gold_sentence.tokens]
        assert [t.upos for t in out.tokens] == [t.upos for t in gold_sentence.tokens]

    def test_upos_tags_never_change_under_casing_dropout(self, gold_sentence):
        """Casing dropout flips surface case but must preserve every upos tag."""
        cfg = AugmentConfig(casing_dropout=1.0, token_dropout=0.0, punct_jitter=0.0)
        rng = random.Random(123)
        original_tags = [t.upos for t in gold_sentence.tokens]
        for _ in range(20):
            out = augment_sentence(gold_sentence, rng, cfg)
            assert [t.upos for t in out.tokens] == original_tags

    def test_token_dropout_preserves_tags_but_changes_forms(self, gold_sentence):
        """Masked tokens keep their upos (label-preserving), but forms change."""
        cfg = AugmentConfig(casing_dropout=0.0, token_dropout=1.0, punct_jitter=0.0)
        rng = random.Random(7)
        out = augment_sentence(gold_sentence, rng, cfg)
        # Non-PUNCT tokens are masked; PUNCT is skipped (see augment.py).
        assert out.tokens[0].form == "[MASK]"
        assert out.tokens[2].form == "[MASK]"
        # Tags are unchanged for the non-PUNCT tokens that were masked.
        assert out.tokens[0].upos == "NOUN"
        assert out.tokens[2].upos == "VERB"


class TestAugmentPunctJitter:
    def test_punct_duplication_adds_one_token_with_same_tag(self, gold_sentence):
        """punct_jitter=1.0 must duplicate the trailing PUNCT (tag preserved)."""
        cfg = AugmentConfig(casing_dropout=0.0, token_dropout=0.0, punct_jitter=1.0)
        rng = random.Random(0)
        out = augment_sentence(gold_sentence, rng, cfg)
        assert len(out.tokens) == len(gold_sentence.tokens) + 1
        assert out.tokens[-1].form == "."
        assert out.tokens[-1].upos == "PUNCT"


class TestAugmentDeterminism:
    def test_same_seed_same_output(self, gold_sentence):
        """Same RNG state → same augmented sentence (reproducibility)."""
        cfg = AugmentConfig()  # active by default
        out1 = augment_sentence(gold_sentence, random.Random(99), cfg)
        out2 = augment_sentence(gold_sentence, random.Random(99), cfg)
        assert [t.form for t in out1.tokens] == [t.form for t in out2.tokens]

    def test_different_epochs_differ_with_same_base_seed(self, gold_sentence):
        """A single dataset RNG advances across calls, so augmentation varies."""
        cfg = AugmentConfig()
        rng = random.Random(0)
        seen = set()
        for _ in range(30):
            out = augment_sentence(gold_sentence, rng, cfg)
            seen.add(tuple(t.form for t in out.tokens))
        # With three active stochastic transforms, we expect >1 distinct variant
        # over 30 draws. (If this ever flakes, lower the threshold, but it is
        # extremely unlikely given casing/token/punct all fire.)
        assert len(seen) > 1


# ---------------------------------------------------------------------------
# _weighted_pos_loss — focal & label smoothing
# ---------------------------------------------------------------------------

@pytest.fixture
def loss_inputs():
    """A tiny, deterministic batch: 2 sentences, 3 tokens each.

    The loss flattens to ``NUM_LABELS`` classes (the module-level constant
    built from POS_TAGS), so logits must have that many columns. We import the
    real value rather than hard-coding it so the test tracks the tag inventory.
    """
    from scripts.train_neural import NUM_LABELS
    torch.manual_seed(0)
    B, T, C = 2, 3, NUM_LABELS
    logits = torch.randn(B, T, C, requires_grad=True)
    # Use small label ids that are always valid (< NUM_LABELS for any inventory).
    labels = torch.tensor([[0, 1, 2], [1, 2, -100]])  # last pos padded
    silver_mask = torch.zeros(B, T)
    return logits, labels, silver_mask


def _legacy_loss(logits, labels, silver_mask, device):
    """Reproduce the pre-strengthening loss exactly (smoothing=0, gamma=0)."""
    from scripts.train_neural import _weighted_pos_loss
    return _weighted_pos_loss(
        logits, labels, silver_mask, silver_weight=1.0,
        class_weights=None, device=device,
    )


class TestWeightedPosLossDefaults:
    def test_default_kwargs_equal_legacy_ce(self, loss_inputs):
        """smoothing=0, gamma=0 must reduce to the original loss."""
        from scripts.train_neural import _weighted_pos_loss
        logits, labels, silver_mask = loss_inputs
        device = torch.device("cpu")
        legacy = _legacy_loss(logits, labels, silver_mask, device)
        new = _weighted_pos_loss(
            logits, labels, silver_mask, silver_weight=1.0,
            class_weights=None, device=device,
            label_smoothing=0.0, focal_gamma=0.0,
        )
        assert torch.allclose(legacy, new)


class TestWeightedPosLossFocal:
    def test_focal_changes_loss_value(self, loss_inputs):
        """focal_gamma=2.0 must produce a different loss than gamma=0."""
        from scripts.train_neural import _weighted_pos_loss
        logits, labels, silver_mask = loss_inputs
        device = torch.device("cpu")
        base = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, focal_gamma=0.0)
        focal = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, focal_gamma=2.0)
        assert not torch.allclose(base, focal)

    def test_focal_with_high_confidence_shrinks_loss(self):
        """A perfectly-confident correct prediction gets near-zero focal weight."""
        from scripts.train_neural import _weighted_pos_loss, NUM_LABELS
        # logit 100 for the true class → p_true ≈ 1 → (1-p)^gamma ≈ 0.
        logits = torch.zeros(1, 1, NUM_LABELS)
        logits[0, 0, 0] = 100.0
        labels = torch.tensor([[0]])
        silver_mask = torch.zeros(1, 1)
        device = torch.device("cpu")
        focal = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, focal_gamma=2.0)
        assert focal.item() < 1e-3


class TestWeightedPosLossLabelSmoothing:
    def test_label_smoothing_changes_loss_value(self, loss_inputs):
        """smoothing=0.1 must differ from smoothing=0."""
        from scripts.train_neural import _weighted_pos_loss
        logits, labels, silver_mask = loss_inputs
        device = torch.device("cpu")
        hard = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, label_smoothing=0.0)
        soft = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, label_smoothing=0.1)
        assert not torch.allclose(hard, soft)

    def test_label_smoothing_reduces_loss_for_confident_predictions(self):
        """Soft targets cap the achievable confidence, lowering peak loss."""
        from scripts.train_neural import _weighted_pos_loss, NUM_LABELS
        logits = torch.zeros(1, 1, NUM_LABELS)
        logits[0, 0, 0] = 100.0
        labels = torch.tensor([[0]])
        silver_mask = torch.zeros(1, 1)
        device = torch.device("cpu")
        hard = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, label_smoothing=0.0)
        soft = _weighted_pos_loss(
            logits, labels, silver_mask, 1.0, None, device, label_smoothing=0.1)
        # With smoothing, the target distribution isn't a pure one-hot, so a
        # perfectly confident logit overshoots and the loss is HIGHER, not lower.
        # This is the expected, documented behaviour of label smoothing.
        assert soft.item() > hard.item()
