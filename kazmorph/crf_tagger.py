"""
Stage 2: CRF Tagger (stub)

Currently uses deterministic fallback — will be replaced with trained CRF model
using sklearn-crfsuite (Step 2 of roadmap).

The CRF uses a context window [t-2, t+2] for POS disambiguation.
For example: distinguishing -ды/-ді as Accusative (NOUN) vs Past 3Sg (VERB).
"""

from __future__ import annotations

import os

from kazmorph.rma import RMAResult


class CRFTagger:
    """Stage 2: Conditional Random Fields POS tagger."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load a trained CRF model from joblib file."""
        import joblib
        self.model = joblib.load(path)

    def tag(self, rma_results: list[RMAResult]) -> list[RMAResult]:
        """
        Adjust POS tags and confidence using CRF or fallback.
        Input/Output: list of RMAResult from Stage 1.
        """
        if self.model is not None:
            return self._model_predict(rma_results)
        return self._fallback_adjust(rma_results)

    def _model_predict(self, rma_results: list[RMAResult]) -> list[RMAResult]:
        """Predict using trained CRF model."""
        features = [self.extract_features(rma_results, i) for i in range(len(rma_results))]
        predictions = self.model.predict([features])[0]

        adjusted = []
        for r, pred_pos in zip(rma_results, predictions):
            adjusted.append(RMAResult(
                token=r.token,
                lemma=r.lemma,
                pos=pred_pos,
                suffixes=r.suffixes,
                suffix_categories=r.suffix_categories,
                confidence=min(r.confidence + 0.1, 1.0),
            ))
        return adjusted

    def _fallback_adjust(self, rma_results: list[RMAResult]) -> list[RMAResult]:
        """
        Deterministic fallback when no CRF model is loaded.
        Applies a confidence penalty based on POS category.
        """
        penalties = {
            "NOUN": 0.05,
            "VERB": 0.10,
            "ADJ": 0.08,
            "ADV": 0.12,
            "UNK": 0.25,
        }
        adjusted = []
        for r in rma_results:
            penalty = penalties.get(r.pos, 0.15)
            adjusted.append(RMAResult(
                token=r.token,
                lemma=r.lemma,
                pos=r.pos,
                suffixes=r.suffixes,
                suffix_categories=r.suffix_categories,
                confidence=round(max(r.confidence - penalty, 0.0), 3),
            ))
        return adjusted

    @staticmethod
    def extract_features(tokens: list[RMAResult], idx: int) -> dict:
        """
        Extract CRF features for token at position idx.
        ~30 features including RMA POS, confidence, lexicon membership,
        character suffixes, word shape, and context window [t-2, t+2].
        """
        t = tokens[idx]
        word = t.token
        word_lower = word.lower()

        # Lexicon membership (lazy import to avoid circular)
        from kazmorph.lexicons import (
            KNOWN_NOUNS, KNOWN_VERBS, KNOWN_ADJ, BORROWINGS, is_borrowing,
        )

        features = {
            # Core features
            "bias": 1.0,
            "rma_pos": t.pos,
            "rma_confidence": t.confidence,
            "suffix_count": len(t.suffixes),
            "has_suffix": bool(t.suffixes),

            # Token shape
            "token_length": len(word),
            "is_upper": word[0].isupper() if word else False,
            "is_all_upper": word.isupper() if word else False,
            "has_digit": any(c.isdigit() for c in word),
            "is_punct": not any(c.isalnum() for c in word) if word else False,
            "word_shape": _word_shape(word),

            # Character suffixes (very discriminative for Kazakh POS)
            "suf_1": word[-1:] if word else "",
            "suf_2": word[-2:] if len(word) >= 2 else "",
            "suf_3": word[-3:] if len(word) >= 3 else "",
            "suf_4": word[-4:] if len(word) >= 4 else "",

            # Character prefixes
            "pref_1": word[:1] if word else "",
            "pref_2": word[:2] if len(word) >= 2 else "",

            # Lexicon membership
            "in_nouns": word_lower in KNOWN_NOUNS,
            "in_verbs": word_lower in KNOWN_VERBS,
            "in_adj": word_lower in KNOWN_ADJ,
            "is_borrowing": is_borrowing(word),

            # Lemma features
            "lemma": t.lemma,
            "lemma_suf_2": t.lemma[-2:] if len(t.lemma) >= 2 else "",
        }

        # Context window [-2, +2]
        for offset in (-2, -1, 1, 2):
            pos = idx + offset
            prefix = f"{'prev' if offset < 0 else 'next'}_{abs(offset)}_"
            if 0 <= pos < len(tokens):
                ctx = tokens[pos]
                features[f"{prefix}pos"] = ctx.pos
                features[f"{prefix}has_suffix"] = bool(ctx.suffixes)
                features[f"{prefix}is_upper"] = ctx.token[0].isupper() if ctx.token else False
            else:
                features[f"{prefix}pos"] = "BOS" if offset < 0 else "EOS"

        # Bigram POS features (transition hints)
        if idx > 0:
            features["bigram_pos"] = f"{tokens[idx-1].pos}_{t.pos}"
        if idx < len(tokens) - 1:
            features["next_pos_pair"] = f"{t.pos}_{tokens[idx+1].pos}"

        return features


def _word_shape(word: str) -> str:
    """Convert word to shape: 'Xxxx' → 'Xxxx', '123' → 'ddd', 'abc' → 'xxx'."""
    if not word:
        return ""
    shape = []
    for c in word[:6]:  # Limit to first 6 chars
        if c.isupper():
            shape.append("X")
        elif c.islower():
            shape.append("x")
        elif c.isdigit():
            shape.append("d")
        else:
            shape.append(c)
    return "".join(shape)
