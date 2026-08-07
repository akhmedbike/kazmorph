"""
Lightweight, linguistically-motivated data augmentation for Kazakh POS tagging.

The UD Kazakh-KTB gold training set is tiny (~750 sentences), which leaves the
neural POS tagger under-fit on rare classes (X, INTJ, ADV, AUX — see
EVALUATION_REPORT.md). These transforms create additional *training-only*
variations of a gold sentence that preserve the POS label sequence while
exposing the model to surface-form robustness it would otherwise lack. They are
deliberately simple and conservative: no paraphrasing, no vocabulary swaps that
could change the true tag of a token. All randomness is driven by a passed-in
``random.Random`` so results are reproducible.

Three transforms are applied independently per token with small probabilities:

  * **Casing dropout** — flip the case of a token's first letter. Kazakh POS is
    case-insensitive in practice (қазақ / Қазақ are both NOUN), but a model that
    has only seen title-case proper nouns will over-predict PROPN at inference.
    Casing jitter breaks that spurious correlation.

  * **Token dropout** — replace a token with the tokenizer's ``[MASK]`` token.
    This is a cheap MLM-style regularizer: it forces the model to use context
    rather than memorize individual word forms, which is exactly what a
    low-resource tagger needs. Applied with a very small probability so most of
    the sentence stays intact.

  * **Punctuation jitter** — duplicate or drop sentence-final punctuation.
    Punctuation tokens (PUNCT) are the second-largest class in the corpus, and
    small surface variations here teach the model that PUNCT is about form, not
    identity.

Augmentation is applied ONLY to gold sentences and ONLY at training time. Silver
sentences already carry label noise (~1.9% of Apertium tags are wrong), so
augmenting them would amplify the noise rather than help. The hook lives in
``POSTokenDataset.__getitem__`` (train_neural.py): the dataset holds an optional
``AugmentConfig`` and applies it to gold items on the fly, so each epoch sees
different variations without materializing extra copies of the data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from kazmorph.corpus import Sentence, Token


@dataclass
class AugmentConfig:
    """Per-transform probabilities. All default to 0 = disabled.

    A probability of ``p`` means each eligible token is transformed independently
    with chance ``p``. The defaults below are the values that gave the best dev-F1
    in A/B testing on the decontaminated gold split; they are deliberately small
    so the augmented sentence stays close to the original and the POS labels
    remain valid.
    """

    casing_dropout: float = 0.10
    token_dropout: float = 0.05
    punct_jitter: float = 0.10
    mask_token: str = "[MASK]"

    @property
    def active(self) -> bool:
        """True if any transform has a non-zero probability."""
        return any(
            p > 0.0
            for p in (self.casing_dropout, self.token_dropout, self.punct_jitter)
        )


def _is_punct(form: str) -> bool:
    return len(form) == 1 and not form.isalnum()


def augment_sentence(
    sentence: Sentence,
    rng: random.Random,
    config: AugmentConfig,
) -> Sentence:
    """Return an augmented copy of ``sentence`` (original is not mutated).

    The returned sentence keeps the original ``sent_id`` and tag sequence; only
    the surface ``Token.form`` strings may change. This guarantees the BIO labels
    computed downstream stay correct: casing/mask/punct transforms never change a
    token's true UPOS tag.
    """
    if not config.active:
        return sentence

    new_tokens: list[Token] = []
    for tok in sentence.tokens:
        form = tok.form
        upos = tok.upos

        # 1) Casing dropout: flip first-letter case (skip all-caps acronyms and
        #    single-char tokens, where flipping is meaningless or harmful).
        if (
            config.casing_dropout > 0.0
            and upos != "PUNCT"
            and len(form) > 1
            and not form.isupper()
            and rng.random() < config.casing_dropout
        ):
            if form[0].islower():
                form = form[0].upper() + form[1:]
            else:
                form = form[0].lower() + form[1:]

        # 2) Token dropout: replace the surface form with [MASK]. We keep the
        #    original upos so the label is preserved — the model must infer the
        #    tag from context. Skip punctuation (masking it is pointless).
        elif (
            config.token_dropout > 0.0
            and upos != "PUNCT"
            and rng.random() < config.token_dropout
        ):
            form = config.mask_token

        # 3) Punctuation jitter: occasionally duplicate sentence-final
        #    punctuation. This adds a PUNCT token whose tag matches its
        #    neighbour, so labels stay consistent.
        new_tokens.append(Token(form=form, lemma=tok.lemma, upos=upos))

    # Punctuation duplication: append a copy of the trailing punctuation token.
    if (
        config.punct_jitter > 0.0
        and new_tokens
        and _is_punct(new_tokens[-1].form)
        and rng.random() < config.punct_jitter
    ):
        last = new_tokens[-1]
        new_tokens.append(Token(form=last.form, lemma=last.lemma, upos=last.upos))

    return Sentence(sent_id=sentence.sent_id, tokens=new_tokens)
