"""Controlled error generation: the nine corruption operators.

Synthetic distortion is the only tractable way to obtain (corrupted, clean)
pairs for Kazakh at evaluation scale — there is no public Kazakh
error-annotated corpus. This module implements nine operators, each gated on
its linguistic preconditions ("each operation was applied only when its
linguistic preconditions were satisfied").

The operators split into three families:

  A. Character-level (1-4): deletion, insertion, substitution, transposition.
     Apply to any alphabetic token of length ≥ 2. These are the classic
     Damerau-edit operations and model keyboard/typing errors.

  B. Script-level (5): mixed-script substitution. Replaces a Kazakh-specific
     grapheme with its visually-similar Russian Cyrillic twin. Requires the
     token to contain at least one Kazakh-only grapheme.

  C. Morphological (6-8): vowel-harmony violation, affix-vowel deletion,
     grammatical-affix substitution. These require the token to carry an
     identifiable suffix (via :class:`kazmorph.rma.ReverseMorphAnalyzer`),
     because they mutate the affix, not the stem. They target the
     morphologically rich error types that character-level operators cannot
     reach.

  Identity (9): the sentence is returned unchanged. A substantial fraction of
  identity examples is required so the false-correction rate has a denominator
  — without clean sentences, FCR is undefined.

Adapted from the IHCONCS_2026_paper project (MIT); only import paths and the
``config`` dependency were changed.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from kazmorph.affixes import AFFIXES
from kazmorph.mfrn import FRONT_VOWELS, BACK_VOWELS
from kazmorph.rma import ReverseMorphAnalyzer
from kazmorph.correction.keyboards import (
    keyboard_neighbours, KAZAKH_VOWELS, mixed_script_substitute, KAZAKH_TO_RUSSIAN,
)

# Default probability that a sentence is corrupted (vs. returned as identity).
# Kept as a module constant for discoverability; ``corrupt_sentence`` accepts a
# ``corrupt_prob`` kwarg so callers (e.g. sweeps) can override it.
CORRUPT_PROB = 0.5


# ── Affix allomorph lookup, grouped by category ──────────────────────────────
# Built once at import. {category: [surface_strings]}. Used by the harmony-
# violation and grammatical-substitution operators to pick a same-category
# sibling (a syntagmatic allomorph) to swap in.
_BY_CATEGORY: dict[str, list[str]] = {}
for _a in AFFIXES:
    _BY_CATEGORY.setdefault(_a["category"], []).append(_a["surface"])


def _harmony_class(vowels_in_form: str) -> str:
    """Return 'front', 'back', or 'mixed' for a sequence of vowels.

    Uses the mfrn FRONT/BACK sets, which is the convention the analyzer's own
    harmony check uses — so corruptions generated against this classification
    will be scored consistently downstream. 'mixed' covers stems whose vowels
    span both classes (these follow last-vowel harmony in real Kazakh).
    """
    has_front = any(v in FRONT_VOWELS for v in vowels_in_form)
    has_back = any(v in BACK_VOWELS for v in vowels_in_form)
    if has_front and has_back:
        return "mixed"
    if has_front:
        return "front"
    if has_back:
        return "back"
    return "none"  # no vowels (rare; consonant-only borrowings)


def _affix_harmony_class(affix: str) -> str:
    """Harmony class of an affix surface, based on its vowels alone."""
    return _harmony_class([c for c in affix if c in FRONT_VOWELS or c in BACK_VOWELS])


@dataclass
class Corruption:
    """One applied corruption."""
    category: str               # one of the 9 operator names below
    position: int               # token index within the sentence
    original: str               # normative surface form
    corrupted: str              # distorted surface form
    detail: str = ""            # operator-specific note (e.g. which allomorph)


# ── Operator names ───────────────────────────────────────────────────────────
OP_DELETION = "character_deletion"
OP_INSERTION = "character_insertion"
OP_SUBSTITUTION = "character_substitution"
OP_TRANSPOSITION = "adjacent_transposition"
OP_MIXED_SCRIPT = "mixed_script_substitution"
OP_HARMONY = "vowel_harmony_violation"
OP_AFFIX_VOWEL_DEL = "affix_vowel_deletion"
OP_AFFIX_SUBST = "grammatical_affix_substitution"
OP_IDENTITY = "identity"

# Operators that require a morphological analysis (have an affix to mutate).
_MORPHOLOGICAL_OPS = {OP_HARMONY, OP_AFFIX_VOWEL_DEL, OP_AFFIX_SUBST}


# ── Character-level operators (family A) ─────────────────────────────────────

def _op_deletion(token: str, rng: random.Random) -> str | None:
    """Remove one grapheme. Returns None if the token is too short."""
    if len(token) < 3:  # keep at least 2 chars so it stays a plausible word
        return None
    i = rng.randrange(len(token))
    return token[:i] + token[i + 1:]


def _op_insertion(token: str, rng: random.Random) -> str | None:
    """Insert a keyboard-neighbour of an adjacent char, or a random vowel."""
    if not token:
        return None
    i = rng.randrange(len(token) + 1)  # insert before position i (incl. end)
    # Prefer a neighbour of the char at i-1 (insertion usually slips next to a key)
    if i > 0:
        nbrs = keyboard_neighbours(token[i - 1])
    else:
        nbrs = keyboard_neighbours(token[0]) if token else []
    # sorted(): KAZAKH_VOWELS is a frozenset — its iteration order depends on
    # string hashes (PYTHONHASHSEED), making corruption output vary across
    # processes. A fixed order makes corrupt_fold byte-reproducible per seed.
    pool = nbrs if nbrs else sorted(KAZAKH_VOWELS)
    ch = rng.choice(pool)
    return token[:i] + ch + token[i:]


def _op_substitution(token: str, rng: random.Random) -> str | None:
    """Replace one grapheme with a keyboard-neighbour or a random vowel."""
    if len(token) < 2:
        return None
    i = rng.randrange(len(token))
    nbrs = keyboard_neighbours(token[i])
    pool = nbrs if nbrs else sorted(KAZAKH_VOWELS)  # hash-seed-independent order
    # Avoid no-op substitutions.
    pool = [c for c in pool if c != token[i].lower()]
    if not pool:
        return None
    repl = rng.choice(pool)
    # Preserve case: if the original was uppercase, uppercase the replacement.
    if token[i].isupper():
        repl = repl.upper()
    return token[:i] + repl + token[i + 1:]


def _op_transposition(token: str, rng: random.Random) -> str | None:
    """Swap two neighbouring graphemes."""
    if len(token) < 3:
        return None
    i = rng.randrange(len(token) - 1)
    chars = list(token)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


# ── Script-level operator (family B) ─────────────────────────────────────────

def _op_mixed_script(token: str, rng: random.Random) -> str | None:
    """Replace a Kazakh-specific grapheme with its Russian visual twin."""
    # Find positions where a Kazakh-only grapheme appears.
    positions = [i for i, c in enumerate(token) if c.lower() in KAZAKH_TO_RUSSIAN]
    if not positions:
        return None
    i = rng.choice(positions)
    repl = mixed_script_substitute(token[i])
    if repl is None:
        return None
    return token[:i] + repl + token[i + 1:]


# ── Morphological operators (family C) ───────────────────────────────────────
# These all need an RMA analysis of the normative token to locate its suffix.
# We pass the analyzer in to avoid re-instantiating it per token.

def _last_affix_split(token: str, rma: ReverseMorphAnalyzer) -> tuple[str, str, str] | None:
    """Split token into (stem, last_affix, category) using RMA.

    Returns None if the token has no recoverable suffix. We use the LAST affix
    (rightmost) because it's the one whose harmony/voicing is most locally
    determined by the stem, and the most natural site for a single-edit
    corruption.
    """
    res = rma.analyze(token)
    if not res.suffixes or not res.suffix_categories:
        return None
    last_affix = res.suffixes[-1]
    last_cat = res.suffix_categories[-1]
    if not token.endswith(last_affix):
        # Stem alternation (e.g. қ↔ғ) means the surface affix isn't a clean
        # suffix of the token; skip this token for morphological corruption.
        return None
    stem = token[: len(token) - len(last_affix)]
    if not stem:
        return None
    return stem, last_affix, last_cat


def _op_harmony_violation(token: str, rng: random.Random, rma: ReverseMorphAnalyzer) -> tuple[str, str] | None:
    """Replace the last affix with a same-category allomorph of opposite harmony.

    Returns (corrupted_form, detail) or None if no suitable swap exists.
    """
    split = _last_affix_split(token, rma)
    if split is None:
        return None
    stem, affix, category = split
    siblings = _BY_CATEGORY.get(category, [])
    if len(siblings) < 2:
        return None
    stem_vowels = [c for c in stem if c in FRONT_VOWELS or c in BACK_VOWELS]
    if not stem_vowels:
        return None
    stem_class = _harmony_class(stem_vowels)
    if stem_class == "mixed":
        # Last-vowel harmony: the affix should match the stem's last vowel.
        stem_class = _harmony_class(stem_vowels[-1])
    # Find an allomorph whose harmony class is the opposite of the stem's.
    target = "back" if stem_class == "front" else "front"
    candidates = [
        s for s in siblings
        if s != affix and _affix_harmony_class(s) == target
    ]
    if not candidates:
        return None
    new_affix = rng.choice(candidates)
    return stem + new_affix, f"{affix}→{new_affix} ({category})"


def _op_affix_vowel_deletion(token: str, rng: random.Random, rma: ReverseMorphAnalyzer) -> tuple[str, str] | None:
    """Delete one vowel from the last affix."""
    split = _last_affix_split(token, rma)
    if split is None:
        return None
    stem, affix, category = split
    vowel_positions = [i for i, c in enumerate(affix) if c in KAZAKH_VOWELS]
    if not vowel_positions:
        return None
    j = rng.choice(vowel_positions)
    new_affix = affix[:j] + affix[j + 1:]
    return stem + new_affix, f"{affix}→{new_affix}"


def _op_affix_substitution(token: str, rng: random.Random, rma: ReverseMorphAnalyzer) -> tuple[str, str] | None:
    """Replace the last affix with a sibling from a DIFFERENT inflectional category.

    Unlike the harmony operator (same category, opposite harmony), this crosses
    categories — e.g. swapping a dative for a genitive — to model the
    "grammatical-affix substitution" error type. We restrict to nominal
    inflectional categories (case/possessive/number) to keep the result
    morphotactically legal as a surface form, just functionally wrong.
    """
    split = _last_affix_split(token, rma)
    if split is None:
        return None
    stem, affix, category = split
    # Nominal inflectional categories eligible for cross-category swap.
    nominal_cats = {"PL", "GEN", "DAT", "ACC", "ABL", "LOC", "INST",
                    "POSS.1SG", "POSS.1PL", "POSS.2SG", "POSS.2PL", "POSS.3"}
    # tuple, not set: set iteration order depends on string hashes
    # (PYTHONHASHSEED), which made corrupt_fold output vary across processes.
    nominal_cats = tuple(sorted(nominal_cats))
    if category not in nominal_cats:
        return None
    other_cats = [c for c in nominal_cats if c != category and c in _BY_CATEGORY]
    if not other_cats:
        return None
    new_cat = rng.choice(other_cats)
    new_affix = rng.choice(_BY_CATEGORY[new_cat])
    return stem + new_affix, f"{affix}({category})→{new_affix}({new_cat})"


# ── Dispatcher ───────────────────────────────────────────────────────────────

# Character-level operators and their callables (no RMA needed).
_CHAR_OPS = {
    OP_DELETION: _op_deletion,
    OP_INSERTION: _op_insertion,
    OP_SUBSTITUTION: _op_substitution,
    OP_TRANSPOSITION: _op_transposition,
    OP_MIXED_SCRIPT: _op_mixed_script,
}


def _is_corruptible(form: str, upos: str) -> bool:
    """Whether a token is eligible to be a corruption target.

    We corrupt only alphabetic content words: punctuation, numerals, and
    closed-class function tokens carry no lexical content worth distorting,
    and corrupting them produces uninformative identity-or-noise pairs.
    """
    if upos in ("PUNCT", "NUM", "SYM", "X"):
        return False
    if len(form) < 3:
        return False
    # Must contain at least one Kazakh alphabet letter.
    return any(c.isalpha() for c in form)


def corrupt_token(
    form: str,
    upos: str,
    rng: random.Random,
    rma: ReverseMorphAnalyzer,
    *,
    allow_morphological: bool = True,
) -> Corruption | None:
    """Apply one randomly-chosen operator to `form`.

    Returns a Corruption, or None if no operator's preconditions are met
    (e.g. the token is too short or has no suffix for a morphological op).
    The caller retries with a different token in that case.
    """
    if not _is_corruptible(form, upos):
        return None

    # Build the candidate operator list. Morphological ops are included only
    # when the analyzer is available and the caller allows them.
    ops = list(_CHAR_OPS.keys())
    if allow_morphological:
        ops += [OP_HARMONY, OP_AFFIX_VOWEL_DEL, OP_AFFIX_SUBST]
    rng.shuffle(ops)

    for op in ops:
        if op in _CHAR_OPS:
            corrupted = _CHAR_OPS[op](form, rng)
            if corrupted and corrupted != form:
                return Corruption(op, -1, form, corrupted)
        else:
            # Morphological operators return (corrupted, detail).
            if op == OP_HARMONY:
                res = _op_harmony_violation(form, rng, rma)
            elif op == OP_AFFIX_VOWEL_DEL:
                res = _op_affix_vowel_deletion(form, rng, rma)
            else:
                res = _op_affix_substitution(form, rng, rma)
            if res is not None:
                corrupted, detail = res
                if corrupted and corrupted != form:
                    return Corruption(op, -1, form, corrupted, detail)
    return None


@dataclass
class SentencePair:
    """One (clean, corrupted) sentence pair with full provenance."""
    sent_id: str
    clean_forms: list[str]
    corrupted_forms: list[str]
    is_identity: bool
    corruption: Corruption | None = None  # None iff is_identity


def corrupt_sentence(
    sent,
    rng: random.Random,
    rma: ReverseMorphAnalyzer,
    *,
    corrupt_prob: float = CORRUPT_PROB,
    allow_morphological: bool = True,
) -> SentencePair:
    """Produce a (clean, corrupted) pair for one gold sentence.

    With probability `corrupt_prob`, exactly one eligible token is corrupted;
    otherwise the sentence is returned unchanged (identity example). At most
    one token per sentence is distorted — this keeps the evaluation focused on
    single-token correction and avoids the confound of multi-error
    interactions that the current single-token ranker cannot model.

    ``sent`` is a :class:`kazmorph.corpus.Sentence` (any object with
    ``.sent_id`` and a ``.tokens`` sequence of objects exposing ``.form`` and
    ``.upos``).
    """
    clean = [t.form for t in sent.tokens]
    if rng.random() >= corrupt_prob:
        return SentencePair(sent_id=getattr(sent, "sent_id", ""), clean_forms=clean,
                            corrupted_forms=list(clean), is_identity=True)

    # Pick a corruption target: try eligible tokens in random order until one
    # yields a non-None corruption. This is rejection sampling over tokens.
    eligible_indices = [
        i for i, t in enumerate(sent.tokens)
        if _is_corruptible(t.form, t.upos)
    ]
    rng.shuffle(eligible_indices)
    for idx in eligible_indices:
        tok = sent.tokens[idx]
        c = corrupt_token(
            tok.form, tok.upos, rng, rma,
            allow_morphological=allow_morphological,
        )
        if c is not None:
            c.position = idx
            corrupted = list(clean)
            corrupted[idx] = c.corrupted
            return SentencePair(
                sent_id=getattr(sent, "sent_id", ""),
                clean_forms=clean, corrupted_forms=corrupted,
                is_identity=False, corruption=c,
            )
    # No eligible token could be corrupted → fall back to identity.
    return SentencePair(sent_id=getattr(sent, "sent_id", ""), clean_forms=clean,
                        corrupted_forms=list(clean), is_identity=True)


def corrupt_fold(
    sentences,
    seed: int,
    *,
    corrupt_prob: float = CORRUPT_PROB,
    allow_morphological: bool = True,
) -> list[SentencePair]:
    """Corrupt every sentence in a fold (after splitting).

    One RMA instance is shared across the fold (it is read-only after init).
    Each fold gets its own RNG seeded from `seed` so corruption is reproducible
    and independent across folds — a normative sentence corrupted in fold k's
    test set is never seen corrupted in any other fold's training data.
    """
    rng = random.Random(seed)
    rma = ReverseMorphAnalyzer()
    return [
        corrupt_sentence(s, rng, rma, corrupt_prob=corrupt_prob,
                         allow_morphological=allow_morphological)
        for s in sentences
    ]


def summarise_pairs(pairs: list[SentencePair]) -> dict:
    """Category counts + identity fraction, for reporting."""
    cat_counts: Counter = Counter()
    n_identity = 0
    for p in pairs:
        if p.is_identity:
            n_identity += 1
        elif p.corruption is not None:
            cat_counts[p.corruption.category] += 1
    n_total = len(pairs)
    return {
        "n_sentences": n_total,
        "n_identity": n_identity,
        "identity_fraction": n_identity / n_total if n_total else 0.0,
        "n_corrupted": n_total - n_identity,
        "category_counts": dict(cat_counts.most_common()),
    }
