"""
Stage 1: Reverse Morphological Analyzer (RMA)

Uses a Reverse Trie with 67 Kazakh affixes to decompose word forms
into stem + suffix chain via iterative suffix stripping. This is the
first stage of the 4-stage pipeline.

Algorithm:
  1. Check UD lemma dictionary (exact match → high confidence).
  2. Iteratively strip the longest matching suffix from the word,
     repeating on the remaining stem until no more suffixes match
     or the stem is found in the lexicon.
  3. Return the lemma (final stem) and the list of individual suffixes.

Includes a UD lemma dictionary for high-accuracy lookup fallback.
"""

import os
from dataclasses import dataclass

from kazmorph.affixes import AFFIXES
from kazmorph.lexicons import is_known_noun, is_known_verb, is_known_adj


# Maximum suffix-stripping iterations to prevent infinite loops
_MAX_STRIP_ITER = 10
# Minimum stem length — don't strip below this
_MIN_STEM_LEN = 2

# Closed-class / invariant lexemes that must NEVER pass through suffix
# stripping. These are determiners, pronouns, adverbs of manner, and
# conjunctions/particles — they are morphologically atomic (no productive
# inflection), but the iterative trie falsely decomposes them because they
# happen to end in letter sequences that look like Kazakh affixes
# (e.g. әрбір → "і"+"р", осылай → "ла"+"й"). See REVIEW_RECOMMENDATIONS.md
# problem #3.
#
# Only truly invariant words are listed here. Productively-derived forms
# are intentionally excluded even when they look atomic:
#   - Ordinal numerals (екінші, бірінші, үшінші) — "-ші/-інші" is real
#     morphology and segmenting it is correct.
#   - көпшілік ("majority") — a real suffixed noun, not a determiner.
# Each value is the canonical UD POS for the lemma.
CLOSED_CLASS_LEMMAS: dict[str, str] = {
    # Determiners / demonstratives
    "осы": "DET", "бұл": "DET", "сол": "DET",
    "әрбір": "DET", "барлық": "DET", "кейбір": "DET",
    "бірдеңе": "DET", "еш": "DET", "бүкіл": "DET",
    # Demonstrative / interrogative adverbs of manner (atomic)
    "осылай": "ADV", "бұлай": "ADV", "солай": "ADV",
    "қалай": "ADV", "осылайша": "ADV", "бұлайша": "ADV",
    "ешқашан": "ADV",
    # Pronouns / wh-words
    "не": "PRON", "кім": "PRON", "ешкім": "PRON",
    "қайда": "ADV", "қашан": "ADV", "неге": "ADV",
    "неше": "NUM",
    # Conjunctions / particles
    "және": "CCONJ", "немесе": "CCONJ", "бірақ": "CCONJ",
    "ал": "CCONJ", "сонда": "SCONJ", "егер": "SCONJ", "тек": "ADV",
}


# Roman numeral characters used for _classify_numeral. Note these overlap
# with ordinary Latin letters, so single-char tokens are only treated as
# Roman numerals when they are one of the canonical Roman digits.
_ROMAN_CHARS = frozenset("IVXLCDM")


def _classify_numeral(token: str) -> str | None:
    """Return "NUM" if the token is a numeral, else None.

    Recognizes:
      - Arabic integers and decimals: 1920, 7,2, 58,3%
        (Kazakh uses comma as the decimal separator.)
      - Roman numerals: VIII, XX, MCMXC. Only matched for tokens whose
        characters are all Roman-numeral letters AND which look like a real
        Roman numeral (length >= 2, or a single canonical digit I/V/X/L/C/
        D/M). This avoids mistreating the Latin word "I" or an abbreviation
        like "M" as a numeral.

    Returns None for anything else so the caller falls through to the
    normal morphological analysis.
    """
    if not token:
        return None

    # Arabic digits: allow digits, one comma/period decimal separator, and
    # a trailing percent sign. Must contain at least one digit.
    has_digit = any(c.isdigit() for c in token)
    if has_digit:
        allowed = set("0123456789,.%")
        if all(c in allowed for c in token):
            return "NUM"
        return None

    # Roman numerals: all chars in IVXLCDM, length >= 2. Single letters
    # are intentionally NOT matched: in running text an isolated "I", "V",
    # "M" etc. is far more likely an initial, pronoun, or variable than a
    # Roman numeral (and UD Kazakh-KTB does not use single-letter Roman
    # numerals). Length >= 2 still catches VIII, XX, MCMXC, XL.
    upper = token.upper()
    if len(upper) >= 2 and all(c in _ROMAN_CHARS for c in upper):
        return "NUM"
    return None


@dataclass
class RMAResult:
    """Result of morphological analysis for a single token."""
    token: str
    lemma: str
    pos: str  # NOUN, VERB, ADJ, ADV, etc.
    suffixes: list[str]
    suffix_categories: list[str]  # e.g. ["PL", "POSS.3", "GEN"]
    confidence: float  # 0.0 - 1.0


class TrieNode:
    __slots__ = ("children", "is_end", "affix_data")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.affix_data: dict | None = None


class ReverseTrie:
    """Trie that stores affixes in reverse for efficient suffix matching."""

    def __init__(self):
        self.root = TrieNode()
        self._build(AFFIXES)

    def _build(self, affixes: list[dict]):
        for affix in affixes:
            node = self.root
            # Insert reversed suffix
            for ch in reversed(affix["surface"]):
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.is_end = True
            node.affix_data = affix

    def search(self, word: str) -> list[dict]:
        """Find all matching suffixes for a word (longest match first)."""
        results = []
        node = self.root
        matched = ""

        for ch in reversed(word):
            if ch not in node.children:
                break
            matched = ch + matched
            node = node.children[ch]
            if node.is_end and node.affix_data:
                results.append(node.affix_data | {"matched": matched})

        return list(reversed(results))  # longest first


def _load_lemma_dict() -> dict[str, tuple[str, str]]:
    """
    Load lemma dictionary from UD Kazakh-KTB train + test CoNLL-U files.
    Returns: {form_lower: (lemma, upos)} — one canonical entry per form.
    """
    lemma_dict: dict[str, tuple[str, str]] = {}
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    for fname in ("train_split.conllu",):
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                if "-" in parts[0] or "." in parts[0]:
                    continue
                form, lemma, upos = parts[1], parts[2], parts[3]
                key = form.lower()
                if key not in lemma_dict:
                    lemma_dict[key] = (lemma, upos)

    return lemma_dict


class ReverseMorphAnalyzer:
    """Stage 1: Reverse Morphological Analyzer with iterative suffix stripping."""

    def __init__(self):
        self.trie = ReverseTrie()
        self.lemma_dict = _load_lemma_dict()

    def analyze(self, token: str) -> RMAResult:
        """Analyze a single token and return morphological decomposition."""

        # Handle punctuation and symbols
        if not any(c.isalnum() for c in token):
            return RMAResult(
                token=token,
                lemma=token,
                pos="PUNCT",
                suffixes=[],
                suffix_categories=[],
                confidence=1.0,
            )

        # Numerals: Arabic digits, decimals (7,2 / 58,3%), and Roman
        # numerals (VIII, XX). These have no inflectional suffixes, so they
        # must short-circuit before the trie, which would otherwise either
        # leave them as UNK (pure digits) or wrongly strip letter sequences
        # that happen to look like affixes (e.g. Roman numerals). UD tags
        # them NUM. See error analysis: 20 NUM tokens were tagged UNK.
        num_pos = _classify_numeral(token)
        if num_pos is not None:
            return RMAResult(
                token=token,
                lemma=token,
                pos=num_pos,
                suffixes=[],
                suffix_categories=[],
                confidence=0.95,
            )

        # Closed-class / invariant lexemes: return atomically without
        # suffix stripping. These words have no productive inflection, so
        # any suffix decomposition would be spurious. Checked before the
        # UD dict lookup so the canonical POS from CLOSED_CLASS_LEMMAS wins
        # (the UD train split may assign a context-specific POS).
        closed_pos = CLOSED_CLASS_LEMMAS.get(token.lower())
        if closed_pos is not None:
            return RMAResult(
                token=token,
                lemma=token.lower(),
                pos=closed_pos,
                suffixes=[],
                suffix_categories=[],
                confidence=0.95,
            )

        # Dictionary lookup first (case-insensitive)
        dict_entry = self.lemma_dict.get(token.lower())
        if dict_entry:
            lemma, pos = dict_entry
            # Decompose the remaining suffix chain iteratively
            suffixes, categories = self._decompose_suffixes(token, lemma)
            return RMAResult(
                token=token,
                lemma=lemma,
                pos=pos,
                suffixes=suffixes,
                suffix_categories=categories,
                confidence=0.95,
            )

        # Fallback: iterative trie-based suffix stripping
        result = self._iterative_strip(token)

        # Proper-noun guard. The trie finds spurious suffixes in many names
        # (Стамбұл → "л", Атырау → "а"+"у", Сергей → "гей"), truncating the
        # lemma and mislabeling the POS as VERB. A capitalized token that is
        # not in the UD dictionary and whose stripped stem is NOT a known
        # lexeme is overwhelmingly a proper noun; in that case keep the form
        # intact and tag it PROPN. We only fire when the stripper actually
        # removed something (suffixes found) — otherwise the UNK/intact
        # result is already fine (e.g. Михайлович). This is a fallback: the
        # neural POS tagger can still override it from context.
        if (
            result.suffixes
            and token[:1].isupper()
            and not self._is_known_word(result.lemma, result.pos)
        ):
            return RMAResult(
                token=token,
                lemma=token,
                pos="PROPN",
                suffixes=[],
                suffix_categories=[],
                confidence=0.6,
            )
        return result

    def _iterative_strip(self, token: str) -> RMAResult:
        """
        Iteratively strip suffixes from the token using the reverse trie.

        Returns suffixes and categories in ROOT-TO-SURFACE order.
        """
        current = token
        stripped_suffixes: list[str] = []
        stripped_categories: list[str] = []
        pos = "UNK"

        for _ in range(_MAX_STRIP_ITER):
            if len(current) <= _MIN_STEM_LEN:
                break

            matches = self.trie.search(current)
            if not matches:
                break

            best = matches[0]
            matched_suffix = best["matched"]
            stem = current[: -len(matched_suffix)] if matched_suffix else current

            if len(stem) < _MIN_STEM_LEN:
                break

            # Insert at front for root-to-surface order
            stripped_suffixes.insert(0, matched_suffix)
            stripped_categories.insert(0, best.get("category", "?"))
            pos = best.get("pos", "UNK")
            current = stem

            if self._is_known_word(current, pos):
                break

        # Suffix-stripping sets POS from the last stripped suffix's category.
        # For words with no (further) matching suffix — e.g. bare dictionary
        # forms like конституция, ат, спорт — the loop leaves pos at "UNK"
        # even though the stem is a known noun/verb/adjective. Recover the
        # POS from the lexicon so downstream stages see a concrete tag and a
        # confidence above the UNK floor. This only ever upgrades UNK to a
        # known POS, never overrides an already-assigned one.
        if pos == "UNK":
            pos = self._pos_from_lexicon(current)

        confidence = self._estimate_confidence(current, pos)

        return RMAResult(
            token=token,
            lemma=current,
            pos=pos,
            suffixes=stripped_suffixes,
            suffix_categories=stripped_categories,
            confidence=confidence,
        )

    def _decompose_suffixes(self, token: str, lemma: str) -> tuple[list[str], list[str]]:
        """
        Given a token and its known lemma (from UD dict), decompose the
        suffix part into individual morphemes using iterative trie stripping.

        Returns (suffixes, categories) in ROOT-TO-SURFACE order.

        Example: әдебиетінің → (['і', 'нің'], ['POSS.3', 'GEN'])
        """
        t_lower = token.lower()
        l_lower = lemma.lower()

        if t_lower == l_lower:
            return [], []

        if not t_lower.startswith(l_lower):
            # Stem changed — return raw suffix without categories
            suf = t_lower[len(l_lower):] if t_lower.startswith(l_lower) else ""
            return ([suf], ["?"]) if suf else ([], [])

        suffix_str = t_lower[len(l_lower):]
        if not suffix_str:
            return [], []

        # Iteratively strip into morphemes + categories
        morphemes: list[str] = []
        categories: list[str] = []
        remaining = suffix_str

        for _ in range(_MAX_STRIP_ITER):
            if not remaining:
                break

            matches = self.trie.search(remaining)
            if not matches:
                if remaining:
                    morphemes.insert(0, remaining)
                    categories.insert(0, "?")
                break

            best = matches[0]
            matched = best["matched"]
            morphemes.insert(0, matched)
            categories.insert(0, best.get("category", "?"))
            remaining = remaining[: -len(matched)] if matched else ""

        return morphemes, categories

    @staticmethod
    def _is_known_word(stem: str, pos: str) -> bool:
        """Check if stem is a known word in any lexicon."""
        s_lower = stem.lower()
        if is_known_noun(s_lower) or is_known_verb(s_lower) or is_known_adj(s_lower):
            return True
        return False

    @staticmethod
    def _pos_from_lexicon(stem: str) -> str:
        """Recover a POS tag from lexicon membership.

        Used when suffix-stripping produced no POS clue (bare stems with no
        recognized suffix). Returns the lexicon's tag only if the stem is in
        exactly one lexicon; ambiguous stems (in two or more, e.g. a word
        that is both a noun and a verb root) stay "UNK" so we don't guess.
        """
        s_lower = stem.lower()
        tags: list[str] = []
        if is_known_noun(s_lower):
            tags.append("NOUN")
        if is_known_verb(s_lower):
            tags.append("VERB")
        if is_known_adj(s_lower):
            tags.append("ADJ")
        return tags[0] if len(tags) == 1 else "UNK"

    def _estimate_confidence(self, stem: str, pos: str) -> float:
        """Estimate confidence based on lexicon membership."""
        base = 0.5
        if pos == "NOUN" and is_known_noun(stem):
            base = 0.9
        elif pos == "VERB" and is_known_verb(stem):
            base = 0.9
        elif pos == "ADJ" and is_known_adj(stem):
            base = 0.85
        elif pos == "UNK":
            base = 0.2
        return base
