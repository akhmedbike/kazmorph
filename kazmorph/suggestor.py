"""
Correction suggestion engine.

For tokens where the analyzer has low confidence or cannot identify the word,
generates correction suggestions by finding similar known words using
Levenshtein distance.

Performance: the lexicon has ~22k entries. A naive per-word Levenshtein scan
is O(n * m) — ~85s for 1200 unique words (the "20-second wait on long text"
failure mode). BK-tree helps only ~2x here because Kazakh words are short,
leaving little room for the triangle-inequality pruning to cut branches.

This module uses the SymSpell algorithm (Wolf Garbe, 2012): precompute every
deletion variant (up to the max edit distance) of every dictionary word into a
hash map, then a query generates the same deletion variants of the input and
does hash-map lookups. Cost per query is O(L^2) string ops + a handful of
hashmap probes, independent of dictionary size — typically 50-500x faster than
brute force, with byte-identical results. Memory cost is ~1-3 MB for 22k words
at max_distance=2.
"""

from kazmorph.lexicons import KNOWN_NOUNS, KNOWN_VERBS, KNOWN_ADJ, BORROWINGS
from kazmorph.rma import RMAResult


# Build vocabulary index (lazy singleton)
_VOCAB: set[str] | None = None


def _get_vocab() -> set[str]:
    """Get combined vocabulary from all lexicons."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = KNOWN_NOUNS | KNOWN_VERBS | KNOWN_ADJ | BORROWINGS
    return _VOCAB


def _levenshtein_le(a: str, b: str, max_dist: int) -> int:
    """
    Levenshtein distance with early exit: returns as soon as the running
    minimum exceeds max_dist, returning max_dist+1. This prunes most candidate
    pairs during the final verification step (the vast majority of candidates
    retrieved by a SymSpell deletion-bucket are NOT within distance).
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Ensure b is the shorter — smaller inner loop.
    if lb > la:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        row_min = cur[0]
        for j in range(1, lb + 1):
            cur[j] = min(
                prev[j] + 1,           # delete
                cur[j - 1] + 1,        # insert
                prev[j - 1] + (ca != b[j - 1]),  # substitute
            )
            if cur[j] < row_min:
                row_min = cur[j]
        # Early exit: if every entry in this row already exceeds max_dist,
        # the final distance cannot be <= max_dist.
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


def _edits_delete(word: str, max_dist: int) -> set[str]:
    """
    Generate all strings reachable from `word` by deleting 1..max_dist chars.
    E.g. _edits_delete("адам", 2) includes "дам", "аам", "ада", "ам", "аа", ...
    Duplicates collapse into the set automatically (e.g. deleting the first or
    second 'а' of "аам" both yield "ам"). Order doesn't matter — these are
    purely bucket keys.
    """
    deletes = {word}
    for _ in range(max_dist):
        new_deletes: set[str] = set()
        for w in deletes:
            if len(w) > 1:
                for i in range(len(w)):
                    new_deletes.add(w[:i] + w[i + 1:])
            elif len(w) == 1:
                # Final deletion yields the empty string. Keeping "" as a key
                # is required for correctness on the degenerate case of two
                # equal-length short words at distance == max_dist (e.g. query
                # "іс" vs dictionary "аз": both delete down to "" at k=2, and
                # that shared empty key is the only one connecting them).
                new_deletes.add("")
        deletes |= new_deletes
    # Drop the original word itself — it is indexed separately by the caller
    # (as its own bucket key) to keep the distance=0 path cheap.
    deletes.discard(word)
    return deletes


class _SymIndex:
    """
    SymSpell index: maps each deletion-variant to the set of dictionary words
    that produce it. A query word within edit distance k of some dictionary
    word shares at least one deletion-variant with it (symmetry of deletes),
    so the candidate set for a query is the union of buckets reachable from
    the query's own deletions — typically a few dozen words, not 22k.
    """

    __slots__ = ("deletes", "max_dist")

    def __init__(self, vocab: set[str], max_dist: int):
        self.max_dist = max_dist
        self.deletes: dict[str, set[str]] = {}
        for w in vocab:
            # Index EVERY deletion variant reachable from w, INCLUDING w
            # itself. The reason: a query q at edit distance k from w shares
            # at least one deletion-variant with w, but that shared variant
            # may equal w itself (e.g. q="халыққа", w="халық": deleting "қа"
            # from q yields "халық" == w). If w is not in the deletes map,
            # such pairs are missed. So we treat each word as its own
            # distance-0 deletion bucket key.
            bucket = self.deletes.get(w)
            if bucket is None:
                self.deletes[w] = {w}
            else:
                bucket.add(w)
            for d in _edits_delete(w, max_dist):
                bucket = self.deletes.get(d)
                if bucket is None:
                    self.deletes[d] = {w}
                else:
                    bucket.add(w)

    def query(self, word: str) -> list[tuple[int, str]]:
        """Return [(distance, word), ...] for dictionary entries within max_dist."""
        results: list[tuple[int, str]] = []

        # Collect every deletion key reachable from `word`, including the word
        # itself. Symmetry with the index build step: if some dictionary word
        # w has a shared deletion-variant equal to `word` itself, we need to
        # probe the bucket under key=word too.
        query_keys = _edits_delete(word, self.max_dist)
        query_keys.add(word)

        seen: set[str] = set()
        for d in query_keys:
            bucket = self.deletes.get(d)
            if not bucket:
                continue
            for cand in bucket:
                if cand in seen:
                    continue
                dist = _levenshtein_le(word, cand, self.max_dist)
                if dist <= self.max_dist:
                    results.append((dist, cand))
                    seen.add(cand)
        return results


_SYMINDEX: _SymIndex | None = None


def _get_symindex() -> _SymIndex:
    """Build the SymSpell index once (lazy). ~22k words in well under a second."""
    global _SYMINDEX
    if _SYMINDEX is None:
        _SYMINDEX = _SymIndex(_get_vocab(), max_dist=2)
    return _SYMINDEX


def suggest(token: str, rma_result: RMAResult, max_suggestions: int = 3) -> list[str]:
    """
    Generate correction suggestions for a token.

    Suggests when the token looks like a likely typo/unknown word:
    - POS is UNK (unknown word), OR
    - Confidence is below 0.85 (the calibrated softmax zone where real accuracy
      drops below ~70%), OR
    - Token is not in the vocabulary (unknown form, possibly misspelled)
    - Never for punctuation or very short tokens.

    Returns list of suggested words sorted by similarity (best first).
    """
    # Don't suggest for punctuation, numbers, or very short tokens
    if rma_result.pos == "PUNCT":
        return []
    if len(token) < 2:
        return []

    t_lower = token.lower()

    # If the word is already known AND high-confidence, no suggestion needed.
    # An unknown-to-lexicon form still gets suggestions even at moderate conf.
    if t_lower in _get_vocab() and rma_result.confidence >= 0.85:
        return []
    # If the word is in the lexicon at all, no suggestion needed (it's a known
    # form, just possibly low-confidence — that's a tagging issue, not a typo).
    if t_lower in _get_vocab():
        return []

    index = _get_symindex()
    candidates = index.query(t_lower)
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [word for _, word in candidates[:max_suggestions]]
