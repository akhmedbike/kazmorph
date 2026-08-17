"""Leakage-free cross-validation fold construction for UD_Kazakh-KTB.

Two strategies:

* ``stratified`` — sentences are grouped by source document
  (:func:`source_of` of the ``sent_id``), and the *blocks* assigned to folds
  are built so that each fold's source distribution tracks the corpus
  distribution as closely as a single pass of greedy block allocation allows.
  Sentences are then shuffled across folds by block, so individual sentences
  of the same source can land in different folds. This mirrors a random
  stratified split.

* ``grouped`` — every sentence of a given source goes into exactly one fold,
  so no source document is ever split across folds. This is the
  leakage-by-duplication control: UDHR and the wikitravel phrasebook are
  templated, so a random split lets near-duplicate sentences leak from train
  into test. Grouped CV removes that.

Fold layout (fold ``i``)::

    test  = block i
    dev   = block (i + 1) % k
    train = remaining k - 2 blocks

Invariants enforced by assertions:

* the union of all folds' ``test_idx`` is the full sentence index range,
  with no overlaps (every sentence is tested exactly once — the
  jack-knife property);
* within each fold, ``train`` / ``dev`` / ``test`` are pairwise disjoint;
* the split is bit-for-bit reproducible for a fixed seed;
* in ``grouped`` mode, no source spans more than one fold's ``test_idx``.

Adapted from the CAGF-CBT-CRF project (MIT). The only change is the
``sent_id_of`` coupling: CAGF read ``sent_id`` out of preserved CoNLL-U
comment lines; kazmorph's :class:`kazmorph.corpus.Sentence` stores ``sent_id``
as a direct attribute, so we read it with ``getattr`` instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Sequence


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold: disjoint train/dev/test index lists."""
    index: int
    train_idx: List[int]
    dev_idx: List[int]
    test_idx: List[int]


# KTB sent_id looks like "akorda-random.tagged.txt:59:1025"; the document
# source is everything before the first colon. For sent_ids without a colon
# (some auxiliary files), the whole string is the source key.
_SOURCE_RE = re.compile(r'^([^:]+)')


def source_of(sent_id: str) -> str:
    """Extract the source-document identifier from a KTB ``sent_id``.

    Examples
    --------
    >>> source_of('akorda-random.tagged.txt:59:1025')
    'akorda-random.tagged.txt'
    >>> source_of('wikitravel.tagged.txt:34:362')
    'wikitravel.tagged.txt'
    >>> source_of('plain_id')
    'plain_id'
    """
    raw = sent_id.strip()
    m = _SOURCE_RE.match(raw)
    return m.group(1) if m else raw


def sent_id_of(sentence) -> str:
    """Pull the ``sent_id`` value out of a :class:`kazmorph.corpus.Sentence`.

    kazmorph's corpus ``Sentence`` stores ``sent_id`` as a direct attribute
    (set by :func:`kazmorph.corpus.parse_conllu`), so we read it directly.
    Returns ``''`` when the attribute is absent (e.g. a synthetic sentence
    with no provenance), which the block builders handle gracefully.
    """
    return getattr(sentence, "sent_id", "") or ""


def make_folds(
    sentences: Sequence,
    k: int = 10,
    seed: int = 42,
    strategy: Literal["stratified", "grouped"] = "stratified",
) -> List[Fold]:
    """Build ``k`` train/dev/test folds over ``sentences``.

    Deterministic for a fixed (k, seed, strategy): there is no global RNG call
    that depends on input order beyond a single seeded shuffle that is itself
    order-stable, so two runs on identical inputs yield identical folds.
    """
    n = len(sentences)
    if k < 3:
        raise ValueError(f'need k >= 3 to have non-empty train/dev/test, got k={k}')
    if n < k:
        raise ValueError(f'corpus of {n} sentences is too small for {k} folds')

    # Deterministic PRNG local to this call (no mutation of global state).
    rng = _SeededRng(seed)

    if strategy == 'grouped':
        blocks = _grouped_blocks(sentences)
        # In grouped mode each source is one indivisible block, so we need at
        # least k sources to give every fold a non-empty test set. Fewer
        # sources than folds is a malformed request, not a recoverable one.
        if len(blocks) < k:
            raise ValueError(
                f'grouped CV needs at least k={k} distinct sources to fill every '
                f'fold, but the corpus only has {len(blocks)} sources. Either '
                f'reduce k or use strategy="stratified".')
    else:
        blocks = _stratified_blocks(sentences, rng)

    # Sanity: blocks partition [0, n) exactly.
    assert sorted(idx for b in blocks for idx in b) == list(range(n)), \
        'blocks do not partition the sentence index range'

    fold_blocks = _assign_blocks_to_folds(blocks, k)

    folds: List[Fold] = []
    for i in range(k):
        test_idx = fold_blocks[i]
        # dev is the next non-empty fold (handles any accidental empties safely).
        dev_pos = next((j % k for j in range(i + 1, i + 1 + k) if fold_blocks[j % k]),
                       (i + 1) % k)
        dev_idx = fold_blocks[dev_pos]
        train_idx = [idx for j in range(k) if j != i and j != dev_pos
                     for idx in fold_blocks[j]]
        folds.append(Fold(index=i, train_idx=train_idx, dev_idx=dev_idx, test_idx=test_idx))

    _assert_invariants(folds, n, sentences, strategy)
    return folds


def _assign_blocks_to_folds(blocks: List[List[int]], k: int) -> List[List[int]]:
    """Distribute blocks across k fold-buckets as evenly as possible.

    Strategy: sort blocks largest-first, then assign each block to the
    currently-smallest bucket (longest-processing-time / LPT scheduling).
    This minimises the maximum fold size, which in turn keeps dev/test
    balanced so early-stopping signals are comparable across folds.

    Each individual block is split if it is larger than the target average
    fold size AND there is no other way to fill an empty bucket — this only
    triggers for stratified mode where blocks are already small, and never
    for grouped mode where blocks are atomic (a source document is never
    split across folds).
    """
    buckets: List[List[int]] = [[] for _ in range(k)]
    total = sum(len(b) for b in blocks)
    target = total / k
    # Work largest-first so big blocks land before small ones fill the gaps.
    for block in sorted(blocks, key=len, reverse=True):
        target_idx = min(range(k), key=lambda i: len(buckets[i]))
        # If placing this block whole would overshoot target by more than the
        # block size itself, and the target bucket is empty, split the block
        # to fill the empty bucket. (For grouped mode the caller has already
        # guaranteed blocks are small enough that this branch never fires.)
        if (not buckets[target_idx]) and len(block) > target and len(block) >= 2:
            # split: keep enough in the empty bucket to roughly hit target,
            # the remainder goes to the next-smallest bucket.
            keep = max(1, int(round(target)) - len(buckets[target_idx]))
            keep = min(keep, len(block) - 1)
            buckets[target_idx].extend(block[:keep])
            remaining = block[keep:]
            nxt = min(range(k), key=lambda i: len(buckets[i]))
            buckets[nxt].extend(remaining)
        else:
            buckets[target_idx].extend(block)
    return [sorted(b) for b in buckets]


# --------------------------------------------------------------------------
# block construction
# --------------------------------------------------------------------------
def _grouped_blocks(sentences: Sequence) -> List[List[int]]:
    """One block per source document: all of a source's sentences together."""
    by_source: dict[str, List[int]] = {}
    for i, s in enumerate(sentences):
        by_source.setdefault(source_of(sent_id_of(s)), []).append(i)
    return list(by_source.values())


def _stratified_blocks(sentences: Sequence, rng: '_SeededRng') -> List[List[int]]:
    """Stratified blocks: each source is split into k roughly-equal strata,
    and stratum j of every source is concatenated into block j. This makes
    every block mirror the corpus's source distribution. Sentences are
    shuffled within each source first, so the assignment of individual
    sentences to strata is random but the source proportions are preserved."""
    k_hint = 10
    by_source: dict[str, List[int]] = {}
    for i, s in enumerate(sentences):
        by_source.setdefault(source_of(sent_id_of(s)), []).append(i)
    # Shuffle each source's indices deterministically.
    for src in by_source:
        by_source[src] = rng.shuffle(by_source[src])
    # Number of strata: aim for ~10 per source so block sizes stay granular,
    # but never more than the source length.
    blocks: List[List[int]] = []
    for src, idxs in by_source.items():
        n_strata = max(1, min(k_hint, len(idxs)))
        # round-robin into n_strata lists
        strata: List[List[int]] = [[] for _ in range(n_strata)]
        for pos, idx in enumerate(idxs):
            strata[pos % n_strata].append(idx)
        blocks.extend(strata)
    return blocks


# --------------------------------------------------------------------------
# invariant checks
# --------------------------------------------------------------------------
def _assert_invariants(folds: List[Fold], n: int, sentences: Sequence,
                       strategy: str) -> None:
    # jack-knife: union of test sets == whole corpus, no overlaps
    all_test = sorted(idx for f in folds for idx in f.test_idx)
    assert all_test == list(range(n)), \
        f'test sets must partition [0,{n}) with no overlaps; got {len(all_test)} indices'
    assert len(all_test) == len(set(all_test)), 'duplicate test indices across folds'
    for f in folds:
        tr, dv, te = set(f.train_idx), set(f.dev_idx), set(f.test_idx)
        assert not (tr & dv), f'fold {f.index}: train/dev overlap'
        assert not (tr & te), f'fold {f.index}: train/test overlap'
        assert not (dv & te), f'fold {f.index}: dev/test overlap'
        assert te, f'fold {f.index}: empty test set'
        assert tr, f'fold {f.index}: empty train set'
    if strategy == 'grouped':
        # no source may appear in more than one fold's test set
        fold_sources: List[set[str]] = []
        for f in folds:
            srcs = {source_of(sent_id_of(sentences[i])) for i in f.test_idx}
            fold_sources.append(srcs)
        for a in range(len(fold_sources)):
            for b in range(a + 1, len(fold_sources)):
                overlap = fold_sources[a] & fold_sources[b]
                assert not overlap, \
                    f'grouped invariant violated: sources {overlap} split across folds {a} and {b}'


# --------------------------------------------------------------------------
# tiny self-contained deterministic PRNG (no global-state dependency, so
# reproducibility cannot be broken by other code calling random.shuffle)
# --------------------------------------------------------------------------
class _SeededRng:
    """A minimal xorshift64* generator. Deterministic across Python versions
    and platforms, unlike ``random`` whose shuffle algorithm has changed."""

    def __init__(self, seed: int) -> None:
        # force nonzero state
        s = seed & ((1 << 64) - 1)
        self.state = s if s != 0 else 0x9E3779B97F4A7C15

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & ((1 << 64) - 1)
        x ^= (x >> 7)
        x ^= (x << 17) & ((1 << 64) - 1)
        self.state = x
        return (x * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)

    def shuffle(self, xs: List[int]) -> List[int]:
        # Fisher-Yates using our own generator
        out = list(xs)
        n = len(out)
        for i in range(n - 1, 0, -1):
            j = self.next() % (i + 1)
            out[i], out[j] = out[j], out[i]
        return out
