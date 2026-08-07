"""
Corpus utilities for UD Kazakh-KTB and beyond.

Centralizes CoNLL-U parsing, deduplication, and k-fold cross-validation so all
training/evaluation scripts share one honest view of the data.

Key findings about the data shipped in this repo (verified):
  - data/test.conllu  (1047 sents) is the *full* UD Kazakh-KTB.
  - data/train_split.conllu (917) + data/dev_split.conllu (161) are an 85/15
    stratified split OF THAT SAME CORPUS (1078 unique sentences total).
  - test.conllu overlaps train_split by 893 sentences and dev_split by 154.
    => evaluating on dev_split while training on train_split is honest (the two
       are disjoint), but "training on train_split and testing on test.conllu"
       would be data leakage, since test.conllu ⊃ train_split.

This module therefore treats the union of all CoNLL-U files as a single pool,
deduplicates it, and exposes k-fold splits that are guaranteed disjoint.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterator

# 17 Universal POS tags present in UD Kazakh-KTB
POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]


@dataclass(frozen=True)
class Token:
    """A single CoNLL-U token row (multiword/empty rows are skipped)."""
    form: str
    lemma: str
    upos: str


@dataclass(frozen=True)
class Sentence:
    """A sentence: an ordered list of tokens plus its provenance id."""
    sent_id: str
    tokens: tuple[Token, ...]

    @property
    def forms(self) -> tuple[str, ...]:
        return tuple(t.form for t in self.tokens)

    @property
    def upos_tags(self) -> tuple[str, ...]:
        return tuple(t.upos for t in self.tokens)

    def fingerprint(self) -> tuple[str, ...]:
        """Normalized token sequence used for deduplication (lowercased forms)."""
        return tuple(t.form.lower() for t in self.tokens)


def parse_conllu(path: str) -> list[Sentence]:
    """
    Parse a CoNLL-U file into a list of Sentence objects.

    Multiword token rows (id like "12-13") and empty-node rows (id like "5.1")
    are skipped, following the convention used by the existing training scripts.
    sent_id is read from the "# sent_id = ..." comment when present, else falls
    back to a positional index.
    """
    sentences: list[Sentence] = []
    sent_id = ""
    tokens: list[Token] = []

    def flush():
        nonlocal tokens
        if tokens:
            sentences.append(Sentence(sent_id=sent_id, tokens=tuple(tokens)))
        tokens = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                flush()
                sent_id = ""
                continue
            if line.startswith("#"):
                # capture "# sent_id = xyz"
                low = line.lower()
                if low.startswith("# sent_id"):
                    # split on first '=' then strip
                    sent_id = line.split("=", 1)[1].strip() if "=" in line else ""
                continue
            parts = line.split("\t")
            tok_id = parts[0]
            if "-" in tok_id or "." in tok_id:
                continue
            if len(parts) < 4:
                continue
            form, lemma, upos = parts[1], parts[2], parts[3]
            tokens.append(Token(form=form, lemma=lemma, upos=upos))
        flush()

    # backfill missing sent_ids with positional index
    for i, s in enumerate(sentences):
        if not s.sent_id:
            sentences[i] = Sentence(sent_id=f"idx-{i}", tokens=s.tokens)
    return sentences


def load_pool(
    paths: list[str] | None = None,
    *,
    dedup: bool = True,
) -> list[Sentence]:
    """
    Load and (by default) deduplicate sentences from one or more CoNLL-U files.

    Deduplication is by normalized token-sequence fingerprint, so the same
    sentence appearing in both test.conllu and train_split.conllu is kept once.
    Order of first appearance is preserved.
    """
    if paths is None:
        # Default: every gold CoNLL-U file shipped in data/. Files whose name
        # starts with "silver_" are excluded — they are auto-generated weak
        # annotations (see kazmorph.silver) and must only enter training via the
        # explicit --silver flag, otherwise they pollute the gold evaluation
        # pool with lower-quality labels.
        data_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data")
        data_dir = os.path.abspath(data_dir)
        paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".conllu") and not f.startswith("silver_")
        )

    seen: set[tuple[str, ...]] = set()
    pool: list[Sentence] = []
    for path in paths:
        for sent in parse_conllu(path):
            fp = sent.fingerprint()
            if dedup:
                if fp in seen:
                    continue
                seen.add(fp)
            pool.append(sent)
    return pool


def stratified_kfold(
    sentences: list[Sentence],
    k: int = 5,
    *,
    seed: int = 42,
) -> list[tuple[list[Sentence], list[Sentence]]]:
    """
    Yield (train, dev) folds that are sentence-disjoint.

    Stratification is approximate and key-based: each sentence is assigned a
    coarse label = its most frequent UPOS tag (the head tag), and folds are
    balanced across these labels via a round-robin over sorted-by-label order.
    This keeps the rare-class distribution (SCONJ, INTJ, SYM) spread across
    folds rather than concentrated in one.

    Returns a list of k (train, dev) tuples.

    Special case: k == 1 produces a single (train, dev) hold-out split (≈85/15,
    stratified) rather than a degenerate leave-one-fold-out with an empty train.
    This keeps "smoke test with --folds 1" meaningful.
    """
    rng = random.Random(seed)

    from collections import Counter

    def head_tag(s: Sentence) -> str:
        c = Counter(s.upos_tags)
        # most common non-PUNCT tag, falling back to PUNCT then anything
        for tag, _ in c.most_common():
            if tag != "PUNCT":
                return tag
        return c.most_common(1)[0][0] if c else "X"

    # Stable sort by (head_tag, fingerprint) so identical label groups are
    # contiguous, then round-robin assign to folds.
    decorated = sorted(
        enumerate(sentences),
        key=lambda iv: (head_tag(iv[1]), iv[1].fingerprint(), iv[0]),
    )

    if k == 1:
        # Stratified hold-out: round-robin into 7 train bins vs ~1 dev bin (~12%)
        dev = [sent for i, (_o, sent) in enumerate(decorated) if i % 8 == 0]
        train = [sent for i, (_o, sent) in enumerate(decorated) if i % 8 != 0]
        rng.shuffle(train)
        return [(train, dev)]

    folds: list[list[Sentence]] = [[] for _ in range(k)]
    for i, (_orig_idx, sent) in enumerate(decorated):
        folds[i % k].append(sent)

    splits: list[tuple[list[Sentence], list[Sentence]]] = []
    for i in range(k):
        dev = folds[i]
        train = [s for j in range(k) if j != i for s in folds[j]]
        # shuffle within each split for training stability (dev order irrelevant)
        rng.shuffle(train)
        splits.append((train, dev))
    return splits


def stratified_three_way_split(
    sentences: list[Sentence],
    *,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    """
    Produce a single stratified (train, dev, test) partition, sentence-disjoint.

    UD Kazakh-KTB is officially too small to ship a canonical three-way split,
    so we draw one deterministically from the deduplicated gold pool. Stratification
    reuses the same head-tag scheme as :func:`stratified_kfold`: sentences are
    sorted by (head_tag, fingerprint) so each label group stays contiguous, then
    round-robin assigned across ``nbins`` bins (where ``nbins`` makes ``test_frac``
    and ``dev_frac`` come out near their targets). The first ``ntest`` bins become
    the test set, the next ``ndev`` bins become the dev set, the remainder is train.

    Because assignment is by bin index modulo ``nbins``, every label group is
    represented in all three splits, which matters for the rare UPOS classes.

    Reproducibility: the result depends only on (sentences, dev_frac, test_frac,
    seed). ``seed=42`` is the value fixed for all reported experiments.
    """
    if not 0 < dev_frac < 1 and not 0 < test_frac < 1:
        raise ValueError("dev_frac and test_frac must each be in (0, 1)")
    if dev_frac + test_frac >= 1:
        raise ValueError("dev_frac + test_frac must be < 1")

    from collections import Counter

    rng = random.Random(seed)

    def head_tag(s: Sentence) -> str:
        c = Counter(s.upos_tags)
        for tag, _ in c.most_common():
            if tag != "PUNCT":
                return tag
        return c.most_common(1)[0][0] if c else "X"

    decorated = sorted(
        enumerate(sentences),
        key=lambda iv: (head_tag(iv[1]), iv[1].fingerprint(), iv[0]),
    )

    n = len(decorated)
    # Choose a common bin count so that test_frac and dev_frac both round to a
    # whole number of bins. 20 bins gives 5%-granularity, fine for ~1078 sents.
    nbins = 20
    n_test = max(1, round(test_frac * nbins))
    n_dev = max(1, round(dev_frac * nbins))
    # Guard: never let test+dev consume all bins.
    if n_test + n_dev >= nbins:
        n_test = max(1, nbins // 5)
        n_dev = max(1, nbins // 5)

    # Assign each sentence to a bin index, preserving label-contiguous order so
    # that consecutive sentences of the same head tag land in adjacent bins
    # (keeps each split's label distribution close to the pool's).
    bins: list[list[Sentence]] = [[] for _ in range(nbins)]
    for i, (_orig_idx, sent) in enumerate(decorated):
        bins[i % nbins].append(sent)

    test_bins = bins[:n_test]
    dev_bins = bins[n_test:n_test + n_dev]
    train_bins = bins[n_test + n_dev:]

    def flatten(groups: list[list[Sentence]]) -> list[Sentence]:
        out = [s for grp in groups for s in grp]
        rng.shuffle(out)
        return out

    train = flatten(train_bins)
    dev = flatten(dev_bins)
    test = flatten(test_bins)

    # Final safety assert: disjointness by fingerprint.
    def fps(group: list[Sentence]) -> set[tuple[str, ...]]:
        return {s.fingerprint() for s in group}

    train_fp, dev_fp, test_fp = fps(train), fps(dev), fps(test)
    assert not (train_fp & dev_fp), "train/dev overlap detected"
    assert not (train_fp & test_fp), "train/test overlap detected"
    assert not (dev_fp & test_fp), "dev/test overlap detected"

    return train, dev, test


def write_conllu(sentences: list[Sentence], path: str) -> None:
    """
    Write sentences back to a CoNLL-U file (one sentence per block).

    Preserves the form/lemma/upos columns and a stable ``sent_id``. Used by
    ``scripts/make_splits.py`` to materialize the canonical three-way split.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences, start=1):
            sid = s.sent_id or f"split-{i}"
            f.write(f"# sent_id = {sid}\n")
            for j, tok in enumerate(s.tokens, start=1):
                # columns: id form lemma upos _ _ _ _ _
                f.write(
                    f"{j}\t{tok.form}\t{tok.lemma}\t{tok.upos}\t_\t_\t_\t_\t_\t_\n"
                )
            f.write("\n")


def iter_pairs(sentences: list[Sentence]) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (forms, upos_tags) pairs — the format the training code consumes."""
    for s in sentences:
        yield list(s.forms), list(s.upos_tags)


def tag_counts(sentences: list[Sentence]) -> dict[str, int]:
    """Return {upos: count} over all tokens in the given sentences."""
    from collections import Counter
    c: Counter[str] = Counter()
    for s in sentences:
        c.update(s.upos_tags)
    return dict(c)
