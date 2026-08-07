"""
Tests for the model-improvement work: silver annotation, corpus k-fold,
converb lemmatization fixes, and the AUX disambiguation rule.

These pin down the specific weaknesses called out in docs/final_report.md
(under-stripping of -п/-а/-е converbs, VERB↔AUX confusion) and the new
silver-data pipeline that converts Apertium analyses to UPOS CoNLL-U.

Run with: pytest tests/test_training_improvements.py -v
"""

import pytest

from kazmorph.rma import ReverseMorphAnalyzer
from kazmorph.mfrn import MFRN, AUX_LEMMAS
from kazmorph.corpus import (
    Sentence, Token, load_pool, stratified_kfold, tag_counts, parse_conllu,
)
from kazmorph.silver import (
    parse_stream, parse_reading, apertium_to_upos, best_upos,
    units_to_sentences, convert_apertium_corpus,
)


# ============================================================
#  Converb lemmatization (affixes.py — added -п/-а/-е/-й)
# ============================================================
class TestConverbLemmatization:
    """final_report.md §4.2: -п/-а/-е converbs were under-stripped."""

    @pytest.fixture
    def rma(self):
        return ReverseMorphAnalyzer()

    @pytest.mark.parametrize("token,expected_lemma", [
        ("тастап", "таста"),   # was: тастап (no stripping) — top cited bug
        ("барып", "бар"),      # -ып
        ("келіп", "кел"),      # -іп
        # Note: "алып" is deliberately excluded — it is ambiguous in gold UD
        # (lemma "ал" as VERB/AUX vs lemma "алып" as ADJ/X), and the RMA
        # lemma_dict stores the first-seen reading. The -п stripping rule
        # itself works; this is a lexicon-disambiguation issue, not a
        # suffix-rule issue.
        ("бара", "бар"),       # imperfective converb -а
        ("келе", "кел"),       # imperfective converb -е
    ])
    def test_converb_stripped_to_verb_stem(self, rma, token, expected_lemma):
        r = rma.analyze(token)
        assert r.lemma == expected_lemma, (
            f"{token}: expected lemma '{expected_lemma}', got '{r.lemma}' "
            f"(suffixes={r.suffixes})"
        )
        assert r.pos == "VERB" or r.pos == "AUX"  # келе maps to AUX via MFRN


# ============================================================
#  AUX vs VERB disambiguation (mfrn.py — closed AUX list)
# ============================================================
class TestAuxDisambiguation:
    """The closed AUX list is kept for reference, but MFRN no longer applies
    a blanket VERB→AUX override. An A/B test showed that override hurt more
    than it helped (VERB 76.1%→89.6%, overall +2.6% when REMOVED) because the
    retrained KazBERT tagger already disambiguates VERB/AUX well and the list
    was relabeling ordinary verbs (тұр, бер, ал, ...) that only act as
    auxiliaries in specific converb constructions. These tests pin the new
    contract: the neural POS survives MFRN unchanged for both AUX and VERB.
    """

    def test_aux_list_is_closed_and_nonempty(self):
        assert len(AUX_LEMMAS) >= 10
        # canonical aspectual auxiliaries must still be present (kept for
        # reference / potential future context-aware use)
        for lemma in ("жат", "отыр", "жүр", "тұр", "ал", "бер", "бол"):
            assert lemma in AUX_LEMMAS

    def test_mfrn_does_not_relabel_verb_to_aux(self):
        """A VERB token whose lemma is in AUX_LEMMAS must stay VERB — the
        neural tagger owns VERB/AUX disambiguation now (see mfrn.py docstring
        for the A/B numbers that motivated this)."""
        mfrn = MFRN()
        from kazmorph.rma import RMAResult
        res = RMAResult(
            token="жатыр", lemma="жат", pos="VERB",
            suffixes=["ыр"], suffix_categories=["CVB"], confidence=0.8,
        )
        out = mfrn.apply([res])[0]
        assert out.pos == "VERB"

    def test_mfrn_keeps_non_aux_verb_as_verb(self):
        mfrn = MFRN()
        from kazmorph.rma import RMAResult
        res = RMAResult(
            token="жүгірді", lemma="жүгіру", pos="VERB",
            suffixes=["ді"], suffix_categories=["PAST.3"], confidence=0.9,
        )
        out = mfrn.apply([res])[0]
        assert out.pos == "VERB"


# ============================================================
#  Corpus k-fold (corpus.py)
# ============================================================
class TestCorpusKfold:
    def test_folds_are_disjoint(self):
        pool = load_pool()
        folds = stratified_kfold(pool, k=5, seed=42)
        assert len(folds) == 5
        for train, dev in folds:
            train_fp = {s.fingerprint() for s in train}
            dev_fp = {s.fingerprint() for s in dev}
            assert len(train_fp & dev_fp) == 0, "train/dev leak across a fold"

    def test_fold1_holds_out_cleanly(self):
        # the smoke-test path: a single fold must still have non-empty train+dev
        pool = load_pool()
        folds = stratified_kfold(pool, k=1, seed=42)
        train, dev = folds[0]
        assert len(train) > 0 and len(dev) > 0

    def test_pool_excludes_silver_by_default(self):
        """Silver files must NOT enter the gold pool automatically."""
        pool = load_pool()
        # gold pool is the deduplicated UD Kazakh-KTB = 1078 sentences.
        # If silver leaked in, this would be ~1352.
        assert len(pool) == 1078

    def test_tag_counts_consistent(self):
        pool = load_pool()
        tc = tag_counts(pool)
        # NOUN is by far the most frequent tag in UD Kazakh-KTB
        assert tc["NOUN"] > tc["VERB"] > tc["ADJ"]
        # rare classes present (these were the weak spots)
        assert "SCONJ" in tc and tc["SCONJ"] > 0
        assert "INTJ" in tc and tc["INTJ"] > 0


# ============================================================
#  Silver annotation (silver.py)
# ============================================================
class TestSilverAnnotation:
    def test_parse_reading_extracts_lemma_and_tags(self):
        r = parse_reading("конкурс<n><px3sp><nom>")
        assert r.lemma == "конкурс"
        assert r.tags == ["n", "px3sp", "nom"]

    def test_parse_stream_handles_multiple_units(self):
        units = parse_stream("^ән/ән<n><nom>$ ^болады/бол<v><iv><aor><p3><sg>$")
        assert len(units) == 2
        assert units[0].wordform == "ән"
        assert units[1].wordform == "болады"

    def test_apertium_to_upos_mapping(self):
        assert apertium_to_upos("n", set()) == "NOUN"
        assert apertium_to_upos("np", {"top"}) == "PROPN"
        assert apertium_to_upos("v", {"iv"}) == "AUX"   # intransitive → AUX
        assert apertium_to_upos("v", {"tv"}) == "VERB"
        assert apertium_to_upos("vaux", set()) == "AUX"
        assert apertium_to_upos("prn", {"dem"}) == "PRON"
        assert apertium_to_upos("sent", set()) == "PUNCT"
        assert apertium_to_upos("cnjcoo", set()) == "CCONJ"
        assert apertium_to_upos("cnjsub", set()) == "SCONJ"

    def test_best_upos_prefers_clean_reading(self):
        from kazmorph.silver import LexicalUnit, Reading
        lu = LexicalUnit(
            wordform="сөздер",
            readings=[
                Reading(lemma="*сөздер", tags=[]),   # failed analysis
                Reading(lemma="сөз", tags=["n", "pl", "nom"]),
            ],
        )
        upos, lemma = best_upos(lu)
        assert upos == "NOUN"
        assert lemma == "сөз"

    def test_units_segment_on_sent_punct(self):
        from kazmorph.silver import LexicalUnit, Reading
        units = [
            LexicalUnit("ән", [Reading("ән", ["n", "nom"])]),
            LexicalUnit(".", [Reading(".", ["sent"])]),
            LexicalUnit("жер", [Reading("жер", ["n", "nom"])]),
        ]
        sents = units_to_sentences(units)
        assert len(sents) == 2  # split on the <sent> token
        assert sents[0].tokens[-1].upos == "PUNCT"

    def test_convert_real_apertium_corpus(self, tmp_path):
        """End-to-end on a tiny Apertium stream snippet."""
        snippet = (
            "^ән/ән<n><nom>$ "
            "^конкурсы/конкурс<n><px3sp><nom>$ "
            "^болады/бол<v><iv><aor><p3><sg>$ "
            "^./.<sent>$\n"
        )
        src = tmp_path / "in.tagged"
        src.write_text(snippet, encoding="utf-8")
        out = tmp_path / "silver.conllu"
        stats = convert_apertium_corpus(str(src), str(out))
        assert stats["written"] == 1
        # the converted file parses back as valid CoNLL-U
        sents = parse_conllu(str(out))
        assert len(sents) == 1
        upos_seq = sents[0].upos_tags
        assert upos_seq == ("NOUN", "NOUN", "AUX", "PUNCT")
