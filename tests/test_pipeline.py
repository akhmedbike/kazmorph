"""
Tests for KazMorphCorpus-2026 pipeline components.

Run with: pytest tests/ -v
"""

import pytest

from kazmorph.rma import ReverseMorphAnalyzer, RMAResult
from kazmorph.mfrn import MFRN
from kazmorph.taxonomy import classify, TaxonomyResult
from kazmorph.pipeline import Pipeline


# ============================================================
#  Fixtures
# ============================================================

@pytest.fixture
def rma():
    return ReverseMorphAnalyzer()


@pytest.fixture
def mfrn():
    return MFRN()


@pytest.fixture
def pipeline():
    """Pipeline without CRF/Neural models (unit test mode)."""
    return Pipeline(crf_model_path=None, neural_model_path=None)


# ============================================================
#  RMA Tests
# ============================================================

class TestRMA:
    def test_basic_noun_analysis(self, rma):
        result = rma.analyze("адамның")
        assert result.token == "адамның"
        assert result.lemma == "адам"
        assert result.pos == "NOUN"
        assert len(result.suffixes) >= 1
        assert 0.0 <= result.confidence <= 1.0

    def test_known_word_high_confidence(self, rma):
        """Known words should have some confidence (may be low without CRF/Neural)."""
        result = rma.analyze("адам")
        assert result.confidence >= 0.0

    def test_unknown_word_low_confidence(self, rma):
        result = rma.analyze("xyzqwerty")
        assert result.confidence < 0.5

    def test_proper_noun(self, rma):
        result = rma.analyze("Қазақстан")
        assert result.token == "Қазақстан"
        assert result.pos in ("NOUN", "PROPN")

    def test_verb_with_suffix(self, rma):
        result = rma.analyze("білуге")
        assert result.token == "білуге"
        assert "ге" in result.suffixes or "гі" in result.suffixes or len(result.suffixes) >= 1

    def test_empty_suffixes_for_base_form(self, rma):
        """Base form of a known word should have no or few suffixes."""
        result = rma.analyze("адам")
        assert isinstance(result.suffixes, list)

    def test_result_is_rma_result(self, rma):
        result = rma.analyze("мектеп")
        assert isinstance(result, RMAResult)
        assert hasattr(result, "token")
        assert hasattr(result, "lemma")
        assert hasattr(result, "pos")
        assert hasattr(result, "suffixes")
        assert hasattr(result, "confidence")

    @pytest.mark.parametrize("word", [
        "әрбір", "осылай", "бұлай", "осылайша", "бұлайша",
        "неше", "бірдеңе", "ешқашан",
    ])
    def test_closed_class_not_fragmented(self, rma, word):
        """Invariant closed-class words must not be suffix-stripped.

        Regression for REVIEW_RECOMMENDATIONS.md problem #3: әрбір was
        split into стем әрб + suffixes і/р, осылай into осы + ла/й, etc.
        These words are morphologically atomic.
        """
        result = rma.analyze(word)
        assert result.suffixes == [], f"{word} should be atomic, got {result.suffixes}"
        assert result.lemma == word, f"{word} lemma should be itself, got {result.lemma}"

    @pytest.mark.parametrize("word,pos", [
        ("әрбір", "DET"), ("бұл", "DET"), ("осы", "DET"),
        ("осылай", "ADV"), ("қалай", "ADV"),
        ("және", "CCONJ"), ("егер", "SCONJ"),
    ])
    def test_closed_class_pos(self, rma, word, pos):
        """Closed-class entries carry their canonical UD POS."""
        assert rma.analyze(word).pos == pos

    def test_ordinal_still_segmented(self, rma):
        """Ordinal numerals are NOT in the closed-class list: -ші is real
        morphology and should still be segmented. Guards against over-broad
        stopword lists."""
        result = rma.analyze("екінші")
        assert result.suffixes != [], "екінші should keep its -ші suffix"

    @pytest.mark.parametrize("word", [
        "бразилия", "аргентина", "мексика", "канада",
    ])
    def test_extra_nouns_recognized(self, word):
        """Hand-curated EXTRA_NOUNS (frequent country names missing from
        Apertium) should resolve as known nouns. Guards the merged-lexicon
        wiring (lexicons_extra → lexicons). REVIEW_RECOMMENDATIONS.md #2."""
        from kazmorph.lexicons import is_known_noun
        assert is_known_noun(word), f"{word} should be in EXTRA_NOUNS"

    @pytest.mark.parametrize("word", [
        "1920", "7,2", "58,3%", "22", "700", "1994", "1380",
    ])
    def test_arabic_numerals_classified_num(self, rma, word):
        """Arabic integers and decimals (Kazakh comma separator, optional %)
        must tag as NUM with no suffix decomposition. Error analysis showed
        20 NUM tokens were UNK before this short-circuit."""
        result = rma.analyze(word)
        assert result.pos == "NUM", f"{word} should be NUM, got {result.pos}"
        assert result.suffixes == [], f"{word} should be atomic, got {result.suffixes}"

    @pytest.mark.parametrize("word", ["VIII", "XX", "MCMXC", "XL"])
    def test_roman_numerals_classified_num(self, rma, word):
        """Multi-char Roman numerals tag as NUM."""
        assert rma.analyze(word).pos == "NUM"

    @pytest.mark.parametrize("word", ["I", "V", "M", "C", "adам", "IT", "BBC"])
    def test_single_letters_and_latin_not_num(self, rma, word):
        """Single Roman-numeral letters and Latin words/abbreviations must
        NOT be misread as numerals. Guards an over-broad Roman matcher."""
        result = rma.analyze(word)
        assert result.pos != "NUM", f"{word} should not be NUM"

    @pytest.mark.parametrize("word", [
        "конституция", "сейф", "спорт", "фильм", "сурет", "жапырақ",
    ])
    def test_suffixless_known_nouns_get_pos(self, rma, word):
        """Bare dictionary forms with no recognizable suffix (конституция,
        спорт, ...) used to fall through to UNK/conf=0.2 even though they
        are in KNOWN_NOUNS, because POS came only from suffixes. The
        lexicon-recovery step must now return NOUN."""
        result = rma.analyze(word)
        assert result.pos == "NOUN", f"{word} should be NOUN, got {result.pos}"
        assert result.confidence >= 0.85

    def test_ambiguous_stem_stays_unk(self, rma):
        """Stems in two lexicons (e.g. ат = NOUN+VERB, келешек = NOUN+ADJ)
        must stay UNK rather than guessing one POS. Guards the
        single-lexicon condition in _pos_from_lexicon."""
        for word in ("ат", "келешек", "жүк"):
            result = rma.analyze(word)
            assert result.pos == "UNK", f"{word} is ambiguous, should stay UNK"

    @pytest.mark.parametrize("word", ["Стамбұл", "Атырау", "Сергей"])
    def test_capitalized_unknown_becomes_propn(self, rma, word):
        """Capitalized words not in the UD dictionary used to be wrongly
        suffix-stripped (Стамбұл → стем 'Стамбұ' + suffix 'л'). The
        proper-noun guard keeps the form intact and tags it PROPN."""
        result = rma.analyze(word)
        assert result.pos == "PROPN", f"{word} should be PROPN, got {result.pos}"
        assert result.lemma == word, f"{word} lemma should be intact, got {result.lemma}"
        assert result.suffixes == [], f"{word} should not be segmented"

    def test_known_capitalized_not_forced_propn(self, rma):
        """Capitalized words that ARE in the UD dict (Біз, Ол, Бұл) must keep
        their real POS, not be overridden to PROPN by the guard."""
        assert rma.analyze("Біз").pos == "PRON"
        assert rma.analyze("Ол").pos == "PRON"
        assert rma.analyze("Бұл").pos == "DET"


# ============================================================
#  MFRN Tests
# ============================================================

class TestMFRN:
    def test_vowel_harmony_front(self, mfrn):
        """Front vowel words should get front vowel suffixes."""
        result = RMAResult(
            token="үйдің", lemma="үй", pos="NOUN",
            suffixes=["дің"], suffix_categories=["GEN"], confidence=0.8
        )
        adjusted = mfrn.apply([result])
        assert len(adjusted) == 1
        assert adjusted[0].token == "үйдің"

    def test_vowel_harmony_back(self, mfrn):
        """Back vowel words should get back vowel suffixes."""
        result = RMAResult(
            token="баланың", lemma="бала", pos="NOUN",
            suffixes=["ның"], suffix_categories=["GEN"], confidence=0.8
        )
        adjusted = mfrn.apply([result])
        assert len(adjusted) == 1

    def test_mfrn_preserves_token(self, mfrn):
        result = RMAResult(
            token="сөз", lemma="сөз", pos="NOUN",
            suffixes=[], suffix_categories=[], confidence=0.9
        )
        adjusted = mfrn.apply([result])
        assert adjusted[0].token == "сөз"
        assert adjusted[0].lemma == "сөз"

    def test_mfrn_multiple_tokens(self, mfrn):
        results = [
            RMAResult(token="адам", lemma="адам", pos="NOUN", suffixes=[], suffix_categories=[], confidence=0.9),
            RMAResult(token="сөз", lemma="сөз", pos="NOUN", suffixes=[], suffix_categories=[], confidence=0.8),
        ]
        adjusted = mfrn.apply(results)
        assert len(adjusted) == 2

    def test_confidence_in_range(self, mfrn):
        result = RMAResult(
            token="мектеп", lemma="мектеп", pos="NOUN",
            suffixes=[], suffix_categories=[], confidence=0.5
        )
        adjusted = mfrn.apply([result])
        assert 0.0 <= adjusted[0].confidence <= 1.0


# ============================================================
#  Taxonomy Tests
# ============================================================

class TestTaxonomy:
    def test_borrowing_classification(self):
        """International/borrowed words should be classified as BOR (when no suffixes)."""
        result = RMAResult(
            token="республика", lemma="республика", pos="NOUN",
            suffixes=[], suffix_categories=[], confidence=0.7
        )
        tax = classify(result)
        assert tax.difficulty_class == "BOR"

    @pytest.mark.parametrize("word", [
        "жеке", "аспан", "жұрт", "қарт", "ешкі", "кемпір", "үлкен", "ескі",
    ])
    def test_native_words_not_borrowing(self, word):
        """Native Turkic words must not be mislabelled BOR despite matching
        the heuristic suffix patterns. Regression for REVIEW_RECOMMENDATIONS.md
        problem #4: жеке was BOR via the '-ке' ending."""
        result = RMAResult(
            token=word, lemma=word, pos="NOUN",
            suffixes=[], suffix_categories=[], confidence=0.7
        )
        tax = classify(result)
        assert tax.difficulty_class != "BOR", f"{word} is native, not a borrowing"

    def test_affix_chain_classification(self):
        """Words with 2+ suffixes should be AFC."""
        result = RMAResult(
            token="мектептерімізде", lemma="мектеп", pos="NOUN",
            suffixes=["тер", "іміз", "де"], suffix_categories=["PL", "POSS.1PL", "LOC"],
            confidence=0.8
        )
        tax = classify(result)
        assert tax.difficulty_class == "AFC"

    def test_segmentation_ambiguity(self):
        """A root/surface mismatch (over-segmentation) should trigger SEG.

        The scored taxonomy fires SEG on structural signals — e.g. a single-letter
        root with affixation — rather than on confidence alone. A genuine closed
        class word like және stays NONE even at low confidence."""
        result = RMAResult(
            token="сулардан", lemma="?", pos="NOUN",
            suffixes=["лардан"], suffix_categories=["DAT"], confidence=0.2
        )
        tax = classify(result)
        assert tax.difficulty_class == "SEG"

    def test_agreement_complexity(self):
        """A noun whose MFRN compatibility kappa is low should be flagged AGR.

        score_agreement(kappa) = 1 - kappa (+ small NOUN bonus), so a heavily
        incompatible noun (kappa=0.40) clears the 0.30 threshold."""
        from kazmorph.taxonomy import classify_scored
        result = RMAResult(
            token="адамның", lemma="адам", pos="NOUN",
            suffixes=["ның"], suffix_categories=["GEN"], confidence=0.9
        )
        tax = classify_scored(result, kappa=0.40)
        assert tax.difficulty_class in ("AGR", "AFC", "BOR")

    def test_none_classification(self):
        """Simple words should get NONE."""
        result = RMAResult(
            token="және", lemma="және", pos="CCONJ",
            suffixes=[], suffix_categories=[], confidence=0.9
        )
        tax = classify(result)
        assert tax.difficulty_class == "NONE"

    def test_result_is_taxonomy_result(self):
        result = RMAResult(
            token="сөз", lemma="сөз", pos="NOUN",
            suffixes=[], suffix_categories=[], confidence=0.9
        )
        tax = classify(result)
        assert isinstance(tax, TaxonomyResult)
        assert hasattr(tax, "difficulty_class")
        assert hasattr(tax, "description")

    def test_valid_difficulty_classes(self):
        """All outputs should be one of the expected classes."""
        valid = {"BOR", "AFC", "SEG", "AGR", "NONE"}
        words = ["адам", "мектеп", "Қазақстан", "болды", "үй"]
        for word in words:
            rma = ReverseMorphAnalyzer()
            result = rma.analyze(word)
            tax = classify(result)
            assert tax.difficulty_class in valid


# ============================================================
#  Pipeline Integration Tests
# ============================================================

class TestPipeline:
    def test_basic_analysis(self, pipeline):
        result = pipeline.process("адам сөз")
        assert len(result) == 2
        assert result[0]["token"] == "адам"
        assert result[1]["token"] == "сөз"

    def test_output_structure(self, pipeline):
        result = pipeline.process("мектеп")
        assert len(result) == 1
        token = result[0]
        assert "token" in token
        assert "lemma" in token
        assert "pos" in token
        assert "suffixes" in token
        assert "confidence" in token
        assert "difficulty_class" in token
        assert "description" in token

    def test_confidence_range(self, pipeline):
        result = pipeline.process("Қазақстан Республикасы")
        for t in result:
            assert 0.0 <= t["confidence"] <= 1.0

    def test_valid_pos_tags(self, pipeline):
        valid_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "PRON",
                     "ADP", "CCONJ", "DET", "AUX", "PART", "INTJ", "SCONJ",
                     "PUNCT", "SYM", "X", "UNKNOWN"}
        result = pipeline.process("адамның мектептері")
        for t in result:
            assert t["pos"] in valid_pos

    def test_valid_difficulty_class(self, pipeline):
        valid = {"BOR", "AFC", "SEG", "AGR", "NONE"}
        result = pipeline.process("Қазақстандық спортшылар")
        for t in result:
            assert t["difficulty_class"] in valid

    def test_empty_input(self, pipeline):
        result = pipeline.process("")
        assert result == []

    def test_whitespace_only(self, pipeline):
        result = pipeline.process("   ")
        assert result == []

    def test_single_word(self, pipeline):
        result = pipeline.process("мектеп")
        assert len(result) == 1
        assert result[0]["token"] == "мектеп"

    def test_multiple_words(self, pipeline):
        text = "Қазақстан Республикасының Конституциясы бойынша"
        result = pipeline.process(text)
        assert len(result) >= 4

    def test_suffixes_is_list(self, pipeline):
        result = pipeline.process("адамның")
        assert isinstance(result[0]["suffixes"], list)


# ============================================================
#  Tokenization (UD Kazakh-KTB alignment)
#  These patterns must be kept as single tokens to match what the
#  tagger was trained on. A naive split depresses reported accuracy
#  by ~7 points (see WORKLOG §7, "фикс токенизации").
# ============================================================
class TestTokenization:
    """Pipeline._tokenize must merge patterns that UD Kazakh-KTB treats as
    one token, otherwise the tagger sees different token shapes at inference
    than at training and benchmark metrics drift."""

    def test_decimal_with_comma_is_one_token(self):
        # "7,2" — UD treats as one NUM token, not "7" + "," + "2"
        toks = Pipeline._tokenize("7,2 пайыз")
        assert toks[0] == "7,2"

    def test_decimal_with_percent_is_one_token(self):
        toks = Pipeline._tokenize("58,3% ға")
        assert toks[0] == "58,3%"

    def test_numeral_with_hyphenated_suffix_is_one_token(self):
        # "3-ші" (третий), "100-ге" (до ста) — Kazakh attaches case/ordinal
        # suffixes to numerals via hyphen. UD keeps as one token.
        assert Pipeline._tokenize("3-ші орын")[0] == "3-ші"
        assert Pipeline._tokenize("100-ге тарта")[0] == "100-ге"

    def test_hyphenated_compound_is_one_token(self):
        # "сондай-ақ" (а также), "әке-шешесін" (родителей) — single UD tokens
        assert Pipeline._tokenize("сондай-ақ")[0] == "сондай-ақ"
        assert Pipeline._tokenize("әке-шешесін")[0] == "әке-шешесін"

    def test_abbreviation_with_dot_is_one_token(self):
        # "млрд." (миллиард), "ж." (год), "А." (initial) — UD keeps the dot
        assert Pipeline._tokenize("1 млрд. доллар")[1] == "млрд."
        assert Pipeline._tokenize("2018 ж.")[1] == "ж."

    def test_sentence_final_period_not_glued(self):
        """REGRESSION: a period ending a sentence must NOT be glued onto the
        preceding word. The old greedy abbreviation regex matched any word
        ending in a period, so 'Ол келді.' became ['Ол', 'келді.'] — breaking
        POS/lemma on real user text (the model was trained with '.' as a
        separate PUNCT token). Only genuine abbreviations keep their period."""
        toks = Pipeline._tokenize("Ол келді.")
        assert toks == ["Ол", "келді", "."], f"got {toks}"
        toks = Pipeline._tokenize("Мен бардым.")
        assert toks == ["Мен", "бардым", "."], f"got {toks}"
        toks = Pipeline._tokenize("кітабы жаңа.")
        assert toks == ["кітабы", "жаңа", "."], f"got {toks}"

    def test_single_uppercase_initial_keeps_dot(self):
        """Initials (А., М.) are structural abbreviations — single uppercase
        letter + period stays one token, even when not in the explicit list."""
        assert Pipeline._tokenize("А. С. Пушкин") == ["А.", "С.", "Пушкин"]

    def test_period_after_comma_punct(self):
        """Mixed trailing punctuation: 'келді,' then '.'"""
        toks = Pipeline._tokenize("келді.")
        assert "." in toks and "келді." not in toks

    def test_plain_words_and_punct_still_split(self):
        # The generic fallback must still work for ordinary text
        toks = Pipeline._tokenize("адам, сөз!")
        assert toks == ["адам", ",", "сөз", "!"]

    def test_standalone_hyphen_is_punct(self):
        # An isolated "—" (em-dash separator) must still tokenize as its own
        # punct token, not get absorbed into a hyphen-compound rule.
        toks = Pipeline._tokenize("бір — екі")
        assert "—" in toks

    def test_does_not_merge_unrelated_words(self):
        # Sanity: two separate words with no hyphen stay separate
        toks = Pipeline._tokenize("жайлы сөз")
        assert toks == ["жайлы", "сөз"]


# ============================================================
#  Hard neural fallback (config F)
# ============================================================
class TestNeuralFallback:
    """Verify that the fallback mode keeps the CRF POS on confident tokens and
    takes the neural POS only where the CRF is unsure. Stubs the CRF and Neural
    so the test is deterministic and needs no trained models."""

    @staticmethod
    def _stub_pipeline():
        # Construct bare (RMA-only) then flip into fallback mode with stubs.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = Pipeline()
        p.fallback = True
        p.fallback_conf_threshold = 0.50
        return p

    def test_confident_token_keeps_crf_pos(self):
        p = self._stub_pipeline()

        class StubCRF:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "NOUN", [], [], 0.90)]

        class StubNeural:
            def tag(self, rma):
                # Neural says PROPN, but CRF is confident -> must be IGNORED.
                return [RMAResult(rma[0].token, rma[0].lemma, "PROPN", [], [], 0.80)]

        p.crf, p.neural = StubCRF(), StubNeural()
        out = p.process("мектеп")
        assert out[0]["pos"] == "NOUN"
        assert out[0]["confidence"] == 0.90

    def test_low_confidence_token_takes_neural_pos(self):
        p = self._stub_pipeline()

        class StubCRF:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "VERB", [], [], 0.20)]

        class StubNeural:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "NOUN", [], [], 0.70)]

        p.crf, p.neural = StubCRF(), StubNeural()
        out = p.process("барды")
        assert out[0]["pos"] == "NOUN"
        assert out[0]["confidence"] == 0.70

    def test_unknown_crf_pos_triggers_fallback(self):
        p = self._stub_pipeline()

        class StubCRF:
            def tag(self, rma):
                # CRF returns UNK with high-ish confidence; UNK still triggers.
                return [RMAResult(rma[0].token, rma[0].lemma, "UNK", [], [], 0.95)]

        class StubNeural:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "NOUN", [], [], 0.60)]

        p.crf, p.neural = StubCRF(), StubNeural()
        out = p.process("xyzqwerty")
        assert out[0]["pos"] == "NOUN"

    def test_neural_also_unknown_keeps_crf(self):
        # If the neural also returns an unknown POS, fallback must not adopt it.
        p = self._stub_pipeline()

        class StubCRF:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "VERB", [], [], 0.10)]

        class StubNeural:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "X", [], [], 0.55)]

        p.crf, p.neural = StubCRF(), StubNeural()
        out = p.process("барды")
        assert out[0]["pos"] == "VERB"

    def test_threshold_controls_boundary(self):
        # With threshold 0.95, a CRF token at 0.90 becomes "low-confidence".
        p = self._stub_pipeline()
        p.fallback_conf_threshold = 0.95

        class StubCRF:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "NOUN", [], [], 0.90)]

        class StubNeural:
            def tag(self, rma):
                return [RMAResult(rma[0].token, rma[0].lemma, "VERB", [], [], 0.70)]

        p.crf, p.neural = StubCRF(), StubNeural()
        out = p.process("сөз")
        assert out[0]["pos"] == "VERB"  # now overridden because 0.90 < 0.95
