"""
Tests for the neural lemmatizer stage (kazmorph/neural_lemma.py).

Covers: the consonant-alternation repair rules, stub-mode pass-through, and
the prefix-length target encoding shared with the training script.
"""

import pytest

from kazmorph.rma import RMAResult
from kazmorph.neural_lemma import NeuralLemmatizer, _repair_lemma, MAX_PREFIX


class TestRepairLemma:
    """The post-processing step that fixes stem-final consonant alternations."""

    def test_no_change_when_keep_len_zero(self):
        # keep_len == 0 means "lemma is the whole word" (uninflected)
        assert _repair_lemma("алматы", 0) == "алматы"

    def test_keep_len_beyond_word_returns_whole_word(self):
        assert _repair_lemma("бала", 99) == "бала"

    def test_simple_prefix_cut(self):
        # барысында → барыс (keep 5), final 'с' is not in devoice set → stays
        assert _repair_lemma("барысында", 5) == "барыс"

    def test_devoice_final_consonant_at_cut(self):
        # бастығы → keep 6 (incl. ғ) → devoice final ғ → қ, giving "бастық".
        # Devoicing now requires a POS where қ/ғ alternation occurs (NOUN/ADJ)
        # and lexicon confirmation, so we pass pos="NOUN" and rely on "бастық"
        # being a known noun while "бастығ" is not.
        assert _repair_lemma("бастығы", 6, pos="NOUN") == "бастық"

    def test_devoice_maps(self):
        # Cyrillic only — the devoice map operates on Cyrillic consonants.
        # All cases need a POS that allows devoicing (NOUN) and a lexicon hit.
        assert _repair_lemma("аймағы", 5, pos="NOUN") == "аймақ"    # ғ → қ
        assert _repair_lemma("кітеп", 5, pos="NOUN") == "кітеп"     # whole-word, no change
        assert _repair_lemma("бастығы", 6, pos="NOUN") == "бастық"  # ғ → қ

    def test_no_repair_when_whole_word_is_lemma(self):
        # keep_len == len(form): the whole word IS the lemma, no alternation repair
        assert _repair_lemma("жол", 3) == "жол"
        assert _repair_lemma("бастық", 6) == "бастық"  # қ stays қ, no devoice

    def test_short_prefix_preserved(self):
        # "жүріп"[:4] = "жүрі" (final 'і' is a vowel → no repair)
        assert _repair_lemma("жүріп", 4) == "жүрі"
        # "жүріп"[:3] = "жүр" (final 'р' not in devoice set → stays)
        assert _repair_lemma("жүріп", 3) == "жүр"

    # ── Regression: over-aggressive devoicing of voiced-final roots ──────────
    # The earlier unconditional devoice (vowel-after-cut → flip) turned valid
    # lemmas ending in a genuine voiced consonant into non-words: өзі→өс,
    # көзі→көс, алдында→алт, сөзі→сөс. These roots keep their voiced consonant
    # in the dictionary form; devoicing must be gated on lexicon + POS.
    def test_no_devoice_for_pronoun_root(self):
        # өзі → өз (PRON), not өс. PRON is not in the devoice-allowed POS set,
        # and even with a noun-style lookup the lexicon does not confirm өс.
        assert _repair_lemma("өзі", 2, pos="PRON") == "өз"

    def test_no_devoice_for_known_noun_voiced_root(self):
        # көзі → көз (NOUN): көз is a known noun, көс is not a noun → keep voiced.
        assert _repair_lemma("көзі", 3, pos="NOUN") == "көз"
        # сөзі → сөз (NOUN): same logic.
        assert _repair_lemma("сөзі", 3, pos="NOUN") == "сөз"

    def test_no_devoice_for_closed_class_root(self):
        # алдында → алд (ADP): closed class, devoicing never applies.
        assert _repair_lemma("алдында", 3, pos="ADP") == "алд"
        # сізде → сіз (ADP): same.
        assert _repair_lemma("сізде", 3, pos="ADP") == "сіз"

    def test_devoice_only_when_lexicon_confirms(self):
        # бастығы → бастық (NOUN): бастық is a known noun, бастығ is not.
        assert _repair_lemma("бастығы", 6, pos="NOUN") == "бастық"
        # аймағы → аймақ (NOUN): аймақ is a known noun.
        assert _repair_lemma("аймағы", 5, pos="NOUN") == "аймақ"
        # құқығы → құқық (NOUN): құқық is a known noun.
        assert _repair_lemma("құқығы", 5, pos="NOUN") == "құқық"

    def test_devoice_skipped_when_voiced_form_also_known(self):
        # көз is a known NOUN; even with a noun-style devoice attempt we must
        # NOT overwrite it with көс. This guards against the (voiced_ok=True)
        # branch: keep the voiced lemma when it is itself a valid lexicon hit.
        assert _repair_lemma("көзді", 3, pos="NOUN") == "көз"


class TestNeuralLemmatizerStub:
    """When no model is loaded, the lemmatizer must pass tokens through."""

    def test_stub_returns_input_unchanged(self):
        lem = NeuralLemmatizer(model_path=None)
        assert lem.model is None
        toks = [
            RMAResult(token="бала", lemma="бала", pos="NOUN",
                      suffixes=[], suffix_categories=[], confidence=0.9),
            RMAResult(token="жолдар", lemma="жол", pos="NOUN",
                      suffixes=["дар"], suffix_categories=["PL"], confidence=0.9),
        ]
        out = lem.tag(toks)
        assert out is toks  # same list object, unmodified
        assert out[0].lemma == "бала"
        assert out[1].lemma == "жол"

    def test_stub_with_nonexistent_path(self):
        lem = NeuralLemmatizer(model_path="/nonexistent/path/xyz")
        assert lem.model is None  # graceful: no crash, stays in stub mode


class TestPrefixLengthEncoding:
    """The label encoding must agree between training and inference."""

    def test_prefix_label_basic(self):
        from kazmorph.neural_lemma import _repair_lemma
        # form "барысында", lemma "барыс" → keep first 5 chars
        # the training script's prefix_label returns the common-prefix length
        # Here we just confirm the repair produces the lemma from that length.
        assert _repair_lemma("барысында", 5) == "барыс"

    def test_max_prefix_cap(self):
        assert MAX_PREFIX == 24  # kept in sync with scripts/train_lemma.py


class TestPrefixLabelAlternation:
    """The conditional +1 bump in prefix_label for stem-alternation pairs.

    This is the core of the q/g fix (REVIEW_RECOMMENDATIONS.md problem #1):
    when form and lemma differ by a voicing/devoicing pair at the cut point,
    the label must INCLUDE the voiced consonant so _apply_alternation can
    devoice it. Without the bump, the model learns to cut before the
    alternating consonant and the repair never fires.
    """

    @pytest.mark.parametrize("form,lemma,expected", [
        # Alternation pairs: label must be LCP+1 (include voiced consonant)
        ("мектебі", "мектеп", 6),   # LCP=5 (мекте), +1 to include б → devoice to п
        ("билігін", "билік", 5),    # LCP=4 (білі), +1 for г → к
        ("орталығы", "орталық", 7), # LCP=6, +1 for ғ → қ
        ("бастығы", "бастық", 6),   # LCP=5, +1 for ғ → қ
        ("құқығы", "құқық", 5),     # LCP=4, +1 for ғ → қ
        ("аймағы", "аймақ", 5),     # LCP=4, +1 for ғ → қ
        ("кітабы", "кітап", 5),     # LCP=4, +1 for б → п
        ("еңбегі", "еңбек", 5),     # LCP=4, +1 for г → к
    ])
    def test_alternation_bumps_to_lcp_plus_one(self, form, lemma, expected):
        from scripts.train_lemma import prefix_label
        assert prefix_label(form, lemma) == expected

    @pytest.mark.parametrize("form,lemma,expected", [
        # Non-alternation: label must stay at LCP (no bump). These are the
        # load-bearing guards against an over-broad +1 rule.
        ("адамның", "адам", 4),     # plain suffix stripping, form[LCP]='ң' not voiced
        ("мектеп", "мектеп", 6),    # form == lemma, full length
        ("үй", "үй", 2),            # form == lemma, short
        ("қыздар", "қыз", 3),       # plain plural, form[LCP]='д' would FALSE-bump if
                                    # unconditional: 'д' is voiced but lemma[3]='з'≠'т'
                                    # so the pair-check correctly blocks it
        ("бала", "бала", 4),        # form == lemma
        ("сөздер", "сөз", 3),       # plain plural, no alternation
    ])
    def test_non_alternation_stays_at_lcp(self, form, lemma, expected):
        from scripts.train_lemma import prefix_label
        assert prefix_label(form, lemma) == expected

    def test_roundtrip_alternation_pairs(self):
        """For every alternation pair, (prefix_label → _apply_alternation)
        must reconstruct the gold lemma. This is the property the model will
        learn to satisfy."""
        from scripts.train_lemma import prefix_label
        from kazmorph.neural_lemma import _repair_lemma
        pairs = [
            ("мектебі", "мектеп"), ("билігін", "билік"), ("орталығы", "орталық"),
            ("бастығы", "бастық"), ("құқығы", "құқық"), ("аймағы", "аймақ"),
            ("кітабы", "кітап"), ("еңбегі", "еңбек"), ("тәуелсіздігін", "тәуелсіздік"),
        ]
        for form, lemma in pairs:
            label = prefix_label(form, lemma)
            result = _repair_lemma(form, label, pos="NOUN")
            assert result == lemma, f"{form}: label={label} repair→{result} ≠ {lemma}"


class TestSynthAlternations:
    """The synthetic alternation-pair generator (scripts/synth_alternations.py)."""

    def test_voice_final_consonants(self):
        from scripts.synth_alternations import _voice_final
        assert _voice_final("құқық") == "құқығ"   # қ → ғ
        assert _voice_final("мектеп") == "мектеб"  # п → б
        assert _voice_final("аймақ") == "аймағ"    # қ → ғ
        assert _voice_final("кітап") == "кітаб"    # п → б

    def test_voice_final_returns_none_for_non_voiceless(self):
        from scripts.synth_alternations import _voice_final
        assert _voice_final("адам") is None    # ends in м
        assert _voice_final("бала") is None    # ends in а (vowel)
        assert _voice_final("а") is None       # too short

    def test_harmony_picks_front_or_back_suffix(self):
        from scripts.synth_alternations import generate_pairs
        # мектеп — last vowel 'е' is front → -і suffix
        front = generate_pairs("мектеп", 1)
        assert ("мектебі", "мектеп") in front, f"front harmony wrong: {front}"
        # кітап — last vowel 'а' is back → -ы suffix
        back = generate_pairs("кітап", 1)
        assert ("кітабы", "кітап") in back, f"back harmony wrong: {back}"

    def test_generated_pairs_roundtrip_through_label_and_repair(self):
        """Every synthesized pair must reconstruct correctly through the
        (prefix_label → _repair_lemma) pipeline. Guards the generator
        against harmony/voicing bugs that would feed wrong labels to training."""
        from scripts.synth_alternations import generate_pairs
        from scripts.train_lemma import prefix_label
        from kazmorph.neural_lemma import _repair_lemma
        for lemma in ["құқық", "бастық", "мектеп", "аймақ", "кітап", "билік",
                      "орталық", "еңбек", "тәуелсіздік"]:
            for form, _ in generate_pairs(lemma, 3):
                label = prefix_label(form, lemma)
                result = _repair_lemma(form, label, pos="NOUN")
                assert result == lemma, \
                    f"{form}→{lemma}: label={label} repair→{result}"


class TestLemmaSilverModule:
    """Silver corpus extraction (kazmorph/lemma_silver.py)."""

    def test_collect_from_inline_stream(self, tmp_path):
        from kazmorph.lemma_silver import collect_lemma_pairs, write_tsv
        # write a tiny stream-format file
        stream = tmp_path / "kaz.tagged"
        stream.write_text(
            "^ән/ән<n><nom>$ ^конкурсы/конкурс<n><px3sp><nom>$ "
            "^болады/бол<v><iv><aor><p3><sg>$ ^./.<sent>$\n",
            encoding="utf-8",
        )
        pairs = collect_lemma_pairs(stream_files=[str(stream)])
        assert ("ән", "NOUN") in pairs.values() or pairs.get("ән") == ("ән", "NOUN")
        assert pairs["конкурсы"][0] == "конкурс"
        assert pairs["болады"][0] == "бол"

        n = write_tsv(pairs, str(tmp_path / "out.tsv"))
        assert n == len(pairs)
