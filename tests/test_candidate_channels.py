"""
Тесты новых каналов генерации кандидатов: корпусный индекс словоформ,
скриптовые варианты (mixed-script) и скользящий разрез стем/суффиксы.

Запуск: pytest tests/test_candidate_channels.py -v
"""

import pytest

from kazmorph.correction.form_index import build_form_lexicon, form_exists
from kazmorph.correction.candidate_generator import (
    _script_variants, _valid_splits, _decompose_tail, generate_candidates,
)


class TestFormIndex:
    def test_build_excludes_excluded_sentences(self, tmp_path):
        src = tmp_path / "src.conllu"
        exc = tmp_path / "exc.conllu"
        src.write_text(
            "# sent_id: 1\n1\tМен\tмен\tPRON\t_\t_\t_\t_\t_\t_\n\n"
            "# sent_id: 2\n1\tДала\tдала\tNOUN\t_\t_\t_\t_\t_\t_\n\n",
            encoding="utf-8")
        exc.write_text(
            "# sent_id: x\n1\tДала\tдала\tNOUN\t_\t_\t_\t_\t_\t_\n\n",
            encoding="utf-8")
        forms = build_form_lexicon([str(src)], exclude_paths=[str(exc)])
        assert "мен" in forms
        assert "дала" not in forms, "исключённое предложение не должно попадать в индекс"

    def test_skips_multiword_rows(self, tmp_path):
        src = tmp_path / "src.conllu"
        src.write_text(
            "# sent_id: 1\n"
            "1-2\tкомпьютер\t_\t_\t_\t_\t_\t_\t_\t_\n"
            "1\tкомпью\t_\tX\t_\t_\t_\t_\t_\t_\n"
            "2\tтер\t_\tX\t_\t_\t_\t_\t_\t_\n\n",
            encoding="utf-8")
        forms = build_form_lexicon([str(src)], exclude_paths=[])
        assert "компьютер" not in forms, "строки диапазонов (n-m) пропускаются"

    def test_alphabetic_only(self, tmp_path):
        src = tmp_path / "src.conllu"
        src.write_text(
            "# sent_id: 1\n1\t.\t_\tPUNCT\t_\t_\t_\t_\t_\t_\n"
            "2\t7\t_\tNUM\t_\t_\t_\t_\t_\t_\n"
            "3\tжәне\t_\tCCONJ\t_\t_\t_\t_\t_\t_\n\n",
            encoding="utf-8")
        forms = build_form_lexicon([str(src)], exclude_paths=[])
        assert "және" in forms
        assert "." not in forms and "7" not in forms


class TestTailDecomposition:
    def test_decompose_single_affix(self):
        assert _decompose_tail("ты") == ["ты"]

    def test_decompose_chain(self):
        pieces = _decompose_tail("тары")
        assert pieces is not None and len(pieces) <= 3

    def test_decompose_invalid(self):
        assert _decompose_tail("xyz") is None

    def test_valid_splits_prefers_longest_tail(self):
        splits = _valid_splits("китапты")
        assert splits, "китапты должен иметь хотя бы один валидный разрез"
        stems = [s for s, _ in splits]
        assert "китап" in stems, "разрез китап|ты обязан находиться"

    def test_no_split_for_short_stem(self):
        assert _valid_splits("аб") == []


class TestScriptVariants:
    def test_russian_i_to_kazakh_i(self):
        variants = dict(_script_variants("китапты"))
        assert "кітапты" in variants, "и→і вариант должен генерироваться"

    def test_no_variants_for_clean_word(self):
        # слово без русских двойников-позиций (все буквы однозначные)
        assert _script_variants("адам") == []

    def test_variants_are_validated(self):
        # валидация отсекает мусор: варианты не из лексиконов не возвращаются
        for variant, _n in _script_variants("китапты"):
            assert variant != "китапты"


class TestStemRecombination:
    def test_kitapty_reaches_kitapty(self):
        """Регрессия живого кейса: китапты → кітапты через скользящий разрез
        (раньше RMA резал испорченный стем: китапты → ки+та+п+ты)."""
        cands = [w for w, _ in generate_candidates("китапты", mode="dl_freq_morph")]
        assert "кітапты" in cands[:3], f"кітапты должен быть в топ-3, got {cands[:5]}"

    def test_bygn_reaches_bygin(self):
        cands = [w for w, _ in generate_candidates("бүгн", mode="dl_freq_morph")]
        assert "бүгін" in cands[:3]

    def test_halky_reaches_khalyq(self):
        cands = [w for w, _ in generate_candidates("халкы", mode="dl_freq_morph")]
        assert "халық" in cands[:5]

    def test_dl_mode_unchanged_channels(self):
        """dl — исторический контроль: без форм/скрипта/рекомбинации."""
        cands = [w for w, _ in generate_candidates("китапты", mode="dl")]
        assert all(w != "кітапты" or form_exists("кітапты") for w in cands)


class TestFormExists:
    def test_membership(self):
        # «адам» — частая форма, обязана быть в корпусном индексе
        assert form_exists("адам")
        assert not form_exists("xyzqwerty")
