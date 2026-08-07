"""
Тесты отбора кандидатов (scripts/select_candidates.py).

Без реального pipeline — используются синтетические списки токенов с
заданными confidence/difficulty_class. Фокус на sentencer и фильтре.
"""

from __future__ import annotations

import json

import pytest

from scripts import select_candidates as sc


def _tok(token: str, *, conf=0.95, diff="NONE", pos="NOUN", lemma=None) -> dict:
    """Быстро собрать токен-словарь в формате Pipeline.process()."""
    return {
        "token": token,
        "lemma": lemma or token.lower(),
        "pos": pos,
        "confidence": conf,
        "difficulty_class": diff,
    }


# ============================================================
#  split_into_sentences (sentencer)
# ============================================================


class TestSplitSentences:
    def test_single_sentence_with_end(self):
        toks = [_tok("Адам"), _tok("келді"), _tok(".", pos="PUNCT")]
        assert sc.split_into_sentences(toks) == [(0, 3)]

    def test_two_sentences(self):
        toks = [
            _tok("Адам"), _tok("келді"), _tok(".", pos="PUNCT"),
            _tok("Ол"), _tok("барады"), _tok(".", pos="PUNCT"),
        ]
        assert sc.split_into_sentences(toks) == [(0, 3), (3, 6)]

    def test_trailing_without_terminator(self):
        # Последнее «предложение» без точки — отдельное окно.
        toks = [_tok("Адам"), _tok("келді"), _tok(".", pos="PUNCT"), _tok("Сөз")]
        assert sc.split_into_sentences(toks) == [(0, 3), (3, 4)]

    def test_empty(self):
        assert sc.split_into_sentences([]) == []

    def test_no_terminators(self):
        toks = [_tok("а"), _tok("б"), _tok("в")]
        assert sc.split_into_sentences(toks) == [(0, 3)]

    def test_question_and_exclamation(self):
        toks = [_tok("Кім"), _tok("?"), _tok("Иә"), _tok("!")]
        assert sc.split_into_sentences(toks) == [(0, 2), (2, 4)]

    def test_closing_quote_as_terminator(self):
        # » — закрывающая кавычка цитаты.
        toks = [_tok("«"), _tok("сөз"), _tok("»"), _tok("басқа")]
        assert sc.split_into_sentences(toks) == [(0, 3), (3, 4)]


class TestSentenceText:
    def test_joins_tokens(self):
        toks = [_tok("А"), _tok("Б"), _tok("В")]
        assert sc.sentence_text(toks, 0, 3) == "А Б В"

    def test_subrange(self):
        toks = [_tok("А"), _tok("Б"), _tok("В"), _tok("Г")]
        assert sc.sentence_text(toks, 1, 3) == "Б В"


class TestFindSentenceOf:
    def test_finds_containing_span(self):
        spans = [(0, 3), (3, 6)]
        assert sc.find_sentence_of([], spans, 1) == (0, 3)
        assert sc.find_sentence_of([], spans, 4) == (3, 6)

    def test_returns_none_if_not_found(self):
        spans = [(0, 3)]
        assert sc.find_sentence_of([], spans, 5) is None


# ============================================================
#  extract_candidates — фильтр low-confidence / hard
# ============================================================


class TestExtractCandidates:
    def test_selects_low_confidence(self):
        toks = [
            _tok("надёжный", conf=0.95),
            _tok("сомнительный", conf=0.40),
        ]
        cands = sc.extract_candidates(toks, threshold=0.70)
        assert len(cands) == 1
        assert cands[0]["token"] == "сомнительный"
        assert cands[0]["confidence"] == 0.40

    def test_selects_hard_difficulty_even_if_high_conf(self):
        # difficulty_class != NONE — кандидат, даже если confidence высокий.
        toks = [
            _tok("заимствование", conf=0.99, diff="BOR"),
            _tok("нормальное", conf=0.95, diff="NONE"),
        ]
        cands = sc.extract_candidates(toks, threshold=0.70)
        assert len(cands) == 1
        assert cands[0]["token"] == "заимствование"
        assert cands[0]["difficulty_class"] == "BOR"

    def test_skips_normal_tokens(self):
        toks = [_tok("а", conf=0.95, diff="NONE"), _tok("б", conf=0.90, diff="NONE")]
        assert sc.extract_candidates(toks, threshold=0.70) == []

    def test_includes_sentence_context(self):
        toks = [
            _tok("Кеше", conf=0.95),
            _tok("оқиға", conf=0.30),  # кандидат
            _tok("болды", conf=0.95),
            _tok(".", pos="PUNCT"),
        ]
        cands = sc.extract_candidates(toks, threshold=0.70)
        assert len(cands) == 1
        assert cands[0]["sentence"] == "Кеше оқиға болды ."
        assert cands[0]["token"] == "оқиға"

    def test_preserves_pos_lemma(self):
        toks = [_tok("сөздер", conf=0.30, pos="NOUN", lemma="сөз")]
        cands = sc.extract_candidates(toks, threshold=0.70)
        assert cands[0]["pos"] == "NOUN"
        assert cands[0]["lemma"] == "сөз"

    def test_source_url_propagated(self):
        toks = [_tok("x", conf=0.30)]
        cands = sc.extract_candidates(toks, threshold=0.70, source_url="http://test/1")
        assert cands[0]["source_url"] == "http://test/1"

    def test_multiple_sentences(self):
        toks = [
            _tok("А", conf=0.30),
            _tok(".", pos="PUNCT"),
            _tok("Б", conf=0.30),
            _tok(".", pos="PUNCT"),
        ]
        cands = sc.extract_candidates(toks, threshold=0.70)
        assert len(cands) == 2
        assert cands[0]["sentence"] == "А ."
        assert cands[1]["sentence"] == "Б ."

    def test_threshold_boundary(self):
        # confidence == threshold НЕ кандидат (строгое <).
        toks = [_tok("x", conf=0.70, diff="NONE")]
        assert sc.extract_candidates(toks, threshold=0.70) == []

    def test_empty_tokens(self):
        assert sc.extract_candidates([], threshold=0.70) == []


# ============================================================
#  load_articles — чтение собранных статей
# ============================================================


class TestLoadArticles:
    def test_reads_json_files(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw_news" / "2026-08-04"
        raw.mkdir(parents=True)
        (raw / "src_abc.json").write_text(
            json.dumps({"text": "привет", "url": "http://x"}), encoding="utf-8")
        (raw / "src_def.json").write_text(
            json.dumps({"text": "мир", "url": "http://y"}), encoding="utf-8")
        # dotfile должен быть пропущен.
        (raw / ".seen.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(sc, "RAW_NEWS_DIR", tmp_path / "raw_news")
        articles = sc.load_articles("2026-08-04")
        assert len(articles) == 2
        texts = [a["text"] for a in articles]
        assert "привет" in texts and "мир" in texts

    def test_missing_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "RAW_NEWS_DIR", tmp_path / "nonexistent")
        assert sc.load_articles("2026-01-01") == []

    def test_skips_corrupted(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw_news" / "2026-08-04"
        raw.mkdir(parents=True)
        (raw / "good.json").write_text(json.dumps({"text": "ок"}), encoding="utf-8")
        (raw / "bad.json").write_text("{не json", encoding="utf-8")
        monkeypatch.setattr(sc, "RAW_NEWS_DIR", tmp_path / "raw_news")
        articles = sc.load_articles("2026-08-04")
        assert len(articles) == 1  # повреждённый пропущен
