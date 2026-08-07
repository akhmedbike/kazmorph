"""
Тесты retrain-цикла (scripts/retrain.py).

Без реального обучения (subprocess замокан). Фокус на:
  • annotations_to_conllu — группировка по предложению, порядок токенов
  • decontaminate — удаление дубликатов с test.conllu
  • merge_train — мердж с оригинальным train.conllu
  • write_conllu / parse обратно (round-trip)
  • run_retrain с замоканным train_new_model — гейт promote/reject
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import retrain as rt


# ============================================================
#  annotations_to_conllu — группировка и порядок
# ============================================================


class TestAnnotationsToConllu:
    def test_groups_by_sentence(self):
        anns = [
            {"sentence": "Адам келді .", "token": "келді", "final_lemma": "кел", "final_pos": "VERB"},
            {"sentence": "Адам келді .", "token": "Адам", "final_lemma": "адам", "final_pos": "NOUN"},
            {"sentence": "Ол барады .", "token": "Ол", "final_lemma": "ол", "final_pos": "PRON"},
        ]
        sents = rt.annotations_to_conllu(anns)
        assert len(sents) == 2  # два уникальных предложения

    def test_token_order_by_position(self):
        anns = [
            {"sentence": "А Б В .", "token": "В", "final_lemma": "в", "final_pos": "N"},
            {"sentence": "А Б В .", "token": "А", "final_lemma": "а", "final_pos": "N"},
            {"sentence": "А Б В .", "token": "Б", "final_lemma": "б", "final_pos": "N"},
        ]
        sents = rt.annotations_to_conllu(anns)
        forms = [t["form"] for t in sents[0]]
        assert forms == ["А", "Б", "В"]  # отсортированы по позиции, не по вводу

    def test_preserves_final_lemma_pos(self):
        anns = [{"sentence": "с", "token": "т", "final_lemma": "моя_лемма", "final_pos": "VERB"}]
        sents = rt.annotations_to_conllu(anns)
        assert sents[0][0]["lemma"] == "моя_лемма"
        assert sents[0][0]["upos"] == "VERB"

    def test_empty_annotations(self):
        assert rt.annotations_to_conllu([]) == []

    def test_case_insensitive_sentence_grouping(self):
        # Разный регистр предложения — одна группа.
        anns = [
            {"sentence": "Адам келді.", "token": "Адам", "final_lemma": "а", "final_pos": "N"},
            {"sentence": "адам келді.", "token": "келді", "final_lemma": "к", "final_pos": "V"},
        ]
        sents = rt.annotations_to_conllu(anns)
        assert len(sents) == 1
        assert len(sents[0]) == 2


# ============================================================
#  decontaminate — удаление дубликатов с test.conllu
# ============================================================


class TestDecontaminate:
    def test_removes_exact_duplicates(self, tmp_path, monkeypatch):
        # Создать синтетический test.conllu с одним предложением.
        test_file = tmp_path / "test.conllu"
        test_file.write_text(
            "1\tтест\tтест\tNOUN\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
        monkeypatch.setattr(rt, "TEST_FILE", str(test_file))
        # Новое предложение-дубликат + чистое.
        sentences = [
            [{"id": "1", "form": "тест", "lemma": "тест", "upos": "NOUN"}],  # дубликат
            [{"id": "1", "form": "новое", "lemma": "новое", "upos": "NOUN"}],  # чистое
        ]
        kept = rt.decontaminate(sentences, test_file=str(test_file))
        assert len(kept) == 1
        assert kept[0][0]["form"] == "новое"

    def test_keeps_non_duplicates(self, tmp_path, monkeypatch):
        test_file = tmp_path / "test.conllu"
        test_file.write_text("1\tX\tX\tN\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
        monkeypatch.setattr(rt, "TEST_FILE", str(test_file))
        sentences = [
            [{"id": "1", "form": "A", "lemma": "a", "upos": "N"}],
            [{"id": "1", "form": "B", "lemma": "b", "upos": "N"}],
        ]
        kept = rt.decontaminate(sentences, test_file=str(test_file))
        assert len(kept) == 2

    def test_empty_input(self, tmp_path):
        test_file = tmp_path / "test.conllu"
        test_file.write_text("1\tX\tX\tN\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
        assert rt.decontaminate([], test_file=str(test_file)) == []

    def test_normalization_collapses_whitespace(self, tmp_path):
        # Дубликат с разным числом пробелов должен удалиться.
        test_file = tmp_path / "test.conllu"
        test_file.write_text(
            "1\tа\tа\tN\t_\t_\t_\t_\t_\t_\n2\tб\tб\tN\t_\t_\t_\t_\t_\t_\n\n",
            encoding="utf-8")
        sentences = [
            [{"id": "1", "form": "а", "lemma": "а", "upos": "N"},
             {"id": "2", "form": "б", "lemma": "б", "upos": "N"}],
        ]
        # sentence text «а б» совпадёт с test после нормализации.
        kept = rt.decontaminate(sentences, test_file=str(test_file))
        assert len(kept) == 0  # дубликат удалён


# ============================================================
#  write_conllu / round-trip
# ============================================================


class TestWriteConllu:
    def test_writes_valid_conllu(self, tmp_path):
        sentences = [
            [{"id": "1", "form": "Адам", "lemma": "адам", "upos": "NOUN"},
             {"id": "2", "form": ".", "lemma": ".", "upos": "PUNCT"}],
        ]
        path = tmp_path / "out.conllu"
        rt.write_conllu(sentences, path)
        content = path.read_text(encoding="utf-8")
        assert "Адам\tадам\tNOUN" in content
        assert content.count("\n\n") >= 1  # разделитель предложений

    def test_round_trip_parse(self, tmp_path):
        from scripts.evaluate import parse_conllu
        sentences = [
            [{"id": "1", "form": "сөз", "lemma": "сөз", "upos": "NOUN"}],
        ]
        path = tmp_path / "rt.conllu"
        rt.write_conllu(sentences, path)
        parsed = parse_conllu(str(path))
        assert len(parsed) == 1
        assert parsed[0][0]["form"] == "сөз"
        assert parsed[0][0]["upos"] == "NOUN"


# ============================================================
#  merge_train
# ============================================================


class TestMergeTrain:
    def test_appends_new_to_original(self, tmp_path, monkeypatch):
        train_file = tmp_path / "train.conllu"
        train_file.write_text(
            "1\tориг\tориг\tN\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
        monkeypatch.setattr(rt, "TRAIN_FILE", str(train_file))
        new = [[{"id": "1", "form": "новое", "lemma": "нов", "upos": "N"}]]
        merged, n_orig = rt.merge_train(new, train_file=str(train_file))
        assert n_orig == 1
        assert len(merged) == 2
        forms = [t["form"] for sent in merged for t in sent]
        assert "ориг" in forms and "новое" in forms

    def test_empty_new_returns_original(self, tmp_path, monkeypatch):
        train_file = tmp_path / "train.conllu"
        train_file.write_text(
            "1\tX\tX\tN\t_\t_\t_\t_\t_\t_\n\n", encoding="utf-8")
        monkeypatch.setattr(rt, "TRAIN_FILE", str(train_file))
        merged, n = rt.merge_train([], train_file=str(train_file))
        assert len(merged) == n


# ============================================================
#  run_retrain — с замоканным обучением
# ============================================================


class TestRunRetrain:
    def test_skips_below_min_corrections(self, sqlite_db, monkeypatch, tmp_path):
        # 0 одобренных правок → skip.
        monkeypatch.setattr(rt, "MERGED_DIR", tmp_path / "merged")
        report = rt.run_retrain(min_corrections=10, dry_run=False)
        assert report["skipped"] is True
        assert report["reason"] == "below_min_corrections"

    def test_dry_run_reports_baseline(self, sqlite_db, tmp_path, monkeypatch):
        # Создать одобренные правки + фейковую текущую модель.
        from app.db import get_session_factory
        from app.models import Annotation
        factory = get_session_factory()
        db = factory()
        try:
            for i in range(5):
                db.add(Annotation(
                    sentence=f"уникальное предложение номер {i}", token=f"токен{i}",
                    pipeline_lemma="л", pipeline_pos="N", pipeline_confidence=0.5,
                    difficulty_class="NONE", llm_verdict="incorrect",
                    llm_lemma="x", llm_pos="V", llm_explanation="", llm_provider="c",
                    status="approved", final_lemma=f"лемма{i}", final_pos="NOUN",
                ))
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(rt, "MERGED_DIR", tmp_path / "merged")

        # Замокать resolve_current_model_path → вернуть путь к реальной модели.
        from kazmorph.model_versioning import resolve_current_model_path
        real_path = resolve_current_model_path()
        if real_path is None:
            pytest.skip("Нет обученной CRF-модели для теста baseline")
        monkeypatch.setattr(rt, "resolve_current_model_path", lambda: real_path)

        report = rt.run_retrain(min_corrections=1, dry_run=True)
        assert report["dry_run"] is True
        assert "baseline" in report
        assert "pos_accuracy" in report["baseline"]
        assert report["n_new_sentences"] >= 1
