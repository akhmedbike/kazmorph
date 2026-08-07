"""
Тесты scripts/llm_review.py — LLM-ревью кандидатов → Annotation.

Реальные API-вызовы не делаются — LLMReviewer подменяется фейком.
БД изолирована через sqlite_db фикстуру. Фокус на load_candidates,
save_annotation и end-to-end run_review.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import llm_review as lr
from kazmorph.llm_review.client import ReviewResult


# ============================================================
#  Фейковый LLMReviewer (без API-вызовов)
# ============================================================


class FakeReviewer:
    """Имитация LLMReviewer для тестов run_review."""

    def __init__(self, verdict="incorrect", lemma="fixed", pos="VERB"):
        self.verdict = verdict
        self.lemma = lemma
        self.pos = pos
        self.provider = "claude"
        self.model = "fake-model"
        self.stats = SimpleNamespace(
            n_calls=0, n_errors=0, total_cost_usd=0.0,
            total_input_tokens=0, total_output_tokens=0,
            verdicts={"correct": 0, "incorrect": 0, "uncertain": 0},
        )

    def review_token(self, sentence, token, pipeline_lemma, pipeline_pos,
                     difficulty_class=None, confidence=None):
        self.stats.n_calls += 1
        self.stats.total_cost_usd += 0.001
        self.stats.total_input_tokens += 100
        self.stats.total_output_tokens += 20
        self.stats.verdicts[self.verdict] = self.stats.verdicts.get(self.verdict, 0) + 1
        return ReviewResult(
            token=token, verdict=self.verdict,
            corrected_lemma=self.lemma if self.verdict == "incorrect" else None,
            corrected_pos=self.pos if self.verdict == "incorrect" else None,
            explanation="test", provider=self.provider, model=self.model,
            latency_ms=10, cost_usd=0.001, input_tokens=100, output_tokens=20,
        )


# ============================================================
#  load_candidates — чтение JSONL
# ============================================================


class TestLoadCandidates:
    def test_reads_jsonl(self, tmp_path, monkeypatch):
        cands_dir = tmp_path / "candidates"
        cands_dir.mkdir()
        path = cands_dir / "2026-08-04.jsonl"
        lines = [
            {"sentence": "s1", "token": "t1", "lemma": "l1", "pos": "NOUN", "confidence": 0.4},
            {"sentence": "s2", "token": "t2", "lemma": "l2", "pos": "VERB", "confidence": 0.3},
        ]
        path.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
        monkeypatch.setattr(lr, "CANDIDATES_DIR", cands_dir)
        cands = lr.load_candidates("2026-08-04")
        assert len(cands) == 2
        assert cands[0]["token"] == "t1"

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "CANDIDATES_DIR", tmp_path / "nonexistent")
        assert lr.load_candidates("2026-01-01") == []

    def test_skips_invalid_json_lines(self, tmp_path, monkeypatch):
        cands_dir = tmp_path / "candidates"
        cands_dir.mkdir()
        path = cands_dir / "2026-08-04.jsonl"
        path.write_text(
            json.dumps({"token": "ok"}) + "\n{не валидный json}\n" + json.dumps({"token": "ok2"}),
            encoding="utf-8")
        monkeypatch.setattr(lr, "CANDIDATES_DIR", cands_dir)
        cands = lr.load_candidates("2026-08-04")
        assert len(cands) == 2  # повреждённая строка пропущена

    def test_empty_lines_skipped(self, tmp_path, monkeypatch):
        cands_dir = tmp_path / "candidates"
        cands_dir.mkdir()
        path = cands_dir / "2026-08-04.jsonl"
        path.write_text("\n\n" + json.dumps({"token": "x"}) + "\n\n", encoding="utf-8")
        monkeypatch.setattr(lr, "CANDIDATES_DIR", cands_dir)
        assert len(lr.load_candidates("2026-08-04")) == 1


# ============================================================
#  save_annotation — создание записи с правильными полями
# ============================================================


class TestSaveAnnotation:
    def test_creates_pending_annotation(self, sqlite_db):
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            cand = {
                "sentence": "Адам келді .",
                "token": "келді", "lemma": "кел", "pos": "VERB",
                "confidence": 0.4, "difficulty_class": "AFC",
                "source_url": "http://example.com/news/1",
            }
            result = ReviewResult(
                token="келді", verdict="incorrect",
                corrected_lemma="кел", corrected_pos="AUX",
                explanation="AUX here", provider="claude", model="claude-haiku-4-5",
                latency_ms=200, cost_usd=0.002, input_tokens=150, output_tokens=30,
            )
            ann_id = lr.save_annotation(db, cand, result)
            assert ann_id is not None

            from app.models import Annotation
            ann = db.get(Annotation, ann_id)
            assert ann.status == "pending"
            assert ann.token == "келді"
            assert ann.sentence == "Адам келді ."
            assert ann.pipeline_lemma == "кел"
            assert ann.pipeline_pos == "VERB"
            assert ann.pipeline_confidence == 0.4
            assert ann.difficulty_class == "AFC"
            assert ann.llm_verdict == "incorrect"
            assert ann.llm_lemma == "кел"
            assert ann.llm_pos == "AUX"
            assert ann.llm_explanation == "AUX here"
            assert ann.llm_provider == "claude"
            assert ann.llm_cost_usd == 0.002
            assert ann.source_url == "http://example.com/news/1"
            assert ann.final_lemma is None  # не одобрено
            assert ann.reviewer_id is None
        finally:
            db.close()

    def test_correct_verdict_stores_no_correction(self, sqlite_db):
        """correct: llm_lemma/llm_pos = None (LLM считает разбор верным)."""
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            cand = {"sentence": "s", "token": "t", "lemma": "l", "pos": "N",
                    "confidence": 0.5, "difficulty_class": "NONE"}
            result = ReviewResult(
                token="t", verdict="correct", provider="claude", model="m",
                latency_ms=10, cost_usd=0.001, input_tokens=50, output_tokens=10,
            )
            ann_id = lr.save_annotation(db, cand, result)
            from app.models import Annotation
            ann = db.get(Annotation, ann_id)
            assert ann.llm_verdict == "correct"
            assert ann.llm_lemma is None
            assert ann.llm_pos is None
        finally:
            db.close()


# ============================================================
#  run_review — end-to-end с замоканным reviewer
# ============================================================


class TestRunReview:
    def _setup_candidates(self, tmp_path, monkeypatch, n=3):
        cands_dir = tmp_path / "candidates"
        cands_dir.mkdir()
        path = cands_dir / "2026-08-04.jsonl"
        lines = [
            {"sentence": f"s{i}", "token": f"t{i}", "lemma": f"l{i}",
             "pos": "NOUN", "confidence": 0.4, "difficulty_class": "BOR",
             "source_url": f"http://x/{i}"}
            for i in range(n)
        ]
        path.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
        monkeypatch.setattr(lr, "CANDIDATES_DIR", cands_dir)
        monkeypatch.setattr(lr, "RUN_LOG_DIR", tmp_path / "runs")
        return cands_dir

    def test_dry_run_no_db_write(self, tmp_path, monkeypatch):
        self._setup_candidates(tmp_path, monkeypatch, n=2)
        stats = lr.run_review(
            "2026-08-04", dry_run=True, sleep_between=0,
            reviewer=FakeReviewer(),
        )
        assert stats["n_candidates"] == 2
        assert stats["n_reviewed"] == 2
        assert stats["dry_run"] is True
        assert stats["total_cost_usd"] > 0
        assert stats["verdicts"]["incorrect"] == 2

    def test_writes_to_db(self, sqlite_db, tmp_path, monkeypatch):
        self._setup_candidates(tmp_path, monkeypatch, n=3)
        stats = lr.run_review(
            "2026-08-04", dry_run=False, sleep_between=0,
            reviewer=FakeReviewer(),
        )
        assert stats["n_reviewed"] == 3
        # Проверить, что 3 pending-аннотации созданы в БД.
        from app.models import Annotation
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            pending = db.query(Annotation).filter(Annotation.status == "pending").all()
            assert len(pending) == 3
            tokens = {a.token for a in pending}
            assert tokens == {"t0", "t1", "t2"}
        finally:
            db.close()

    def test_max_candidates_limit(self, tmp_path, monkeypatch):
        self._setup_candidates(tmp_path, monkeypatch, n=5)
        stats = lr.run_review(
            "2026-08-04", dry_run=True, sleep_between=0,
            max_candidates=2, reviewer=FakeReviewer(),
        )
        assert stats["n_candidates"] == 2
        assert stats["n_reviewed"] == 2

    def test_empty_candidates_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lr, "CANDIDATES_DIR", tmp_path / "nonexistent")
        stats = lr.run_review("2026-08-04", dry_run=True, reviewer=FakeReviewer())
        assert stats["n_candidates"] == 0
        assert stats["n_reviewed"] == 0

    def test_run_log_written(self, tmp_path, monkeypatch):
        runs_dir = tmp_path / "runs"
        self._setup_candidates(tmp_path, monkeypatch, n=1)
        monkeypatch.setattr(lr, "RUN_LOG_DIR", runs_dir)
        lr.run_review("2026-08-04", dry_run=True, sleep_between=0,
                      reviewer=FakeReviewer())
        log_files = list(runs_dir.glob("*.json"))
        assert len(log_files) == 1
        log = json.loads(log_files[0].read_text(encoding="utf-8"))
        assert log["n_reviewed"] == 1
        assert log["provider"] == "claude"
        assert "total_cost_usd" in log
