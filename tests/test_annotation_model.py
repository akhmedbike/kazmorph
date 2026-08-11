"""
Тесты модели Annotation и require_admin (Active Learning persistence).

Покрывает: создание аннотации, жизненный цикл статусов, поля LLM-вердикта,
роль is_admin и require_admin-зависимость (403 для не-админа, 200 для админа).
"""

from __future__ import annotations

import pytest

from app.auth import require_admin, require_user
from app.models import ANNOTATION_STATUSES, LLM_VERDICTS, Annotation, User


# ============================================================
#  Константы-алфавиты
# ============================================================


class TestConstants:
    def test_annotation_statuses(self):
        assert set(ANNOTATION_STATUSES) == {"pending", "approved", "rejected", "edited"}

    def test_llm_verdicts(self):
        assert set(LLM_VERDICTS) == {"correct", "incorrect", "uncertain"}


# ============================================================
#  Модель Annotation — CRUD и инварианты
# ============================================================


class TestAnnotationModel:
    def test_create_minimal(self, sqlite_db):
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = Annotation(
                sentence="Адам келді .",
                token="келді",
                pipeline_lemma="кел",
                pipeline_pos="VERB",
                pipeline_confidence=0.40,
                difficulty_class="AFC",
                llm_verdict="incorrect",
                llm_lemma="кел",
                llm_pos="AUX",
                llm_explanation="AUX vs VERB",
                llm_provider="claude",
            )
            db.add(ann)
            db.commit()
            db.refresh(ann)
            assert ann.id is not None
            assert ann.status == "pending"  # дефолт
            assert ann.final_lemma is None  # ещё не одобрено
            assert ann.llm_cost_usd == 0.0  # дефолт
        finally:
            db.close()

    def test_lifecycle_pending_to_approved(self, sqlite_db):
        """Жизненный цикл: pending → approved (final = llm_*)."""
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = Annotation(
                sentence="с", token="т", pipeline_lemma="л", pipeline_pos="NOUN",
                pipeline_confidence=0.5, difficulty_class="BOR",
                llm_verdict="incorrect", llm_lemma="исправленная", llm_pos="VERB",
                llm_explanation="", llm_provider="claude",
            )
            db.add(ann)
            db.commit()
            db.refresh(ann)
            assert ann.status == "pending"

            # Одобряем: final = llm-версия.
            ann.status = "approved"
            ann.final_lemma = ann.llm_lemma
            ann.final_pos = ann.llm_pos
            db.commit()
            db.refresh(ann)
            assert ann.status == "approved"
            assert ann.final_lemma == "исправленная"
            assert ann.final_pos == "VERB"
        finally:
            db.close()

    def test_edited_uses_human_override(self, sqlite_db):
        """edited: final = своя версия человека, не LLM."""
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = Annotation(
                sentence="с", token="т", pipeline_lemma="л", pipeline_pos="NOUN",
                pipeline_confidence=0.5, difficulty_class="NONE",
                llm_verdict="incorrect", llm_lemma="llm_версия", llm_pos="VERB",
                llm_explanation="", llm_provider="claude",
            )
            db.add(ann)
            db.commit()
            db.refresh(ann)
            ann.status = "edited"
            ann.final_lemma = "человеческая_версия"
            ann.final_pos = "ADJ"
            db.commit()
            db.refresh(ann)
            assert ann.final_lemma != ann.llm_lemma
            assert ann.final_lemma == "человеческая_версия"
        finally:
            db.close()

    def test_query_pending_for_review(self, sqlite_db):
        """Админка запрашивает pending-аннотации для ревью."""
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            for i in range(5):
                db.add(Annotation(
                    sentence=f"s{i}", token=f"t{i}", pipeline_lemma="l",
                    pipeline_pos="NOUN", pipeline_confidence=0.5, difficulty_class="BOR",
                    llm_verdict="incorrect", llm_lemma="x", llm_pos="V",
                    llm_explanation="", llm_provider="c",
                    status="pending" if i < 3 else "approved",
                ))
            db.commit()
            pending = db.query(Annotation).filter(Annotation.status == "pending").all()
            assert len(pending) == 3
        finally:
            db.close()

    def test_query_for_retrain_excludes_unreviewed(self, sqlite_db):
        """retrain читает ТОЛЬКО approved/edited, никогда pending/rejected."""
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            for status in ["pending", "approved", "rejected", "edited"]:
                db.add(Annotation(
                    sentence="s", token="t", pipeline_lemma="l", pipeline_pos="N",
                    pipeline_confidence=0.5, difficulty_class="NONE",
                    llm_verdict="incorrect", llm_lemma="x", llm_pos="V",
                    llm_explanation="", llm_provider="c", status=status,
                ))
            db.commit()
            trainable = db.query(Annotation).filter(
                Annotation.status.in_(["approved", "edited"])
            ).all()
            assert len(trainable) == 2
            statuses = {a.status for a in trainable}
            assert statuses == {"approved", "edited"}
        finally:
            db.close()

    def test_nullable_fields_default_none(self, sqlite_db):
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = Annotation(
                sentence="s", token="t", pipeline_lemma="l", pipeline_pos="N",
                pipeline_confidence=0.5, difficulty_class="NONE",
                llm_verdict="uncertain", llm_explanation="", llm_provider="c",
            )
            db.add(ann)
            db.commit()
            db.refresh(ann)
            assert ann.llm_lemma is None
            assert ann.llm_pos is None
            assert ann.final_lemma is None
            assert ann.final_pos is None
            assert ann.reviewer_id is None
            assert ann.reviewed_at is None
            assert ann.source_query_id is None
        finally:
            db.close()


# ============================================================
#  User.is_admin — поле
# ============================================================


class TestIsAdminField:
    def test_default_is_false(self, sqlite_db):
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            user = make_user(db, email="normal@test.kz", is_admin=False)
            assert user.is_admin is False
        finally:
            db.close()

    def test_admin_flag_true(self, sqlite_db):
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            admin = make_user(db, email="boss@test.kz", is_admin=True)
            assert admin.is_admin is True
        finally:
            db.close()


# ============================================================
#  require_user / require_admin — зависимости FastAPI
# ============================================================
#
# После разделения браузерного (Accept: text/html) и API-флоу сигнатуры
# зависимостей приняли ``request: Request`` и поведение разветвилось:
#
#   • API (нет text/html или /api/*) → HTTPException с JSON;
#   • браузер (Accept: text/html)    → BrowserAuthResponse, несущий готовый
#     RedirectResponse (302/303 на логин) или TemplateResponse (403).
#
# Полный end-to-end путь (через TestClient) покрыт в test_admin_routes.py;
# здесь — прямые unit-вызовы на фейковом Starlette-запросе, чтобы проверить
# логику каждой ветви в изоляции.


def _make_request(path: str = "/admin", *, accept: str | None = "text/html",
                  query: str = "") -> "Request":
    """Собрать Starlette Request из минимального scope для прямого вызова.

    ``accept=None`` → заголовок не добавляется (эмулирует AJAX/API-клиент).
    Сессия пустая — аноним; require_user/require_admin её не читают сами
    (пользователь приходит параметром), так что для этих тестов её содержимое
    не важно.
    """
    from starlette.requests import Request
    headers: list = []
    if accept is not None:
        headers.append((b"accept", accept.encode()))
    scope = {
        "type": "http", "method": "GET",
        "path": path, "raw_path": path.encode(),
        "query_string": query.encode(),
        "headers": headers,
        "session": {},
    }
    return Request(scope)


class TestRequireUser:
    def test_authenticated_returns_user(self, sqlite_db):
        """Залогиненный пользователь проходит как для API, так и для браузера."""
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            user = make_user(db, email="u@t.kz")
            assert require_user(_make_request(), user=user) is user
            assert require_user(_make_request(accept="application/json"), user=user) is user
        finally:
            db.close()

    def test_anonymous_api_raises_401(self):
        """API-клиент (без text/html) получает JSON 401."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            require_user(_make_request(accept="application/json"), user=None)
        assert exc.value.status_code == 401

    def test_anonymous_browser_redirects_to_login(self):
        """Браузер (Accept: text/html) получает redirect на /login?next=."""
        from app.auth import BrowserAuthResponse
        with pytest.raises(BrowserAuthResponse) as exc:
            require_user(_make_request(path="/admin"), user=None)
        resp = exc.value.response
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/login?next=")
        assert "/admin" in resp.headers["location"]

    def test_anonymous_browser_redirect_carries_query_string(self):
        """Оригинальный query сохраняется в next (lossless round-trip)."""
        from app.auth import BrowserAuthResponse
        with pytest.raises(BrowserAuthResponse) as exc:
            require_user(
                _make_request(path="/admin/review", query="page=2&status=pending"),
                user=None,
            )
        location = exc.value.response.headers["location"]
        assert "page=2" in location
        assert "status=pending" in location

    def test_anonymous_api_path_stays_json_even_with_html_accept(self):
        """/api/* никогда не уходит в HTML/redirect — JS ждёт JSON."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            # Даже с Accept: text/html путь /api/ форсирует JSON-ветвь.
            require_user(_make_request(path="/api/low_confidence"), user=None)
        assert exc.value.status_code == 401


class TestRequireAdmin:
    def test_admin_passes(self, sqlite_db):
        """Админ проходит require_admin без исключения (обе ветви Accept)."""
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            admin = make_user(db, email="a@t.kz", is_admin=True)
            assert require_admin(_make_request(), user=admin) is admin
            assert require_admin(_make_request(accept="application/json"), user=admin) is admin
        finally:
            db.close()

    def test_non_admin_api_raises_403(self, sqlite_db):
        from fastapi import HTTPException
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            regular = make_user(db, email="r@t.kz", is_admin=False)
            with pytest.raises(HTTPException) as exc:
                require_admin(_make_request(accept="application/json"), user=regular)
            assert exc.value.status_code == 403
        finally:
            db.close()

    def test_non_admin_browser_gets_html_403(self, sqlite_db):
        from app.auth import BrowserAuthResponse
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            regular = make_user(db, email="r@t.kz", is_admin=False)
            with pytest.raises(BrowserAuthResponse) as exc:
                require_admin(_make_request(), user=regular)
            resp = exc.value.response
            assert resp.status_code == 403
            assert resp.media_type == "text/html"
            assert "Недостаточно прав" in resp.body.decode()
        finally:
            db.close()
