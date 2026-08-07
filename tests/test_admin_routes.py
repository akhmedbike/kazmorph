"""
Тесты admin-роутов (app/routes/admin.py).

Проверяют авторизацию (401 аноним, 403 не-админ, 200 админ), рендер
страниц, POST decision (approve/reject/edit), JSON-эндпоинты
(/api/low_confidence, /api/admin/stats).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from tests.conftest import make_user


@pytest.fixture
def client_setup(sqlite_db):
    """TestClient + готовая БД (через sqlite_db фикстуру).

    Возвращает фабрику для создания пользователей и helper для логина.
    """
    from app.db import get_session_factory

    factory = get_session_factory()
    client = TestClient(main.app)

    def make_and_login(email: str, is_admin: bool = False):
        db = factory()
        try:
            user = make_user(db, email=email, is_admin=is_admin)
        finally:
            db.close()
        client.post("/login", data={"email": email, "password": "password123"})
        return user

    return client, make_and_login


def _create_annotation(db, status="pending", verdict="incorrect",
                       llm_lemma="fixed", llm_pos="VERB"):
    """Создать тестовую аннотацию."""
    from app.models import Annotation
    ann = Annotation(
        sentence="Адам келді .", token="келді",
        pipeline_lemma="кел", pipeline_pos="NOUN",
        pipeline_confidence=0.4, difficulty_class="AFC",
        llm_verdict=verdict, llm_lemma=llm_lemma if verdict == "incorrect" else None,
        llm_pos=llm_pos if verdict == "incorrect" else None,
        llm_explanation="test", llm_provider="claude",
        status=status,
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return ann


# ============================================================
#  Авторизация
# ============================================================


class TestAdminAuth:
    def test_anonymous_redirected_to_login(self, client_setup):
        client, _ = client_setup
        # require_admin → require_user → 401 для анонима.
        resp = client.get("/admin", follow_redirects=False)
        assert resp.status_code == 401

    def test_non_admin_forbidden(self, client_setup):
        client, mk = client_setup
        mk("user@test.kz", is_admin=False)
        resp = client.get("/admin")
        assert resp.status_code == 403

    def test_admin_allowed(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "Active Learning" in resp.text

    def test_review_requires_admin(self, client_setup):
        client, mk = client_setup
        mk("user@test.kz", is_admin=False)
        assert client.get("/admin/review").status_code == 403

    def test_low_confidence_api_requires_admin(self, client_setup):
        client, mk = client_setup
        mk("user@test.kz", is_admin=False)
        assert client.get("/api/low_confidence").status_code == 403

    # ── Browser (Accept: text/html) behaviour ───────────────────────────
    # The dependency above returns JSON 401/403 for clients that don't ask
    # for HTML (the TestClient default, AJAX). A real browser navigation
    # sends Accept: text/html and must get a friendly redirect/login or an
    # HTML 403 page — never a raw JSON {"detail": ...} body.

    def test_anonymous_browser_redirected_to_login_with_next(self, client_setup):
        client, _ = client_setup
        resp = client.get(
            "/admin", follow_redirects=False, headers={"Accept": "text/html"}
        )
        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/login?next=")
        # next carries the original path so login can return to it.
        assert "/admin" in resp.headers["location"]

    def test_anonymous_browser_next_includes_query_string(self, client_setup):
        client, _ = client_setup
        resp = client.get(
            "/admin/review?page=2&status=pending",
            follow_redirects=False, headers={"Accept": "text/html"},
        )
        assert resp.status_code == 303
        # Query must survive into the next param so the round-trip is lossless.
        assert "page=2" in resp.headers["location"]
        assert "status=pending" in resp.headers["location"]

    def test_non_admin_browser_gets_html_403(self, client_setup):
        client, mk = client_setup
        mk("user@test.kz", is_admin=False)
        resp = client.get("/admin", headers={"Accept": "text/html"})
        assert resp.status_code == 403
        assert "text/html" in resp.headers["content-type"]
        # Rendered page, not JSON: the forbidden card has a clear message.
        assert "Недостаточно прав" in resp.text

    def test_api_endpoint_stays_json_even_with_html_accept(self, client_setup):
        client, _ = client_setup
        # /api/* must never redirect/render HTML — JS consumers expect JSON
        # regardless of the Accept header. This guards the /api/ short-circuit
        # in _wants_html.
        resp = client.get(
            "/api/low_confidence", headers={"Accept": "text/html"}
        )
        assert resp.status_code == 401
        assert resp.headers["content-type"] == "application/json"
        assert "Требуется вход в систему" in resp.text


# ============================================================
#  Рендер страниц
# ============================================================


class TestAdminPages:
    def test_dashboard_shows_counts(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        # Создать несколько аннотаций.
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            _create_annotation(db, status="pending")
            _create_annotation(db, status="pending")
            _create_annotation(db, status="approved")
        finally:
            db.close()

        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "На ревью" in resp.text or "pending" in resp.text.lower()

    def test_review_list_empty(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        resp = client.get("/admin/review")
        assert resp.status_code == 200
        # Нет кандидатов → подсказка.
        assert "Нет кандидатов" in resp.text or "harvest_sources" in resp.text

    def test_review_list_shows_annotation(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = _create_annotation(db, status="pending")
        finally:
            db.close()

        resp = client.get("/admin/review")
        assert resp.status_code == 200
        assert ann.token in resp.text
        assert "incorrect" in resp.text  # вердикт-бейдж

    def test_review_filter_by_status(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            _create_annotation(db, status="pending")
            _create_annotation(db, status="approved")
        finally:
            db.close()

        resp_pending = client.get("/admin/review?status=pending")
        resp_approved = client.get("/admin/review?status=approved")
        # На странице pending — есть pending-аннотация, approved — её нет.
        assert resp_pending.status_code == 200
        assert resp_approved.status_code == 200


# ============================================================
#  POST decision — approve/reject/edit
# ============================================================


class TestAdminDecision:
    def test_approve_sets_final_from_llm(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        from app.models import Annotation
        factory = get_session_factory()
        db = factory()
        try:
            ann = _create_annotation(db, status="pending", llm_lemma="исправлено", llm_pos="AUX")
            ann_id = ann.id
        finally:
            db.close()

        resp = client.post(
            f"/admin/review/{ann_id}/decision",
            json={"decision": "approved"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["status"] == "approved"
        assert data["final_lemma"] == "исправлено"
        assert data["final_pos"] == "AUX"

        db = factory()
        try:
            ann = db.get(Annotation, ann_id)
            assert ann.status == "approved"
            assert ann.final_lemma == "исправлено"
            assert ann.reviewer_id is not None
            assert ann.reviewed_at is not None
        finally:
            db.close()

    def test_reject_clears_final(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        from app.models import Annotation
        factory = get_session_factory()
        db = factory()
        try:
            ann = _create_annotation(db, status="pending")
            ann_id = ann.id
        finally:
            db.close()

        resp = client.post(
            f"/admin/review/{ann_id}/decision",
            json={"decision": "rejected"},
        )
        assert resp.status_code == 200
        db = factory()
        try:
            ann = db.get(Annotation, ann_id)
            assert ann.status == "rejected"
            assert ann.final_lemma is None
            assert ann.final_pos is None
        finally:
            db.close()

    def test_edit_uses_human_override(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        from app.models import Annotation
        factory = get_session_factory()
        db = factory()
        try:
            ann = _create_annotation(db, status="pending", llm_lemma="llm_version")
            ann_id = ann.id
        finally:
            db.close()

        resp = client.post(
            f"/admin/review/{ann_id}/decision",
            json={"decision": "edited", "final_lemma": "моя_версия", "final_pos": "adj"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_lemma"] == "моя_версия"
        assert data["final_pos"] == "ADJ"  # нормализуется к верхнему регистру

        db = factory()
        try:
            ann = db.get(Annotation, ann_id)
            assert ann.status == "edited"
            assert ann.final_lemma == "моя_версия"
            assert ann.final_lemma != ann.llm_lemma  # override, не LLM
        finally:
            db.close()

    def test_edit_requires_final_fields(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = _create_annotation(db, status="pending")
            ann_id = ann.id
        finally:
            db.close()

        # edited без final_lemma → 422.
        resp = client.post(
            f"/admin/review/{ann_id}/decision",
            json={"decision": "edited"},
        )
        assert resp.status_code == 422

    def test_invalid_decision_rejected(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            ann = _create_annotation(db, status="pending")
            ann_id = ann.id
        finally:
            db.close()

        resp = client.post(
            f"/admin/review/{ann_id}/decision",
            json={"decision": "maybe"},
        )
        assert resp.status_code == 422

    def test_non_admin_cannot_decide(self, client_setup):
        client, mk = client_setup
        mk("user@test.kz", is_admin=False)
        resp = client.post(
            "/admin/review/1/decision",
            json={"decision": "approved"},
        )
        assert resp.status_code == 403


# ============================================================
#  JSON API
# ============================================================


class TestAdminJsonApi:
    def test_low_confidence_returns_pending(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            _create_annotation(db, status="pending")
            _create_annotation(db, status="pending")
            _create_annotation(db, status="approved")  # не должна попасть
        finally:
            db.close()

        resp = client.get("/api/low_confidence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["candidates"]) == 2
        assert all(c["status"] == "pending" for c in data["candidates"])
        assert "threshold" in data  # ACTIVE_LEARNING_THRESHOLD

    def test_admin_stats(self, client_setup):
        client, mk = client_setup
        mk("admin@test.kz", is_admin=True)
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            _create_annotation(db, status="pending")
            _create_annotation(db, status="approved")
        finally:
            db.close()

        resp = client.get("/api/admin/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status_counts"]["pending"] == 1
        assert data["status_counts"]["approved"] == 1
        assert data["total"] == 2
        # reviewed=1 (approved), agreement=100%.
        assert data["llm_agreement"]["n_reviewed"] == 1
        assert data["llm_agreement"]["agreement_rate"] == 1.0
