"""
Тесты авторизации, сбора данных и личного кабинета.

Используют SQLite во временной БД (фикстура sqlite_db) — быстро, без внешних
зависимостей. Pipeline подменяется лёгким экземпляром (без моделей).

Запуск: pytest tests/test_auth_db.py -v
"""

import pytest
from fastapi.testclient import TestClient

import main
from kazmorph.pipeline import Pipeline


@pytest.fixture
def client(sqlite_db):
    """TestClient с SQLite-БД и лёгким pipeline (без моделей)."""
    test_pipeline = Pipeline(crf_model_path=None, neural_model_path=None)
    main._pipeline = test_pipeline
    # Секрет сессии для подписи cookie.
    main.app.dependency_overrides.clear()
    c = TestClient(main.app)
    yield c
    main._pipeline = None
    main.app.dependency_overrides.clear()


# ============================================================
#  Регистрация
# ============================================================

class TestRegister:
    def test_register_success_redirects_to_dashboard(self, client):
        resp = client.post("/register", data={
            "email": "user@example.com",
            "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        assert resp.status_code == 303
        assert "/dashboard" in resp.headers["location"]

    def test_register_logs_user_in(self, client):
        client.post("/register", data={
            "email": "user@example.com",
            "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        # Сессия установлена — dashboard доступен без отдельного логина.
        assert client.get("/dashboard").status_code == 200

    def test_register_duplicate_email_rejected(self, client):
        data = {"email": "dup@example.com", "password": "password123",
                "password_confirm": "password123"}
        client.post("/register", data=data, follow_redirects=False)
        resp = client.post("/register", data=data, follow_redirects=False)
        assert resp.status_code == 400
        assert "уже зарегистрирован" in resp.text  # переведённое сообщение

    def test_register_invalid_email(self, client):
        resp = client.post("/register", data={
            "email": "not-an-email", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        assert resp.status_code == 400
        assert "Некорректный email" in resp.text

    def test_register_short_password(self, client):
        resp = client.post("/register", data={
            "email": "ok@example.com", "password": "short",
            "password_confirm": "short",
        }, follow_redirects=False)
        assert resp.status_code == 400
        assert "не короче 8" in resp.text

    def test_register_mismatched_passwords(self, client):
        resp = client.post("/register", data={
            "email": "ok@example.com", "password": "password123",
            "password_confirm": "different123",
        }, follow_redirects=False)
        assert resp.status_code == 400
        assert "не совпадают" in resp.text

    def test_password_is_hashed_not_stored_plaintext(self, client):
        from app.db import get_session_factory
        from app.models import User
        client.post("/register", data={
            "email": "hash@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        db = get_session_factory()()
        try:
            user = db.query(User).filter_by(email="hash@example.com").first()
            assert user is not None
            assert user.password_hash != "password123"
            assert "$" in user.password_hash  # bcrypt-формат: $2b$...
        finally:
            db.close()


# ============================================================
#  Вход / выход
# ============================================================

class TestLogin:
    def _register(self, client):
        client.post("/register", data={
            "email": "login@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        client.post("/logout", follow_redirects=False)

    def test_login_success(self, client):
        self._register(client)
        resp = client.post("/login", data={
            "email": "login@example.com", "password": "password123", "next": "/",
        }, follow_redirects=False)
        assert resp.status_code == 303

    def test_login_wrong_password(self, client):
        self._register(client)
        resp = client.post("/login", data={
            "email": "login@example.com", "password": "wrongpassword", "next": "/",
        }, follow_redirects=False)
        assert resp.status_code == 401
        assert "Неверный email или пароль" in resp.text

    def test_login_unknown_email(self, client):
        resp = client.post("/login", data={
            "email": "nobody@example.com", "password": "password123", "next": "/",
        }, follow_redirects=False)
        assert resp.status_code == 401

    def test_logout_clears_session(self, client):
        self._register(client)
        client.post("/login", data={
            "email": "login@example.com", "password": "password123", "next": "/",
        }, follow_redirects=False)
        assert client.get("/dashboard").status_code == 200
        client.post("/logout", follow_redirects=False)
        assert client.get("/dashboard").status_code == 401

    def test_login_open_redirect_blocked(self, client):
        self._register(client)
        resp = client.post("/login", data={
            "email": "login@example.com", "password": "password123",
            "next": "//evil.com",
        }, follow_redirects=False)
        assert resp.headers["location"] == "/"

    def test_login_redirect_to_next_local(self, client):
        self._register(client)
        resp = client.post("/login", data={
            "email": "login@example.com", "password": "password123",
            "next": "/dashboard",
        }, follow_redirects=False)
        assert resp.headers["location"] == "/dashboard"


# ============================================================
#  Сбор данных (анализ сохраняется в БД)
# ============================================================

class TestDataCollection:
    def test_analyze_anonymous_saved_with_null_user(self, client):
        from app.db import get_session_factory
        from app.models import QueryRecord
        resp = client.post("/api/analyze", json={"text": "адам сөз"})
        assert resp.status_code == 200
        qid = resp.json()["query_id"]
        assert qid is not None

        db = get_session_factory()()
        try:
            rec = db.get(QueryRecord, qid)
            assert rec is not None
            assert rec.user_id is None  # анонимный
            assert rec.input_text == "адам сөз"
            assert rec.tokens_count == 2
            assert isinstance(rec.result_json, list)
            assert len(rec.result_json) == 2
        finally:
            db.close()

    def test_analyze_logged_in_saved_with_user(self, client):
        from app.db import get_session_factory
        from app.models import QueryRecord, User
        client.post("/register", data={
            "email": "collector@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        resp = client.post("/api/analyze", json={"text": "мектеп баласы"})
        qid = resp.json()["query_id"]

        db = get_session_factory()()
        try:
            rec = db.get(QueryRecord, qid)
            user = db.query(User).filter_by(email="collector@example.com").first()
            assert rec.user_id == user.id
        finally:
            db.close()

    def test_result_json_stores_full_tokens(self, client):
        """result_json должен содержать полные данные токенов для обучения."""
        from app.db import get_session_factory
        from app.models import QueryRecord
        resp = client.post("/api/analyze", json={"text": "адам"})
        qid = resp.json()["query_id"]
        db = get_session_factory()()
        try:
            rec = db.get(QueryRecord, qid)
            tok = rec.result_json[0]
            # Все ключи контракта пайплайна сохранены.
            for key in ("token", "lemma", "pos", "confidence", "suffixes",
                        "difficulty_class", "suggestions"):
                assert key in tok, f"missing key: {key}"
        finally:
            db.close()


# ============================================================
#  Личный кабинет (dashboard)
# ============================================================

class TestDashboard:
    def test_dashboard_requires_auth(self, client):
        assert client.get("/dashboard").status_code == 401

    def test_dashboard_lists_user_queries(self, client):
        client.post("/register", data={
            "email": "dash@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        client.post("/api/analyze", json={"text": "бірінші мәтін"})
        client.post("/api/analyze", json={"text": "екінші мәтін"})

        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "бірінші мәтін" in resp.text
        assert "екінші мәтін" in resp.text

    def test_query_detail_shows_saved_result(self, client):
        client.post("/register", data={
            "email": "detail@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        resp = client.post("/api/analyze", json={"text": "детальды тексеру"})
        qid = resp.json()["query_id"]

        detail = client.get(f"/dashboard/query/{qid}")
        assert detail.status_code == 200
        assert "детальды тексеру" in detail.text
        assert "SAVED_RESULT" in detail.text

    def test_user_cannot_access_others_query(self, client):
        # Пользователь A создаёт запрос.
        client.post("/register", data={
            "email": "userA@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        resp = client.post("/api/analyze", json={"text": "секрет A"})
        qid_a = resp.json()["query_id"]
        client.post("/logout", follow_redirects=False)

        # Пользователь B регистрируется.
        client.post("/register", data={
            "email": "userB@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)

        # B не должен видеть запрос A.
        assert client.get(f"/dashboard/query/{qid_a}").status_code == 404

    def test_query_detail_nonexistent_returns_404(self, client):
        client.post("/register", data={
            "email": "nf@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        assert client.get("/dashboard/query/99999").status_code == 404

    def test_dashboard_stats_api(self, client):
        client.post("/register", data={
            "email": "stats@example.com", "password": "password123",
            "password_confirm": "password123",
        }, follow_redirects=False)
        client.post("/api/analyze", json={"text": "статистика сөздері"})
        resp = client.get("/api/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] >= 1
        assert data["total_tokens"] >= 1


# ============================================================
#  Безопасность паролей
# ============================================================

class TestPasswordSecurity:
    def test_verify_password_correct(self):
        from app.security import hash_password, verify_password
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_password_wrong(self):
        from app.security import hash_password, verify_password
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_hash_is_unique_per_call(self):
        from app.security import hash_password
        # bcrypt использует соль — два хеша одного пароля различаются.
        assert hash_password("same") != hash_password("same")
