import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def sqlite_db(monkeypatch, tmp_path):
    """Переключить БД-слой на временную SQLite для быстрых изолированных тестов.

    Сбрасывает кэш подключения app.db, создаёт таблицы, отдаёт путь.
    После теста — снова сброс, чтобы следующий тест получил чистую БД.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Сбрасываем кэш состояния app.db, чтобы оно переподключилось к SQLite.
    import app.db as dbmod
    dbmod._engine = None
    dbmod._SessionLocal = None
    dbmod._db_checked = False
    dbmod._db_ok = False
    # DATABASE_URL читается один раз при импорте — обновим явно.
    dbmod.DATABASE_URL = f"sqlite:///{db_path}"
    assert dbmod.init_db(), "SQLite init failed"

    # Создаём таблицы.
    from app.db import Base, get_engine
    from app import models  # noqa: F401 — регистрирует модели
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    yield db_path

    # Teardown: сброс, чтобы другие тесты (без фикстуры) видели «нет БД».
    dbmod._engine = None
    dbmod._SessionLocal = None
    dbmod._db_checked = False
    dbmod._db_ok = False
    dbmod.DATABASE_URL = ""


def make_user(db, *, email: str, password: str = "password123",
              is_admin: bool = False) -> "object":
    """Создать пользователя в БД напрямую (без прохождения через /register).

    Хелпер для тестов admin-функционала: позволяет создать и админа, и
    обычного пользователя с любыми правами.
    """
    from app.models import User
    from app.security import hash_password

    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        is_active=True,
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def admin_user(sqlite_db):
    """Создать пользователя-админа и залогинить его в сессии.

    Возвращает (client, user). client — TestClient с проставленным
    session cookie. Используется тестами /admin/* роутов.
    """
    from fastapi.testclient import TestClient
    import main

    with sqlite_db.parent.joinpath("test.db").open() as _:  # БД уже создана фикстурой
        pass

    from app.db import get_session_factory
    factory = get_session_factory()
    db = factory()
    try:
        user = make_user(db, email="admin@test.kz", is_admin=True)
    finally:
        db.close()

    # TestClient без lifespan — модели не грузятся, что и нужно для быстрых тестов.
    client = TestClient(main.app)
    # Логиним через session cookie напрямую (TestClient поддерживает).
    client.post("/login", data={"email": "admin@test.kz", "password": "password123"})
    yield client, user


@pytest.fixture
def regular_user(sqlite_db):
    """Создать обычного (не-админ) пользователя и залогинить. Для тестов 403."""
    from fastapi.testclient import TestClient
    import main
    from app.db import get_session_factory

    factory = get_session_factory()
    db = factory()
    try:
        user = make_user(db, email="user@test.kz", is_admin=False)
    finally:
        db.close()

    client = TestClient(main.app)
    client.post("/login", data={"email": "user@test.kz", "password": "password123"})
    yield client, user
