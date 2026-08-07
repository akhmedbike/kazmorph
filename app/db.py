"""
Движок базы данных и сессии SQLAlchemy.

DATABASE_URL читается из окружения. По умолчанию — PostgreSQL в Docker
(порт 5433 для локальной разработки, см. docker-compose.yml).
Если переменная не задана ИЛИ подключение не удаётся — БД-слой отключается
(graceful degradation), приложение работает без сохранения данных.

Функция db_available() — единая точка проверки: «есть ли живая БД?».
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""


# DATABASE_URL: postgresql://user:pass@host:port/dbname
# Для тестов/разработки без PostgreSQL можно задать sqlite:///./kazmorph.db
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Флаг: инициализирован ли движок (не задан URL → БД-слой отключен).
_engine = None
_SessionLocal: sessionmaker | None = None
_db_checked = False
_db_ok = False


def _create_engine():
    """Создать движок и проверить подключение. Возвращает True при успехе."""
    global _engine, _SessionLocal, _db_checked, _db_ok
    if not DATABASE_URL:
        logger.info("DATABASE_URL не задана — БД-слой отключён (stateless-режим).")
        _db_checked = True
        _db_ok = False
        return False

    # SQLite (тесты) не требует pool_pre_ping и дополнительных аргументов.
    is_sqlite = DATABASE_URL.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    try:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,          # переподключение при упавшем соединении
            future=True,
            connect_args=connect_args,
        )

        # SQLite не знает функции now() — а модели используют
        # server_default=func.now() (PostgreSQL-стиль). Эмулируем now() и
        # CURRENT_TIMESTAMP на SQLite, чтобы ORM-вставки работали без
        # поднятия PostgreSQL. На PostgreSQL этот listener не активен.
        if is_sqlite:
            @event.listens_for(_engine, "connect")
            def _register_sqlite_funcs(dbapi_conn, _connection_record):
                def _now(_ignored=None):
                    return datetime.now(timezone.utc).isoformat()
                def _current_timestamp(_ignored=None):
                    return datetime.now(timezone.utc).isoformat()
                dbapi_conn.create_function("now", 0, _now)
                dbapi_conn.create_function("current_timestamp", 0, _current_timestamp)
        # Проверяем реальное подключение.
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        _db_checked = True
        _db_ok = True
        logger.info("БД подключена: %s", _mask_url(DATABASE_URL))
        return True
    except Exception as exc:  # noqa: BLE001 — БД опциональна
        logger.warning("Не удалось подключиться к БД (%s): %s — работаем без неё.",
                       _mask_url(DATABASE_URL), exc)
        _engine = None
        _SessionLocal = None
        _db_checked = True
        _db_ok = False
        return False


def _mask_url(url: str) -> str:
    """Спрятать пароль в URL для логов."""
    if "@" in url and "://" in url:
        scheme_rest = url.split("://", 1)
        creds_host = scheme_rest[1].split("@", 1)
        if len(creds_host) == 2:
            user = creds_host[0].split(":", 1)[0]
            return f"{scheme_rest[0]}://{user}:***@{creds_host[1]}"
    return url


def init_db() -> bool:
    """Инициализировать движок (вызывается при старте приложения)."""
    if not _db_checked:
        return _create_engine()
    return _db_ok


def db_available() -> bool:
    """Доступна ли БД прямо сейчас. Лениво инициализирует при первом вызове."""
    if not _db_checked:
        _create_engine()
    return _db_ok


def get_engine():
    """Вернуть движок или None, если БД недоступна."""
    if not _db_checked:
        _create_engine()
    return _engine


def get_session_factory() -> sessionmaker | None:
    """Вернуть фабрику сессий или None."""
    if not _db_checked:
        _create_engine()
    return _SessionLocal
