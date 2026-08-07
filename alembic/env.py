"""Alembic env: читает DATABASE_URL из окружения, использует модели из app.models."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Добавляем корень проекта в sys.path, чтобы импортировать app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import Base  # noqa: E402
from app import models  # noqa: F401,E402 — регистрирует модели в Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Целевые метаданные для autogenerate.
target_metadata = Base.metadata

# URL из окружения (главный источник — не прописан в ini-файле).
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if DATABASE_URL:
    config.set_main_option("sqlalchemy.url", DATABASE_URL)


def render_item(type_, obj, autogen_context):
    """Рендерить кастомный тип JSONB явно, чтобы миграция была самодостаточной.

    Без этого Alembic генерирует app.models.JSONB() без импорта. Здесь мы
    подставляем платформо-нативный тип напрямую в DDL. obj может быть как
    классом, так и экземпляром TypeDecorator — проверяем оба случая.
    """
    if type_ == "type":
        cls = obj if isinstance(obj, type) else type(obj)
        if cls.__name__ == "JSONB":
            autogen_context.imports.add(
                "from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB"
            )
            autogen_context.imports.add("from sqlalchemy.types import JSON")
            return "PG_JSONB().with_variant(JSON(), 'sqlite')"
    return False  # default rendering for everything else


def run_migrations_offline() -> None:
    """Offline-режим: генерация SQL без подключения к БД."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Online-режим: применение миграций к живой БД."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
