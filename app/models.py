"""
ORM-модели: пользователи, сохранённые запросы, аннотации Active Learning.

Схема данных:

  User
    id, email (unique), password_hash, is_active, is_admin, created_at

  QueryRecord
    id, user_id (FK→User, nullable — анонимные запросы), input_text,
    result_json (JSONB — полный ответ пайплайна), tokens_count, created_at

  Annotation
    Low-confidence токен + предложение LLM-ревьюера + решение человека.
    Жизненный цикл: pending (LLM предложил) → approved/rejected/edited
    (человек подтвердил в админке). Одобренные/edited попадают в retrain.

result_json хранит полный список токенов в формате JSON. Это даёт:
  • переиспользование результата в личном кабинете без повторного анализа;
  • готовый датасет для дообучения модели в будущем (экспорт одной таблицы).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator

from app.db import Base


class JSONB(TypeDecorator):
    """Платформо-независимый JSON: JSONB в PostgreSQL, JSON — в остальных СУБД.

    Позволяет одним кодом работать и с PostgreSQL (прод), и со SQLite (тесты).
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
            return dialect.type_descriptor(PG_JSONB())
        # SQLite и прочие — обычный JSON.
        return dialect.type_descriptor(JSON())


class User(Base):
    """Зарегистрированный пользователь."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # is_admin: доступ к /admin/* — ревью LLM-предложений и retrain.
    # server_default=false, чтобы NOT NULL-колонка не упала при миграции
    # существующих строк (важно для batch-режима Alembic в SQLite).
    is_admin: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Связь: запросы пользователя (lazy, чтобы не тянуть N строк без нужды).
    queries: Mapped[list["QueryRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


class QueryRecord(Base):
    """Сохранённый запрос анализа: входной текст + полные результаты пайплайна.

    user_id может быть NULL — анонимные запросы тоже сохраняются (согласно
    решению: публичный доступ, сохраняем всё).
    """

    __tablename__ = "queries"
    __table_args__ = (
        # Индекс для быстрой выборки истории пользователя по дате.
        Index("ix_queries_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Полный ответ пайплайна: список словарей с token/lemma/pos/...
    result_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    tokens_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    user: Mapped[User | None] = relationship(back_populates="queries")

    def __repr__(self) -> str:
        return f"<QueryRecord id={self.id} user_id={self.user_id} tokens={self.tokens_count}>"


# Статусы жизненного цикла аннотации (Active Learning).
# pending  — LLM предложил правку, ждёт решения человека в админке.
# approved — человек одобрил версию LLM как есть (final = llm_*).
# rejected — человек отклонил (ни пайплайн, ни LLM не правы; тоже ценный сигнал).
# edited   — человек ввёл свою версию (final = human override).
ANNOTATION_STATUSES = ("pending", "approved", "rejected", "edited")

# Вердикты LLM-ревьюера (из kazmorph.llm_review.client.ReviewResult).
LLM_VERDICTS = ("correct", "incorrect", "uncertain")


class Annotation(Base):
    """Кандидат на правку из RSS+LLM контура Active Learning.

    Жизненный цикл:
      harvest → select_candidates → llm_review создаёт строку (status=pending)
      → человек в /admin/review решает (approved/rejected/edited)
      → одобренные (approved/edited) попадают в retrain.

    Ключевой инвариант: retrain читает ТОЛЬКО status IN (approved, edited),
    никогда pending/rejected — это защита от заливки непроверенных правок.
    """

    __tablename__ = "annotations"
    __table_args__ = (
        # Быстрая выборка «что на ревью» (pending, по дате).
        Index("ix_annotations_status_created", "status", "created_at"),
        # Поиск аннотаций одного запроса-источника (если есть source_query_id).
        Index("ix_annotations_source_query", "source_query_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ── Происхождение токена ────────────────────────────────────────────
    # source_query_id: если токен из сохранённого запроса пользователя
    # (QueryRecord); nullable — кандидаты из RSS-харвестера не привязаны к запросу.
    # index задан явно в __table_args__ (ix_annotations_source_query), здесь без
    # index=True, чтобы не плодить дублирующие индексы.
    source_query_id: Mapped[int | None] = mapped_column(
        ForeignKey("queries.id", ondelete="SET NULL"), nullable=True
    )
    # URL статьи-источника (для RSS-кандидатов; пусто для пользовательских).
    source_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    # Дата сбора (для группировки в админке по «сводкам»).
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # ── Контекст и разбор пайплайна (что LLM проверял) ──────────────────
    sentence: Mapped[str] = mapped_column(Text, nullable=False)
    token: Mapped[str] = mapped_column(String(255), nullable=False)
    pipeline_lemma: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    pipeline_pos: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    pipeline_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    difficulty_class: Mapped[str] = mapped_column(String(16), default="NONE", nullable=False)

    # ── Предложение LLM-ревьюера ────────────────────────────────────────
    llm_verdict: Mapped[str] = mapped_column(String(16), default="uncertain", nullable=False)
    llm_lemma: Mapped[str | None] = mapped_column(String(255), nullable=True)
    llm_pos: Mapped[str | None] = mapped_column(String(32), nullable=True)
    llm_explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    llm_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # ── Решение человека ────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    # final_*: итоговый разбор для retrain.
    #   approved → final = llm_*
    #   edited   → final = human override (заполняется при редактировании)
    #   pending/rejected → NULL (в retrain не идут)
    final_lemma: Mapped[str | None] = mapped_column(String(255), nullable=True)
    final_pos: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Annotation id={self.id} token={self.token!r} "
            f"status={self.status} verdict={self.llm_verdict}>"
        )
