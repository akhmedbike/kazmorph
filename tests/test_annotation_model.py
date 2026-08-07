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
#  require_admin — зависимость FastAPI
# ============================================================


class TestRequireAdmin:
    def test_admin_passes(self, sqlite_db):
        """Админ проходит require_admin без исключения."""
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            admin = make_user(db, email="a@t.kz", is_admin=True)
            # require_admin(admin) не должен бросать.
            result = require_admin(user=admin)
            assert result is admin
        finally:
            db.close()

    def test_non_admin_raises_403(self, sqlite_db):
        from fastapi import HTTPException
        from tests.conftest import make_user
        from app.db import get_session_factory
        factory = get_session_factory()
        db = factory()
        try:
            regular = make_user(db, email="r@t.kz", is_admin=False)
            with pytest.raises(HTTPException) as exc:
                require_admin(user=regular)
            assert exc.value.status_code == 403
        finally:
            db.close()

    def test_anonymous_raises_401(self):
        """require_admin зависит от require_user, который даёт 401 для анонима.

        Здесь проверяем сам require_user (require_admin его вызывает первым).
        """
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            require_user(user=None)
        assert exc.value.status_code == 401
