"""
Админка Active Learning: ревью LLM-предложений + ручной сбор источников.

  GET  /admin                          — дашборд: счётчики pending/approved/rejected
  GET  /admin/review                   — таблица pending-кандидатов (пагинация)
  GET  /admin/review/{annotation_id}   — детальная карточка одного кандидата
  POST /admin/review/{annotation_id}/decision — approve/reject/edit (JSON-API)
  GET  /api/low_confidence             — pending-кандидаты как JSON (для интеграций)
  GET  /api/admin/stats                — сводная статистика для дашборда

  GET  /admin/sources                  — управление циклом AL: список источников + 3 кнопки
  POST /admin/sources/run              — запустить шаг (harvest/select/review) фоновой задачей
  GET  /admin/sources/status           — статус последнего/текущего шага (для polling)
  GET  /admin/sources/candidates          — список кандидатов за дату (для модального окна)
  GET  /admin/sources/candidates/preview — превью кандидатов + оценка стоимости LLM

Доступ только для администраторов (require_admin → 403 для не-админа,
401 для анонима). Все правки LLM проходят ручное подтверждение здесь —
это защита от рисков №1 (некомпетентность LLM) и №2 (model collapse).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.auth import get_db, require_admin
from app.models import ANNOTATION_STATUSES, Annotation, User
from app.services.harvest_runner import (
    STEPS,
    TaskError,
    count_candidates_today,
    get_task_state,
    start_task,
)
from kazmorph.pipeline import ACTIVE_LEARNING_THRESHOLD

router = APIRouter()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger("admin")

PER_PAGE = 25


# ============================================================
#  Контекст и хелперы
# ============================================================


def _ctx(request: Request, user: User, **extra) -> dict:
    return {
        "request": request,
        "lang": "kk",
        "current_user": user,
        "is_authenticated": True,
        "is_admin": True,
        "al_threshold": ACTIVE_LEARNING_THRESHOLD,
        **extra,
    }


def _get_annotation(db: Session, annotation_id: int) -> Annotation:
    ann = db.get(Annotation, annotation_id)
    if ann is None:
        raise HTTPException(status_code=404, detail="Аннотация не найдена")
    return ann


def _require_db(db: Session | None) -> Session:
    if db is None:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    return db


# ============================================================
#  Модель решения (Pydantic) для POST decision
# ============================================================


class DecisionRequest(BaseModel):
    """Тело POST /admin/review/{id}/decision.

    decision: approved  → final = llm_* (человек согласен с LLM)
              rejected  → ни пайплайн, ни LLM не правы (final = NULL)
              edited    → final = human override (требует final_lemma/pos)
    """
    decision: str = Field(..., description="approved | rejected | edited")
    final_lemma: str | None = Field(None, description="Своя лемма (только для edited)")
    final_pos: str | None = Field(None, description="Свой POS (только для edited)")


class RunRequest(BaseModel):
    """Тело POST /admin/sources/run.

    step:   harvest | select | review — шаг цикла Active Learning.
    source: только для harvest — имя источника или null (все).
    """
    step: str = Field(..., description="harvest | select | review")
    source: str | None = Field(None, description="Имя источника (только для harvest)")


def _apply_decision(ann: Annotation, body: DecisionRequest, reviewer: User) -> None:
    """Применить решение к аннотации (валидация + заполнение final_*)."""
    if body.decision not in ("approved", "rejected", "edited"):
        raise HTTPException(status_code=422, detail=f"Неизвестное decision: {body.decision!r}")

    ann.status = body.decision
    ann.reviewer_id = reviewer.id
    ann.reviewed_at = datetime.now(timezone.utc)

    if body.decision == "approved":
        # final = версия LLM. Если LLM не дал исправления (verdict=correct/
        # uncertain), сохраняем pipeline-версию — иначе final будет NULL и
        # правка не попадёт в retrain (хотя человек её одобрил).
        ann.final_lemma = ann.llm_lemma or ann.pipeline_lemma
        ann.final_pos = ann.llm_pos or ann.pipeline_pos
    elif body.decision == "edited":
        # Человек ввёл свою версию — она обязательна.
        if not body.final_lemma or not body.final_pos:
            raise HTTPException(
                status_code=422,
                detail="Для edited обязательны final_lemma и final_pos",
            )
        ann.final_lemma = body.final_lemma.strip()
        ann.final_pos = body.final_pos.strip().upper()
    else:  # rejected
        ann.final_lemma = None
        ann.final_pos = None


# ============================================================
#  Роуты
# ============================================================


@router.get("/admin")
async def admin_dashboard(
    request: Request,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Дашборд админки: счётчики по статусам."""
    db = _require_db(db)
    # Считаем аннотации по статусам.
    status_counts = {}
    for status in ANNOTATION_STATUSES:
        status_counts[status] = db.query(func.count(Annotation.id)).filter(
            Annotation.status == status
        ).scalar() or 0
    total = sum(status_counts.values())
    # Последняя дата сбора (для заголовка «сводка за …»).
    latest = db.query(func.max(Annotation.fetched_at)).scalar()
    # Согласие LLM с человеком (из уже размеченных).
    reviewed = db.query(Annotation).filter(Annotation.status.in_(["approved", "rejected", "edited"])).all()
    llm_agreement = _compute_agreement(reviewed)
    return templates.TemplateResponse(
        request,
        "admin/dashboard.html",
        _ctx(request, user,
             status_counts=status_counts, total=total,
             latest_fetched=latest, llm_agreement=llm_agreement),
    )


def _compute_agreement(reviewed: list[Annotation]) -> dict | None:
    """Доля случаев, где человек согласился с LLM (approved vs rejected/edited).

    Метрика доверия к LLM-ревьюеру со временем (риск №5 из idea-документа).
    Возвращает None, если ещё нет размеченных данных.
    """
    if not reviewed:
        return None
    approved = sum(1 for a in reviewed if a.status == "approved")
    return {
        "n_reviewed": len(reviewed),
        "n_approved": approved,
        "agreement_rate": round(approved / len(reviewed), 4),
    }


@router.get("/admin/review")
async def admin_review_list(
    request: Request,
    page: int = 1,
    status: str = "pending",
    db: Session | None = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Таблица кандидатов на ревью (по умолчанию pending)."""
    db = _require_db(db)
    if status not in ANNOTATION_STATUSES:
        status = "pending"
    page = max(1, page)
    offset = (page - 1) * PER_PAGE

    base = db.query(Annotation)
    if status != "all":
        base = base.filter(Annotation.status == status)
    total = base.count()
    annotations = (
        base.order_by(desc(Annotation.created_at))
        .offset(offset)
        .limit(PER_PAGE)
        .all()
    )
    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    return templates.TemplateResponse(
        request,
        "admin/review.html",
        _ctx(request, user,
             annotations=annotations, status=status,
             page=page, pages=pages, total=total,
             statuses=ANNOTATION_STATUSES),
    )


@router.get("/admin/review/{annotation_id}")
async def admin_review_detail(
    request: Request,
    annotation_id: int,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Детальная карточка одного кандидата."""
    db = _require_db(db)
    ann = _get_annotation(db, annotation_id)
    return templates.TemplateResponse(
        request,
        "admin/review.html",  # та же страница, но с одним выделенным элементом
        _ctx(request, user, annotations=[ann], status=ann.status,
             page=1, pages=1, total=1, statuses=ANNOTATION_STATUSES,
             focus_id=annotation_id),
    )


@router.post("/admin/review/{annotation_id}/decision")
async def admin_decision(
    request: Request,
    annotation_id: int,
    body: DecisionRequest,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Применить решение: approved/rejected/edited.

    JSON-API для AJAX-кнопок в таблице ревью.
    """
    db = _require_db(db)
    ann = _get_annotation(db, annotation_id)
    _apply_decision(ann, body, user)
    db.commit()
    db.refresh(ann)
    return JSONResponse({
        "ok": True,
        "annotation_id": ann.id,
        "status": ann.status,
        "final_lemma": ann.final_lemma,
        "final_pos": ann.final_pos,
    })


# ============================================================
#  Ручное управление циклом Active Learning (harvest / select / review)
# ============================================================


def _load_sources_list() -> list[dict]:
    """Прочитать config/sources.yaml → список источников (для страницы /admin/sources).

    Ленивый импорт: harvest_sources при import делает sys.path.insert, что
    безопасно, но оставлять его на старте приложения незачем.
    """
    from scripts.harvest_sources import load_sources

    try:
        return load_sources()
    except (OSError, ValueError) as exc:
        # Конфиг отсутствует/битый — отдаём пустой список, страница покажет
        # сообщение вместо крэша.
        logger.warning("Не удалось прочитать sources.yaml: %s", exc)
        return []


@router.get("/admin/sources")
async def admin_sources(
    request: Request,
    user: User = Depends(require_admin),
):
    """Страница управления циклом AL: список источников + кнопки 3 шагов."""
    sources = _load_sources_list()
    return templates.TemplateResponse(
        request,
        "admin/sources.html",
        _ctx(
            request,
            user,
            sources=sources,
            task_state=get_task_state(),
            candidates_preview=count_candidates_today(),
        ),
    )


@router.post("/admin/sources/run")
async def admin_al_run(
    body: RunRequest,
    user: User = Depends(require_admin),
):
    """Запустить шаг цикла AL фоновой задачей. 202 + снимок состояния.

    step: harvest | select | review. Для harvest можно указать source
    (имя источника или null = все). 409 — уже выполняется другой шаг;
    422 — неизвестный шаг или источник не найден в sources.yaml.
    """
    if body.step not in STEPS:
        raise HTTPException(
            status_code=422,
            detail=f"Неизвестный шаг {body.step!r}",
        )
    if body.step == "harvest" and body.source is not None:
        names = {s["name"] for s in _load_sources_list()}
        if body.source not in names:
            raise HTTPException(
                status_code=422,
                detail=f"Источник {body.source!r} не найден в sources.yaml",
            )
    try:
        start_task(body.step, source=body.source if body.step == "harvest" else None)
    except TaskError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JSONResponse(status_code=202, content=get_task_state())


@router.get("/admin/sources/status")
async def admin_al_status(
    user: User = Depends(require_admin),
):
    """Снимок состояния последнего/текущего шага (для polling из JS)."""
    return JSONResponse(get_task_state())


@router.get("/admin/sources/candidates/preview")
async def admin_candidates_preview(
    user: User = Depends(require_admin),
):
    """Превью кандидатов за сегодня + оценка стоимости LLM-ревью.

    Для confirm-диалога перед шагом review: показывает, сколько кандидатов
    будет обработано и сколько это будет стоить (~$). Если кандидатов 0 —
    UI блокирует кнопку review.
    """
    return JSONResponse(count_candidates_today())


@router.get("/admin/sources/candidates")
async def admin_candidates_list(
    date: str | None = None,
    limit: int = 200,
    user: User = Depends(require_admin),
):
    """Список кандидатов за дату (JSON) — для модального окна в /admin/sources.

    Читает data/candidates/{date}.jsonl (по умолчанию — сегодня). Параметр
    ``limit`` ограничивает размер выдачи (default 200), чтобы не тянуть
    тысячи строк в браузер. Возвращает сводку по сложности + сами записи.
    """
    from scripts.llm_review import load_candidates

    pv = count_candidates_today()
    date_str = date or pv["date"]
    limit = max(1, min(limit, 1000))
    candidates = load_candidates(date_str)[:limit]
    # Сводка по классам сложности (для шапки модального окна).
    by_diff: dict[str, int] = {}
    for c in candidates:
        dc = c.get("difficulty_class", "NONE")
        by_diff[dc] = by_diff.get(dc, 0) + 1
    return JSONResponse({
        "date": date_str,
        "n_shown": len(candidates),
        "n_total": pv["n_candidates"] if date is None else None,
        "by_difficulty": by_diff,
        "candidates": [
            {
                "token": c.get("token", ""),
                "lemma": c.get("lemma", ""),
                "pos": c.get("pos", ""),
                "confidence": c.get("confidence", 0),
                "difficulty_class": c.get("difficulty_class", "NONE"),
                "sentence": c.get("sentence", ""),
                "source_url": c.get("source_url", ""),
            }
            for c in candidates
        ],
    })


# ============================================================
#  JSON API
# ============================================================


@router.get("/api/low_confidence")
async def api_low_confidence(
    request: Request,
    page: int = 1,
    limit: int = 50,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Pending-кандидаты как JSON — наконец подключает ACTIVE_LEARNING_THRESHOLD.

    Для будущих интеграций (внешний скрипт, экспорт, аналитика).
    """
    db = _require_db(db)
    page = max(1, page)
    limit = max(1, min(limit, 200))
    offset = (page - 1) * limit
    total = db.query(func.count(Annotation.id)).filter(Annotation.status == "pending").scalar() or 0
    annotations = (
        db.query(Annotation)
        .filter(Annotation.status == "pending")
        .order_by(desc(Annotation.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return JSONResponse({
        "page": page,
        "pages": max(1, (total + limit - 1) // limit),
        "total": total,
        "threshold": ACTIVE_LEARNING_THRESHOLD,
        "candidates": [_annotation_to_dict(a) for a in annotations],
    })


@router.get("/api/admin/stats")
async def api_admin_stats(
    request: Request,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Сводная статистика для дашборда (JSON)."""
    db = _require_db(db)
    status_counts = {}
    for s in ANNOTATION_STATUSES:
        status_counts[s] = db.query(func.count(Annotation.id)).filter(
            Annotation.status == s
        ).scalar() or 0
    reviewed = db.query(Annotation).filter(Annotation.status.in_(["approved", "rejected", "edited"])).all()
    return JSONResponse({
        "status_counts": status_counts,
        "total": sum(status_counts.values()),
        "llm_agreement": _compute_agreement(reviewed),
    })


def _annotation_to_dict(a: Annotation) -> dict:
    return {
        "id": a.id,
        "sentence": a.sentence,
        "token": a.token,
        "pipeline_lemma": a.pipeline_lemma,
        "pipeline_pos": a.pipeline_pos,
        "pipeline_confidence": a.pipeline_confidence,
        "difficulty_class": a.difficulty_class,
        "llm_verdict": a.llm_verdict,
        "llm_lemma": a.llm_lemma,
        "llm_pos": a.llm_pos,
        "llm_explanation": a.llm_explanation,
        "status": a.status,
        "source_url": a.source_url,
        "fetched_at": a.fetched_at.isoformat() if a.fetched_at else None,
    }
