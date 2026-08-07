"""
Личный кабинет: история запросов текущего пользователя.

  GET  /dashboard                 — список запросов (пагинация)
  GET  /dashboard/query/{id}      — детальный просмотр сохранённого запроса
  GET  /api/dashboard/queries     — JSON-пагинация (для AJAX)
  GET  /api/dashboard/stats       — сводная статистика пользователя

Доступ только для залогиненных. Анонимные запросы сохраняются в БД, но
их может видеть только администратор (в этом scope — только свои).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db, require_user
from app.models import QueryRecord, User

router = APIRouter()
templates = Jinja2Templates(directory="templates")

PER_PAGE = 20


def _ctx(request: Request, **extra) -> dict:
    """Базовый контекст с пользователем."""
    user = extra.pop("current_user", None)
    ctx = {
        "request": request,
        "lang": "kk",
        "current_user": user,
        "is_authenticated": user is not None,
    }
    ctx.update(extra)
    return ctx


def _get_own_query(db: Session, user: User, query_id: int) -> QueryRecord:
    """Вернуть запрос пользователя по id или 404 (чужой/несуществующий)."""
    record = db.get(QueryRecord, query_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Запрос не найден")
    return record


@router.get("/dashboard")
async def dashboard(
    request: Request,
    page: int = 1,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_user),
):
    """Личный кабинет: список запросов с пагинацией."""
    page = max(1, page)
    if db is None:
        raise HTTPException(status_code=503, detail="База данных недоступна")

    offset = (page - 1) * PER_PAGE
    total = db.query(func.count(QueryRecord.id)).filter(QueryRecord.user_id == user.id).scalar() or 0
    records = (
        db.query(QueryRecord)
        .filter(QueryRecord.user_id == user.id)
        .order_by(desc(QueryRecord.created_at))
        .offset(offset)
        .limit(PER_PAGE)
        .all()
    )

    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    return templates.TemplateResponse(
        request,
        "account.html",
        _ctx(request, current_user=user, records=records, page=page, pages=pages, total=total),
    )


@router.get("/dashboard/query/{query_id}")
async def query_detail(
    request: Request,
    query_id: int,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_user),
):
    """Детальный просмотр сохранённого запроса (входной текст + результаты)."""
    if db is None:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    record = _get_own_query(db, user, query_id)
    return templates.TemplateResponse(
        request,
        "query_detail.html",
        _ctx(request, current_user=user, record=record),
    )


@router.get("/api/dashboard/queries")
async def api_queries(
    request: Request,
    page: int = 1,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_user),
):
    """JSON-список запросов для AJAX-пагинации."""
    if db is None:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    page = max(1, page)
    offset = (page - 1) * PER_PAGE
    total = db.query(func.count(QueryRecord.id)).filter(QueryRecord.user_id == user.id).scalar() or 0
    records = (
        db.query(QueryRecord)
        .filter(QueryRecord.user_id == user.id)
        .order_by(desc(QueryRecord.created_at))
        .offset(offset)
        .limit(PER_PAGE)
        .all()
    )
    return JSONResponse({
        "page": page,
        "pages": max(1, (total + PER_PAGE - 1) // PER_PAGE),
        "total": total,
        "queries": [
            {
                "id": r.id,
                "input_preview": (r.input_text[:120] + "…") if len(r.input_text) > 120 else r.input_text,
                "tokens_count": r.tokens_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    })


@router.get("/api/dashboard/stats")
async def api_stats(
    request: Request,
    db: Session | None = Depends(get_db),
    user: User = Depends(require_user),
):
    """Сводная статистика пользователя."""
    if db is None:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    total_queries = db.query(func.count(QueryRecord.id)).filter(QueryRecord.user_id == user.id).scalar() or 0
    total_tokens = db.query(func.sum(QueryRecord.tokens_count)).filter(QueryRecord.user_id == user.id).scalar() or 0
    return JSONResponse({
        "total_queries": total_queries,
        "total_tokens": int(total_tokens),
    })
