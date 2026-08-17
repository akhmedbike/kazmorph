"""
KazMorphCorpus-2026: FastAPI application
Online morphological analyzer and corrector for Kazakh language.

Архитектура:
  • NLP-ядро kazmorph/ — чистые функции, без знаний о вебе.
  • Веб-слой app/ — БД (SQLAlchemy), авторизация (сессии), личный кабинет.
  • main.py — сборка приложения, страницы, точка сбора данных.

Сбор данных: каждый запрос анализа сохраняется в БД (входной текст + полный
ответ пайплайна) для последующего дообучения модели. Анонимные запросы
сохраняются с user_id=NULL. Если БД недоступна — приложение работает
в прежнем stateless-режиме (graceful degradation).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app.auth import BrowserAuthResponse, get_current_user
from app.db import Base, db_available, init_db
from app.models import Annotation, QueryRecord  # noqa: F401 — регистрирует модели для Base.metadata
from app.routes.auth import router as auth_router
from app.routes.dashboard import router as dashboard_router
from app.routes.admin import router as admin_router
from kazmorph.pipeline import Pipeline
from kazmorph.api_v1 import router as v1_router

logger = logging.getLogger(__name__)

# Pipeline is initialized once and reused across requests
_pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация: NLP-пайплайн + подключение к БД."""
    global _pipeline
    # CRF: сначала pointer versions/ (retrain-loop), затем fallback на legacy.
    from kazmorph.model_versioning import resolve_current_model_path
    crf_model = resolve_current_model_path()
    neural_path = "models/neural_pos"
    neural_model = neural_path if os.path.exists(neural_path) else None
    lemma_path = "models/neural_lemma"
    lemma_model = lemma_path if os.path.exists(lemma_path) else None
    _pipeline = Pipeline(
        crf_model_path=crf_model,
        neural_model_path=neural_model,
        lemma_model_path=lemma_model,
    )
    app.state.pipeline = _pipeline

    # Подключаемся к БД (если DATABASE_URL задана). При успехе создаём таблицы
    # как fallback (основной путь — миграции Alembic, но это гарантирует старт
    # при деплое без запуска миграций).
    if init_db():
        from app.db import get_engine

        engine = get_engine()
        if engine is not None:
            try:
                Base.metadata.create_all(bind=engine)
            except Exception as exc:  # noqa: BLE001
                logger.warning("create_all не выполнен: %s", exc)

    yield
    _pipeline = None


app = FastAPI(
    title="KazMorphCorpus-2026",
    description="Morphological analyzer and online corrector for Kazakh language",
    version="0.1.0",
    lifespan=lifespan,
)

# Сессии в подписанных cookies — обязательно для multi-worker uvicorn.
# Секрет из окружения; безопасный флаг включается за HTTPS.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "dev-insecure-change-me-in-production"),
    session_cookie="kazmorph_session",
    max_age=30 * 24 * 3600,  # 30 дней
    same_site="lax",
    https_only=os.environ.get("SESSION_SECURE", "").lower() in ("1", "true", "yes"),
)


# Браузерные ответы аутентификации (редирект на логин / HTML 403) поднимаются
# из require_user/require_admin как исключение, потому что возврат Response
# прямо из зависимости FastAPI игнорирует (роут всё равно выполняет свой код).
# Этот handler отдаёт уже построенный Response клиенту. См. app/auth.py.
@app.exception_handler(BrowserAuthResponse)
async def _browser_auth_handler(request, exc: BrowserAuthResponse):
    return exc.response

# Static files (CSS, JS) — mounted before routes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Integration API v1 (correct, batch, evaluate, info)
app.include_router(v1_router)
# Auth (login/register/logout) + личный кабинет (dashboard)
app.include_router(auth_router)
app.include_router(dashboard_router)
# Админка Active Learning (ревью LLM-предложений, /api/low_confidence)
app.include_router(admin_router)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Default sample text
DEFAULT_TEXT = (
    "Қазақстан Республикасының Конституциясы бойынша әрбір адамның "
    "өмір сүруге, жеке басының бостандығына және қорғалуына құқығы бар. "
    "Осы құқықтарды бұзуға ешкімнің құқығы жоқ."
)


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    tokens: list[dict]
    query_id: int | None = None  # ID сохранённой записи (для ссылки в личный кабинет)


def _get_active_stages() -> list[str]:
    """Get list of active pipeline stages."""
    stages = ["RMA"]
    if _pipeline and _pipeline.crf.model is not None:
        stages.append("CRF")
    if _pipeline and _pipeline.neural.model is not None:
        stages.append("Neural")
    if _pipeline and _pipeline.lemmatizer.model is not None:
        stages.append("Lemma")
    stages.append("MFRN")
    return stages


def _save_query(request: Request, text: str, tokens: list[dict]) -> int | None:
    """Сохранить запрос в БД. Вернуть ID записи или None (БД недоступна).

    Привязывает к user_id из сессии или NULL для анонима.
    Ошибки БД не валивают запрос — анализ в любом случае отдаётся пользователю.
    """
    if not db_available():
        return None
    from app.db import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return None
    session = factory()
    try:
        user_id = request.session.get("user_id")
        record = QueryRecord(
            user_id=user_id,
            input_text=text,
            result_json=tokens,
            tokens_count=len(tokens),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    except Exception as exc:  # noqa: BLE001 — БД не должна валить анализ
        session.rollback()
        logger.warning("Не удалось сохранить запрос в БД: %s", exc)
        return None
    finally:
        session.close()


@app.get("/")
async def index(request: Request):
    """Dashboard UI (default) — KPIs, distributions, highlighted text, table."""
    user = None
    if db_available():
        from app.db import get_session_factory

        factory = get_session_factory()
        if factory:
            db = factory()
            try:
                user = get_current_user(request, db)
            finally:
                db.close()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "lang": "kk",
            "stages": _get_active_stages(),
            "neural_active": _pipeline is not None and _pipeline.neural.model is not None,
            "default_text": DEFAULT_TEXT,
            "current_user": user,
            "is_authenticated": user is not None,
            "db_enabled": db_available(),
        },
    )


@app.get("/legacy")
async def legacy(request: Request):
    """Legacy two-panel analyzer UI — kept for reference / fallback."""
    user = None
    if db_available():
        from app.db import get_session_factory

        factory = get_session_factory()
        if factory:
            db = factory()
            try:
                user = get_current_user(request, db)
            finally:
                db.close()
    return templates.TemplateResponse(
        request=request,
        name="legacy.html",
        context={
            "lang": "kk",
            "stages": _get_active_stages(),
            "neural_active": _pipeline is not None and _pipeline.neural.model is not None,
            "default_text": DEFAULT_TEXT,
            "current_user": user,
            "is_authenticated": user is not None,
            "db_enabled": db_available(),
        },
    )


@app.get("/api-docs")
async def api_docs(request: Request):
    """Страница с полной документацией API и примерами интеграции."""
    return templates.TemplateResponse(
        request=request,
        name="docs.html",
        context={
            "lang": "kk",
            "base_url": str(request.base_url).rstrip("/"),
        },
    )


@app.get("/compare")
async def compare(request: Request):
    """A/B страница сравнения корректоров: /correct (бинарный) vs /correct3 (KEEP/CORRECT/REVIEW).

    Один мәтін — екі тәсіл. Сол жақта ескі корректор әр төмен-сенімді сөзді
    ауыстырады; оң жақта жаңа үш жақты корректор шешімді KEEP/CORRECT/REVIEW
    бойынша түспен көрсетеді. REVIEW түпнұсқаны сақтайды және үміткерлерді
    көрсетеді — бұл жаңа мүмкіндікті көрнекі түрде сынап көруге арналған.
    """
    user = None
    if db_available():
        from app.db import get_session_factory

        factory = get_session_factory()
        if factory:
            db = factory()
            try:
                user = get_current_user(request, db)
            finally:
                db.close()
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "lang": "kk",
            "default_text": DEFAULT_TEXT,
            "current_user": user,
            "is_authenticated": user is not None,
            "db_enabled": db_available(),
        },
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, req: AnalyzeRequest):
    """Analyze Kazakh text through the full 4-stage pipeline.

    Сохраняет запрос в БД для последующего дообучения модели
    (входной текст + полные результаты). Анонимные запросы — с user_id=NULL.
    """
    result = _pipeline.process(req.text)
    query_id = _save_query(request, req.text, result)
    return AnalyzeResponse(tokens=result, query_id=query_id)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "stages": _get_active_stages()}
