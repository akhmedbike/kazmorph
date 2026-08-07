"""
Логика сессий и получения текущего пользователя.

Сессии хранятся в подписанных cookies (SessionMiddleware в main.py) —
это обязательно при multi-worker (uvicorn --workers N): у каждого процесса
своя память, поэтому in-memory сессии не подходят.

Текущий пользователь доступен:
  • как зависимость FastAPI:  user = Depends(current_user)
  • в шаблонах:               context["current_user"], context["is_authenticated"]
"""

from __future__ import annotations

import logging
import time

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db import db_available, get_session_factory
from app.models import User

logger = logging.getLogger(__name__)

# Один общий экземпляр шаблонов для рендера HTML-страниц ошибок аутентификации
# (403 «Недостаточно прав»). Логин/регистрация рендерятся своими роутами.
templates = Jinja2Templates(directory="templates")


class BrowserAuthResponse(Exception):
    """Несёт готовый Response для браузерных (``Accept: text/html``) запросов,
    провалившихся через ``require_user``/``require_admin``.

    FastAPI не прерывает роут, если зависимость просто возвращает Response — он
    передаёт его как обычное значение дальше, и роут всё равно рендерит свою
    страницу. Поэтому зависимости поднимают это исключение с уже построенным
    Response (редирект на логин или HTML 403), а единый handler в main.py
    отдаёт его клиенту. Для API (``/api/*`` или без ``text/html``) зависимости
    по-прежнему поднимают ``HTTPException`` с JSON.
    """

    def __init__(self, response):
        self.response = response
        super().__init__("browser auth response")

# Модульная «тормозилка» перебора: после неудачного входа по email — пауза.
# Просто-доступно: словарь {email: timestamp}. Для инстанса в один процесс.
# В продакшене с N воркерами каждый считает свой счётчик, но базовая защита есть.
_login_attempts: dict[str, float] = {}
LOGIN_COOLDOWN_SECONDS = 1.5


def get_db() -> Session | None:
    """Зависимость: сессия БД на время запроса (None, если БД недоступна).

    Сессия закрывается после запроса. Все роуты, работающие с БД,
    должны сначала проверить db_available().
    """
    if not db_available():
        yield None
        return
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(request: Request, db: Session | None = Depends(get_db)) -> User | None:
    """Вернуть текущего пользователя из сессии или None.

    None означает «аноним / не залогинен» — это валидное состояние,
    т.к. доступ к инструменту публичный.
    """
    if db is None:
        return None
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.get(User, user_id)


def current_user(user: User | None = Depends(get_current_user)) -> User | None:
    """Тонкая обёртка над get_current_user для использования как Depends()."""
    return user


def _wants_html(request: Request) -> bool:
    """True, если клиент ожидает HTML (обычный переход браузера по ссылке).

    Браузер всегда шлёт ``Accept: text/html``; AJAX-вызовы (fetch/XHR) по
    умолчанию — нет. На этом различении построено поведение зависимостей ниже:
    браузеру нужен читаемый редирект/страница, а API — JSON с кодом ошибки.
    Путь также проверяется на ``/api/`` как подстраховку (наш API живёт там).
    """
    accept = request.headers.get("accept", "")
    if "/api/" in request.url.path:
        return False
    # text/html может идти с приоритетом (*/*;q=0.8 и т.п.) — достаточно факта
    # его присутствия раньше, чем application/json.
    html_idx = accept.find("text/html")
    json_idx = accept.find("application/json")
    if html_idx == -1:
        return False
    return json_idx == -1 or html_idx < json_idx


def require_user(
    request: Request,
    user: User | None = Depends(get_current_user),
) -> User:
    """Зависимость: только для залогиненных.

    Поведение при отсутствии пользователя зависит от типа запроса:

      * браузер (``Accept: text/html``) → редирект на ``/login?next=<текущий
        путь>``, чтобы после входа вернуться обратно. next ограничен локальными
        путями защитой от open-redirect в login_submit;
      * API (``/api/*`` или без ``text/html``) → ``401 Unauthorized`` с JSON
        ``{"detail": "Требуется вход в систему"}`` для корректной обработки в JS.

    Раньше всегда поднимался HTTPException — и браузер показывал голый JSON,
    что и выглядело как «страница ``{detail: ...}``».

    Возврат ``RedirectResponse``/``TemplateResponse`` прямо из зависимости не
    работает: FastAPI передаёт его как обычное значение в роут, который всё
    равно рендерит свою страницу. Поэтому браузерные ответы пробрасываются
    через исключение ``BrowserAuthResponse``, перехватываемое единым handler-ом
    в main.py (он и отдаёт готовый Response клиенту).
    """
    if user is not None:
        return user

    if _wants_html(request):
        next_url = request.url.path
        if request.url.query:
            next_url += "?" + request.url.query
        # На всякий случай — только локальный путь (login_submit тоже проверяет).
        if not next_url.startswith("/") or next_url.startswith("//"):
            next_url = "/"
        raise BrowserAuthResponse(
            RedirectResponse(f"/login?next={next_url}", status_code=303)
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Требуется вход в систему",
        headers={"Location": "/login"},
    )


def require_admin(
    request: Request,
    user: User = Depends(require_user),
) -> User:
    """Зависимость: только для администраторов (user.is_admin=True).

    Залогиненность проверяется через ``require_user`` (анонимов он отправит на
    логин). Здесь остаётся только проверка прав:

      * браузер → HTML-страница 403 «Недостаточно прав» (templates/403.html),
        чтобы пользователь видел нормальный интерфейс, а не JSON;
      * API → ``403 Forbidden`` с JSON ``{"detail": "Требуются права
        администратора"}``.

    См. ``require_user`` про исключение ``BrowserAuthResponse``.
    """
    if user.is_admin:
        return user

    if _wants_html(request):
        raise BrowserAuthResponse(templates.TemplateResponse(
            request,
            "403.html",
            {
                "request": request,
                "lang": "kk",
                "current_user": user,
                "is_authenticated": True,
                "is_admin": False,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        ))

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Требуются права администратора",
    )


def login_user(request: Request, db: Session, email: str, password: str) -> User | None:
    """Проверить email+пароль и залогинить. Вернуть User или None.

    Вносит паузу при неудаче (защита от перебора).
    """
    # Охлаждение после прошлой неудачи по этому email.
    last_fail = _login_attempts.get(email.lower(), 0)
    elapsed = time.monotonic() - last_fail
    if elapsed < LOGIN_COOLDOWN_SECONDS:
        time.sleep(LOGIN_COOLDOWN_SECONDS - elapsed)

    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if user is None or not user.is_active:
        _login_attempts[email.lower()] = time.monotonic()
        return None

    from app.security import verify_password
    if not verify_password(password, user.password_hash):
        _login_attempts[email.lower()] = time.monotonic()
        return None

    # Успех — чистим счётчик, кладём id в сессию.
    _login_attempts.pop(email.lower(), None)
    request.session["user_id"] = user.id
    return user


def logout_user(request: Request) -> None:
    """Завершить сессию пользователя."""
    request.session.clear()
