"""
Роуты аутентификации: регистрация, вход, выход.

  GET  /register  — форма регистрации
  POST /register  — создать аккаунт
  GET  /login     — форма входа
  POST /login     — залогинить
  POST /logout    — выйти → redirect на /

Доступ к инструменту публичный; аккаунт — опциональная возможность
для сохранения личной истории запросов.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db, login_user, logout_user
from app.db import db_available
from app.models import User
from app.security import hash_password

router = APIRouter()
templates = Jinja2Templates(directory="templates")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8


def _ctx(request: Request, **extra) -> dict:
    """Базовый контекст шаблона с пользователем и флагом БД."""
    db = next(get_db())
    user = None
    if db is not None:
        try:
            user = get_current_user(request, db)
        finally:
            db.close()
    ctx = {
        "request": request,
        "lang": "kk",
        "current_user": user,
        "is_authenticated": user is not None,
        "db_enabled": db_available(),
    }
    ctx.update(extra)
    return ctx


@router.get("/register")
async def register_form(request: Request):
    """Форма регистрации."""
    if not db_available():
        # Без БД регистрация невозможна — но не валим приложение.
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "register.html", _ctx(request))


@router.post("/register")
async def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    """Создать аккаунт. При ошибке — та же форма с сообщением."""
    if not db_available():
        return RedirectResponse("/", status_code=303)

    errors = _validate_registration(email, password, password_confirm)
    email = email.lower().strip()

    if not errors:
        db_gen = get_db()
        db: Session = next(db_gen)
        try:
            exists = db.query(User).filter(User.email == email).first()
            if exists:
                errors.append("email_taken")
            else:
                user = User(email=email, password_hash=hash_password(password))
                db.add(user)
                db.commit()
                db.refresh(user)
                # Автологин после регистрации.
                request.session["user_id"] = user.id
                return RedirectResponse("/dashboard", status_code=303)
        finally:
            db.close()

    ctx = _ctx(request, email=email, errors=errors)
    return templates.TemplateResponse(request, "register.html", ctx, status_code=400)


@router.get("/login")
async def login_form(request: Request, next_url: str = "/"):
    """Форма входа."""
    if not db_available():
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", _ctx(request, next_url=next_url))


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next_url: str = Form("/", alias="next"),
):
    """Залогинить пользователя."""
    if not db_available():
        return RedirectResponse("/", status_code=303)

    email = email.lower().strip()
    db_gen = get_db()
    db: Session = next(db_gen)
    try:
        user = login_user(request, db, email, password)
    finally:
        db.close()

    if user is None:
        ctx = _ctx(request, email=email, next_url=next_url, errors=["invalid_credentials"])
        return templates.TemplateResponse(request, "login.html", ctx, status_code=401)

    # Защита от open-redirect: только локальные пути.
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = "/"
    return RedirectResponse(next_url, status_code=303)


@router.post("/logout")
async def logout(request: Request):
    """Завершить сессию и вернуться на главную."""
    logout_user(request)
    return RedirectResponse("/", status_code=303)


def _validate_registration(email: str, password: str, password_confirm: str) -> list[str]:
    """Вернуть список кодов ошибок валидации (пустой = всё ок)."""
    errors: list[str] = []
    if not EMAIL_RE.match(email or ""):
        errors.append("email_invalid")
    if len(password or "") < MIN_PASSWORD_LEN:
        errors.append("password_short")
    if password != password_confirm:
        errors.append("password_mismatch")
    return errors
