"""Создание/повышение администратора.

Usage:
  python scripts/create_admin.py EMAIL

Пароль запрашивается скрытым вводом (getpass), чтобы не оставался в истории shell.

Поведение:
  • email не существует → создаётся User(is_active=True, is_admin=True);
  • email существует    → повышается до admin (is_admin=True, is_active=True),
    пароль НЕ меняется (чтобы не было тихого захвата чужого аккаунта).
    Сменить пароль: --reset-password.

DATABASE_URL берётся из окружения (в Docker подставляется compose).
Запуск внутри контейнера:
  docker compose exec kazmorph python scripts/create_admin.py admin@example.com
"""

from __future__ import annotations

import getpass
import sys

from sqlalchemy import select

from app.db import db_available, get_session_factory, init_db
from app.models import User
from app.security import hash_password, verify_password


def main(argv: list[str]) -> int:
    if len(argv) < 2 or not argv[1].strip() or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv[1:] else 2

    email = argv[1].strip().lower()
    reset_password = "--reset-password" in argv[2:]

    if not init_db() or not db_available():
        print("ERROR: БД недоступна — задайте DATABASE_URL.", file=sys.stderr)
        return 1

    factory = get_session_factory()
    with factory() as db:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            password = getpass.getpass("Пароль: ")
            if getpass.getpass("Ещё раз: ") != password:
                print("ERROR: пароли не совпадают.", file=sys.stderr)
                return 1
            user = User(
                email=email,
                password_hash=hash_password(password),
                is_active=True,
                is_admin=True,
            )
            db.add(user)
            db.commit()
            print(f"Создан админ: id={user.id} email={user.email}")
        else:
            user.is_admin = True
            user.is_active = True
            if reset_password:
                password = getpass.getpass("Новый пароль: ")
                user.password_hash = hash_password(password)
            db.commit()
            action = "пароль сброшен" if reset_password else "пароль без изменений"
            print(f"Повышен до admin: id={user.id} email={user.email} ({action})")

        # ponytail: самопроверка — хеш действительно валидируется.
        assert verify_password("__definitely_wrong__", user.password_hash) is False
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
