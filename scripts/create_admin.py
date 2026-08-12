#!/usr/bin/env python3
"""
Управление учётной записью администратора.

Создаёт нового пользователя с правами администратора (is_admin=True) либо
повышает уже существующего пользователя до администратора. При необходимости
сбрасывает пароль.

Использование:

  # Создать администратора (пароль спросится скрытым вводом):
  DATABASE_URL=... python scripts/create_admin.py admin@example.kz

  # Повысить существующего пользователя и сразу сменить пароль:
  DATABASE_URL=... python scripts/create_admin.py admin@example.kz --reset-password

Правила валидации те же, что в веб-форме регистрации (app/routes/auth.py):
email по EMAIL_RE, пароль длиной от MIN_PASSWORD_LEN символов. email
нормализуется (lower + strip) — так же, как при регистрации.

Требует DATABASE_URL в окружении. Без неё скрипт завершается с ошибкой
(как и все скрипты, работающие с БД).
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

# Добавить корень проекта в sys.path, чтобы `import app.*` работал при запуске
# из любого места: python scripts/create_admin.py ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# EMAIL_RE и MIN_PASSWORD_LEN продублированы локально, чтобы не тянуть весь
# веб-слой (FastAPI, шаблоны) в простой CLI-скрипт. Держите в синхроне с
# app/routes/auth.py.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8


def _prompt_password(confirm: bool = True) -> str:
    """Спросить пароль скрытым вводом, с подтверждением и проверкой длины."""
    while True:
        pw = getpass.getpass("Пароль: ")
        if len(pw) < MIN_PASSWORD_LEN:
            print(f"  пароль короче {MIN_PASSWORD_LEN} символов, попробуйте снова",
                  file=sys.stderr)
            continue
        if confirm:
            pw2 = getpass.getpass("Подтвердите пароль: ")
            if pw != pw2:
                print("  пароли не совпадают, попробуйте снова", file=sys.stderr)
                continue
        return pw


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Создать администратора или повысить пользователя до администратора.",
    )
    parser.add_argument("email", help="E-mail учётной записи (нижний регистр).")
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="Сменить пароль (для существующего пользователя).",
    )
    args = parser.parse_args()

    email = args.email.strip().lower()
    if not EMAIL_RE.match(email):
        print(f"Ошибка: некорректный e-mail: {email!r}", file=sys.stderr)
        return 1

    # Импорт app.* после разбора аргументов — чтобы --help работал без БД.
    from app.db import db_available, get_session_factory, init_db
    from app.models import User  # noqa: F401 — регистрирует модель в Base
    from app.security import hash_password

    if not init_db() or not db_available():
        # init_db()/db_available() уже залогировали причину.
        print("Ошибка: БД недоступна. Задайте DATABASE_URL в окружении.",
              file=sys.stderr)
        return 1

    factory = get_session_factory()
    if factory is None:
        print("Ошибка: фабрика сессий не инициализирована (БД недоступна).",
              file=sys.stderr)
        return 1

    with factory() as session:
        user = session.query(User).filter(User.email == email).first()

        if user is None:
            # Новый администратор: обязательная установка пароля.
            print(f"Создание администратора {email} ...")
            password = _prompt_password(confirm=True)
            user = User(
                email=email,
                password_hash=hash_password(password),
                is_admin=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            print(f"Готово: создан администратор id={user.id} email={user.email}.")
            return 0

        # Существующий пользователь: повышение + опциональная смена пароля.
        changed: list[str] = []
        if not user.is_admin:
            user.is_admin = True
            changed.append("is_admin=True")
        if args.reset_password:
            password = _prompt_password(confirm=True)
            user.password_hash = hash_password(password)
            changed.append("password_hash (сброшен)")

        if not changed:
            print(f"Пользователь {email} уже администратор, изменений нет "
                  f"(используйте --reset-password для смены пароля).")
            return 0

        session.commit()
        print(f"Готово: {email} обновлён — {', '.join(changed)}.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
