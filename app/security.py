"""
Хеширование и проверка паролей через bcrypt напрямую.

bcrypt — адаптивный алгоритм с солью; устойчив к перебору даже при утечке БД.
Используем библиотеку bcrypt напрямую (без passlib): passlib 1.7.4 несовместим
с bcrypt 5.x (ищет убранный bcrypt.__about__), а прямой вызов проще и надёжнее.

bcrypt обрезает пароли до 72 байт — это ограничение самого алгоритма. Мы
хешируем пароль как есть; для паролей короче 72 байт (практически все) это
не имеет значения. Хеш хранится в виде utf-8 строки.
"""

from __future__ import annotations

import bcrypt

# bcrypt требует байты; кодируем utf-8 (поддерживает казахскую кириллицу в паролях).
_BCRYPT_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    """Вернуть bcrypt-хеш пароля как строку."""
    pw = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Проверить пароль против хеша. True при совпадении."""
    try:
        pw = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
