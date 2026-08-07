"""
Текстовый корректор KazMorph.

Собирает исправленный текст из результатов анализа пайплайна: для токенов с
низкой уверенностью или нераспознанных слов подставляет лучший вариант
исправления (Levenshtein, см. kazmorph/suggestor.py), сохраняя исходное
форматирование (пробелы, знаки препинания) нетронутым.

Это главная функция «системы автоматической коррекции» из ТЗ (ЧАСТЬ B,
Этап 4): пайплайн анализирует, а корректор применяет правки и возвращает
готовый исправленный текст.
"""

from __future__ import annotations

import re

from kazmorph.pipeline import Pipeline

# Регулярка токенизации, идентичная Pipeline._tokenize. Хранится копией
# (а не импортируется), чтобы избежать циклического импорта corrector↔pipeline.
# Если правила токенизации в Pipeline._tokenize меняются, эту копию надо
# обновить синхронно — иначе позиции токенов в correct() разъедутся с
# результатами pipeline.process(), и исправления попадут не на те слова.
#
# Сборка паттерна повторяет Pipeline._tokenize: десятичные числа, числительные
# с суффиксом через дефис, дефисные композиты, аббревиатуры с точкой.
_KZ = r"а-яәөүғқңһұіА-ЯӘӨҮҒҚҢҺҰІa-zA-Z"
_TOKEN_RE = re.compile(
    r"\d+[.,]\d+%?"
    r"|\d+-[а-яәөүғқңһұі]+"
    r"|[" + _KZ + r"]+-[" + _KZ + r"]+"
    r"|[" + _KZ + r"]+\."
    r"|[" + _KZ + r"\d]+"
    r"|[^\s]"
)


def correct(
    pipeline: Pipeline,
    text: str,
    confidence_threshold: float = 0.85,
) -> dict:
    """
    Исправить текст: заменить нераспознанные / низкоуверенные токены на лучший
    вариант исправления.

    Параметры:
        pipeline: экземпляр Pipeline.
        text: исходный текст.
        confidence_threshold: токены с уверенностью ниже порога (либо POS=UNK),
            у которых есть варианты исправления, заменяются. По умолчанию 0.85 —
            это зона откалиброванного softmax, где реальная точность падает
            ниже ~70% (раньше было 0.5, но эвристический confidence почти
            никогда не опускался так низко, поэтому корректор молчал).

    Возвращает::

        {
          "original": str,
          "corrected": str,
          "corrections": [
            {"position": int, "token": str, "suggestion": str,
             "confidence": float, "pos": str}, ...
          ],
          "stats": {"tokens": int, "corrected": int},
        }
    """
    results = pipeline.process(text)
    matches = list(_TOKEN_RE.finditer(text))

    corrections: list[dict] = []
    out_parts: list[str] = []
    last_end = 0
    applied = 0

    # results и matches выровнены по одной регулярке; берём минимум для безопасности
    n = min(len(results), len(matches))

    for i in range(n):
        res = results[i]
        m = matches[i]
        # сохраняем текст между токенами (пробелы, знаки) без изменений
        out_parts.append(text[last_end:m.start()])
        last_end = m.end()

        original = res["token"]
        suggestions = res.get("suggestions") or []
        low_conf = res.get("confidence", 1.0) < confidence_threshold
        is_unknown = res.get("pos") == "UNK"

        suggestion = None
        if suggestions and (low_conf or is_unknown):
            suggestion = suggestions[0]

        if suggestion and suggestion != original:
            out_parts.append(suggestion)
            corrections.append({
                "position": m.start(),
                "token": original,
                "suggestion": suggestion,
                "confidence": res.get("confidence", 0.0),
                "pos": res.get("pos", ""),
            })
            applied += 1
        else:
            out_parts.append(original)

    # дописываем хвост после последнего токена
    out_parts.append(text[last_end:])

    return {
        "original": text,
        "corrected": "".join(out_parts),
        "corrections": corrections,
        "stats": {"tokens": len(results), "corrected": applied},
    }
