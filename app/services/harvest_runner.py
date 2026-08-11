"""Фоновый запуск шагов Active Learning из админки.

Три шага пайплайна запускаются отсюда как асинхронные задачи (через
``anyio.to_thread``, без отдельных воркеров/подпроцессов):

  harvest → scripts/harvest_sources.run_harvest
  select  → scripts/select_candidates.run_select
  review  → scripts/llm_review.run_review       (тратит Claude-токены)

Один модульный трекер ``task_state`` хранит состояние последнего/текущего
запуска: шаги линейны по данным, одновременный запуск бессмысленен → при
``status=="running"`` новый запуск отвергается (409 в роуте).

Импорты функций скриптов — ленивые (внутри ``_run_*``), чтобы не тащить
feedparser/torch/anthropic при старте приложения и не зависеть от порядка
``sys.path``. Скрипты лежат в репо рядом и безопасно импортируются.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anyio

logger = logging.getLogger("al_runner")

STEPS = ("harvest", "select", "review")

# Казахстанская таймзона — та же, что в harvest_sources/select_candidates:
# даты папок raw_news и имён candidates/{date}.jsonl группируются по ней.
_KZ_TZ = timezone(timedelta(hours=5))


# Модульный синглтон состояния. per-process; при рестарте история теряется —
# постоянное логирование ведут сами скрипты (data/llm_runs/, log-файлы).
task_state: dict = {
    "status": "idle",        # idle | running | done | error
    "step": None,            # harvest | select | review
    "source": None,          # только для harvest: имя источника или None (все)
    "started_at": None,
    "finished_at": None,
    "result": None,          # dict-статистика (done) | str-сообщение (error)
}


class TaskError(Exception):
    """Ошибка запуска/выполнения шага."""


def get_task_state() -> dict:
    """Снимок состояния для UI/API."""
    return dict(task_state)


def start_task(step: str, source: str | None = None) -> None:
    """Запустить шаг в фоне.

    ``step`` — один из :data:`STEPS`. Для ``harvest`` параметр ``source``
    выбирает конкретный источник (или ``None`` — все); для select/review
    игнорируется (они работают по всем собранным за сегодня данным).
    Поднимает :class:`TaskError`, если уже есть running-задача или step неизвестен.
    """
    if step not in STEPS:
        raise TaskError(f"Неизвестный шаг {step!r}")
    if task_state["status"] == "running":
        raise TaskError("Шаг уже выполняется — дождитесь завершения")

    task_state.update(
        status="running",
        step=step,
        source=source if step == "harvest" else None,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        result=None,
    )
    asyncio.create_task(_run(step, source))
    logger.info("AL-шаг запущен: %s (source=%s)", step, source)


async def _run(step: str, source: str | None) -> None:
    """Диспетчер: запускает нужный ``_run_*`` в рабочем потоке."""
    handler = {
        "harvest": lambda: _run_harvest(source),
        "select": lambda: _run_select(),
        "review": lambda: _run_review(),
    }[step]
    try:
        await anyio.to_thread.run_sync(handler)
    except Exception as exc:  # noqa: BLE001
        logger.exception("AL-шаг %s провалился", step)
        task_state.update(
            status="error",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result=str(exc),
        )


# ============================================================
#  Шаги
# ============================================================


def _finish(stats: dict) -> None:
    """Записать успешный результат в трекер."""
    task_state.update(
        status="done",
        finished_at=datetime.now(timezone.utc).isoformat(),
        result=stats,
    )


def _run_harvest(source: str | None) -> None:
    """Шаг 1: сбор текстов из источников (RSS/HTML)."""
    from scripts.harvest_sources import run_harvest

    stats = run_harvest(source_filter=source)
    _finish(stats)
    logger.info("Harvest завершён: %s", stats.get("sources"))


def _run_select() -> None:
    """Шаг 2: отбор low-confidence токенов из собранных статей."""
    from scripts.select_candidates import run_select

    stats = run_select()
    _finish(stats)
    logger.info("Select завершён: кандидатов=%s", stats.get("n_candidates"))


def _run_review() -> None:
    """Шаг 3: LLM-ревью кандидатов → Annotation(pending) в БД.

    Тратит Claude-токены; вызывается только после подтверждения в UI (см.
    роут /admin/sources/candidates/preview с оценкой стоимости).
    """
    from scripts.llm_review import run_review

    stats = run_review()
    _finish(stats)
    logger.info(
        "Review завершён: обработано=%s, cost=$%s",
        stats.get("n_reviewed"), stats.get("total_cost_usd"),
    )


# ============================================================
#  Оценка для confirm-диалога перед LLM-шагом
# ============================================================


def count_candidates_today() -> dict:
    """Снять превью кандидатов за сегодня + оценку стоимости LLM-ревью.

    Возвращает ``{"n_candidates": int, "est_cost_usd": float, "date": str}``.

    Число кандидатов = строки в ``data/candidates/{сегодня}.jsonl``. Оценка
    стоимости — N × среднюю стоимость одного review из последнего лога в
    ``data/llm_runs/*.json`` (если логов ещё нет — fallback $0.0008/кандидат,
    типичная цена Claude Haiku на одно review с контекстом).
    """
    from scripts.select_candidates import CANDIDATES_DIR  # type: ignore[attr-defined]

    today = datetime.now(_KZ_TZ).strftime("%Y-%m-%d")
    jsonl = Path(CANDIDATES_DIR) / f"{today}.jsonl"
    n = 0
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())

    avg_cost = _last_avg_review_cost()
    return {
        "date": today,
        "n_candidates": n,
        "est_cost_usd": round(n * avg_cost, 6),
    }


def _last_avg_review_cost() -> float:
    """Средняя стоимость одного review из последнего прогона LLM.

    Нет логов → 0.0008 (эмпирическая оценка для Claude Haiku, см. докстринг
    ``count_candidates_today``).
    """
    runs_dir = Path("data/llm_runs")
    if not runs_dir.exists():
        return 0.0008
    logs = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs[:1]:
        try:
            data = json.loads(log.read_text(encoding="utf-8"))
            n = int(data.get("n_reviewed", 0))
            cost = float(data.get("total_cost_usd", 0.0))
            if n > 0:
                return cost / n
        except (json.JSONDecodeError, ValueError, OSError):
            continue
    return 0.0008
