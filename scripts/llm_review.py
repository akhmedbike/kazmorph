#!/usr/bin/env python3
"""
LLM-ревью отобранных кандидатов → таблица Annotation.

Читает data/candidates/{date}.jsonl (вывод select_candidates.py), для
каждого кандидата вызывает LLM-ревьюер (Claude Haiku) и пишет результат
в БД — таблица Annotation со status=pending. Дальше эти строки попадают
в админку /admin/review для ручного подтверждения.

ВАЖНО: скрипт НЕ пишет правки напрямую в retrain — все LLM-предложения
проходят через status=pending → человек в админке решает. Это защита от
риска №1 (некомпетентность LLM в казахском) и №2 (model collapse).

Запуск:
  ANTHROPIC_API_KEY=sk-... python scripts/llm_review.py
  ANTHROPIC_API_KEY=sk-... python scripts/llm_review.py --date 2026-08-04
  ANTHROPIC_API_KEY=sk-... python scripts/llm_review.py --max-candidates 10
  ANTHROPIC_API_KEY=sk-... python scripts/llm_review.py --dry-run  # без записи в БД
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kazmorph.llm_review.client import LLMReviewer, ReviewResult  # noqa: E402

logger = logging.getLogger("llm_review")

CANDIDATES_DIR = Path("data/candidates")
RUN_LOG_DIR = Path("data/llm_runs")
KZ_TZ = timezone(timedelta(hours=5))


# ============================================================
#  Чтение кандидатов
# ============================================================


def load_candidates(date_str: str | None = None) -> list[dict]:
    """Прочитать JSONL кандидатов за дату (или за сегодня по KZ tz)."""
    if date_str is None:
        date_str = datetime.now(KZ_TZ).strftime("%Y-%m-%d")
    path = CANDIDATES_DIR / f"{date_str}.jsonl"
    if not path.exists():
        logger.warning("Файл кандидатов не найден: %s", path)
        return []
    candidates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Пропуск строки (невалидный JSON): %s", exc)
    logger.info("Загружено %d кандидатов из %s", len(candidates), path)
    return candidates


# ============================================================
#  Запись в БД
# ============================================================


def init_database():
    """Инициализировать БД и вернуть фабрику сессий или None."""
    from app.db import init_db, get_session_factory
    from app.models import Annotation  # noqa: F401 — регистрирует модель
    if not init_db():
        logger.error("БД недоступна — задайте DATABASE_URL или используйте --dry-run")
        return None
    return get_session_factory()


def save_annotation(db, candidate: dict, result: ReviewResult) -> int:
    """Создать строку Annotation (status=pending). Возвращает id."""
    from app.models import Annotation
    ann = Annotation(
        # Происхождение.
        source_query_id=None,
        source_url=candidate.get("source_url", ""),
        fetched_at=datetime.now(KZ_TZ),
        # Контекст + разбор пайплайна.
        sentence=candidate.get("sentence", ""),
        token=candidate.get("token", ""),
        pipeline_lemma=candidate.get("lemma", ""),
        pipeline_pos=candidate.get("pos", ""),
        pipeline_confidence=candidate.get("confidence", 0.0),
        difficulty_class=candidate.get("difficulty_class", "NONE"),
        # Предложение LLM.
        llm_verdict=result.verdict,
        llm_lemma=result.corrected_lemma,
        llm_pos=result.corrected_pos,
        llm_explanation=result.explanation,
        llm_provider=result.provider,
        llm_cost_usd=result.cost_usd,
        # Человек ещё не решал.
        status="pending",
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return ann.id


# ============================================================
#  Главный цикл
# ============================================================


def run_review(
    date_str: str | None = None,
    *,
    max_candidates: int | None = None,
    sleep_between: float = 0.2,
    dry_run: bool = False,
    reviewer: LLMReviewer | None = None,
) -> dict:
    """Прогнать LLM по всем кандидатам за дату → Annotation (pending)."""
    candidates = load_candidates(date_str)
    if not candidates:
        return {"date": date_str, "n_candidates": 0, "n_reviewed": 0}
    if max_candidates:
        candidates = candidates[:max_candidates]

    if reviewer is None:
        reviewer = LLMReviewer()
    logger.info("LLM-ревьюер: provider=%s, model=%s", reviewer.provider, reviewer.model)

    session_factory = None if dry_run else init_database()
    if not dry_run and session_factory is None:
        raise RuntimeError("БД недоступна — используйте --dry-run для прогона без записи")

    db = None
    try:
        if session_factory:
            db = session_factory()
        n_reviewed = 0
        for i, cand in enumerate(candidates, 1):
            result = reviewer.review_token(
                sentence=cand.get("sentence", ""),
                token=cand.get("token", ""),
                pipeline_lemma=cand.get("lemma", ""),
                pipeline_pos=cand.get("pos", ""),
                difficulty_class=cand.get("difficulty_class"),
                confidence=cand.get("confidence"),
            )
            ann_id = None
            if db is not None:
                ann_id = save_annotation(db, cand, result)
            n_reviewed += 1
            if i % 10 == 0 or i == len(candidates):
                logger.info(
                    "  ...%d/%d (verdict=%s), накопленный cost ≈ $%.4f",
                    i, len(candidates), result.verdict, reviewer.stats.total_cost_usd,
                )
            if sleep_between > 0 and i < len(candidates):
                time.sleep(sleep_between)
    finally:
        if db is not None:
            db.close()

    # Лог запуска (даже при dry-run — для учёта расхода).
    run_log = {
        "date": date_str or datetime.now(KZ_TZ).strftime("%Y-%m-%d"),
        "n_candidates": len(candidates),
        "n_reviewed": n_reviewed,
        "dry_run": dry_run,
        "provider": reviewer.provider,
        "model": reviewer.model,
        "total_cost_usd": round(reviewer.stats.total_cost_usd, 6),
        "total_input_tokens": reviewer.stats.total_input_tokens,
        "total_output_tokens": reviewer.stats.total_output_tokens,
        "verdicts": reviewer.stats.verdicts,
        "n_parse_errors": reviewer.stats.n_errors,
        "generated_at": datetime.now(KZ_TZ).isoformat(),
    }
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"{run_log['date']}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    logger.info("Лог запуска: %s", log_path)
    return run_log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", default=None, help="Дата кандидатов (YYYY-MM-DD).")
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="Лимит кандидатов (для экономии на первом прогоне).")
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="Пауза между вызовами API (сек), защита от rate-limit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Не писать в БД, только вызвать LLM и залогировать расход.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = run_review(
        date_str=args.date,
        max_candidates=args.max_candidates,
        sleep_between=args.sleep,
        dry_run=args.dry_run,
    )
    print("\n" + "=" * 50)
    print("LLM-ревью кандидатов")
    print("=" * 50)
    print(f"  Дата:            {stats.get('date', '-')}")
    print(f"  Кандидатов:      {stats.get('n_candidates', 0)}")
    print(f"  Проверено:       {stats.get('n_reviewed', 0)}")
    print(f"  Стоимость:       ${stats.get('total_cost_usd', 0):.4f}")
    print(f"  Вердикты:        {stats.get('verdicts', {})}")
    print(f"  Ошибок парсинга: {stats.get('n_parse_errors', 0)}")
    if stats.get("dry_run"):
        print("  (dry-run — в БД не записано)")
    else:
        print(f"  → записано в Annotation (status=pending), ждут ревью в /admin")
    print("=" * 50)


if __name__ == "__main__":
    main()
