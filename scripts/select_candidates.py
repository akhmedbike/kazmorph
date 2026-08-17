#!/usr/bin/env python3
"""
Отбор low-confidence токенов из собранных статей.

Прогоняет каждую статью через Pipeline.process() и отбирает токены, где
пайплайн не уверен — это кандидаты для LLM-ревью. Кандидатом считается
токен, у которого:
    confidence < ACTIVE_LEARNING_THRESHOLD (0.70)
    ИЛИ difficulty_class != "NONE" (BOR/AFC/SEG/AGR — потенциальная ошибка)

К каждому кандидату прикладывается контекст — предложение целиком, в
котором он встречается (важно для LLM: разбор слова зависит от контекста).
Пайплайн не возвращает границы предложений, поэтому они восстанавливаются
здесь же по терминальной пунктуации (. ! ? » …).

Вывод: data/candidates/{date}.jsonl — по одной строке на кандидата:
    {sentence, token, lemma, pos, confidence, difficulty_class, source_url}

Запуск:
  python scripts/select_candidates.py                   # за сегодня
  python scripts/select_candidates.py --date 2026-08-04 # конкретная дата
  python scripts/select_candidates.py --max-articles 5  # лимит для теста
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kazmorph.pipeline import ACTIVE_LEARNING_THRESHOLD, Pipeline  # noqa: E402

logger = logging.getLogger("select_candidates")

RAW_NEWS_DIR = Path("data/raw_news")
CANDIDATES_DIR = Path("data/candidates")
KZ_TZ = timezone(timedelta(hours=5))

# Терминальная пунктуация для восстановления границ предложений.
# » — закрывающая кавычка цитаты; … — многоточие.
_SENT_END = {".", "!", "?", "»", "…"}


# ============================================================
#  Восстановление границ предложений (sentencer)
# ============================================================


def split_into_sentences(tokens: list[dict]) -> list[tuple[int, int]]:
    """Разбить список токенов (из Pipeline.process) на предложения.

    Возвращает список (start, end) — полуоткрытых индексов в tokens.
    Пайплайн не возвращает границы предложений, поэтому восстанавливаем
    их по терминальной пунктуации. Каждое предложение включает свой
    завершающий знак препинания.

    Это лёгкий эвристический sentencer — он не претендует на linguistic
    точность, его задача лишь дать LLM разумный контекст. Если токен
    остаётся без завершителя (последнее «предложение» в тексте), он
    всё равно попадает в своё окно.
    """
    if not tokens:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for i, tok in enumerate(tokens):
        # Токен-завершитель: содержит хотя бы один терминальный знак.
        if any(ch in _SENT_END for ch in tok.get("token", "")):
            spans.append((start, i + 1))
            start = i + 1
    # Хвост без завершителя — отдельное «предложение».
    if start < len(tokens):
        spans.append((start, len(tokens)))
    return spans


def sentence_text(tokens: list[dict], start: int, end: int) -> str:
    """Собрать текст предложения из диапазона токенов."""
    return " ".join(t["token"] for t in tokens[start:end])


def find_sentence_of(tokens: list[dict], spans: list[tuple[int, int]],
                     token_idx: int) -> tuple[int, int] | None:
    """Найти (start, end) предложения, содержащего token_idx."""
    for start, end in spans:
        if start <= token_idx < end:
            return start, end
    return None


# ============================================================
#  Отбор кандидатов из одной статьи
# ============================================================


def extract_candidates(
    tokens: list[dict],
    *,
    threshold: float = ACTIVE_LEARNING_THRESHOLD,
    source_url: str = "",
) -> list[dict]:
    """Отобрать low-confidence токены с контекстом предложения.

    Кандидат — токен, где confidence < threshold ИЛИ difficulty != NONE.
    Возвращает список словарей для JSONL.
    """
    spans = split_into_sentences(tokens)
    candidates = []
    for idx, tok in enumerate(tokens):
        conf = tok.get("confidence", 1.0)
        diff = tok.get("difficulty_class", "NONE")
        is_low_conf = conf < threshold
        is_hard = diff != "NONE"
        if not (is_low_conf or is_hard):
            continue
        span = find_sentence_of(tokens, spans, idx)
        if span is None:
            continue
        s_start, s_end = span
        candidates.append({
            "sentence": sentence_text(tokens, s_start, s_end),
            "token": tok["token"],
            "lemma": tok.get("lemma", ""),
            "pos": tok.get("pos", ""),
            "confidence": round(conf, 4),
            "difficulty_class": diff,
            "source_url": source_url,
        })
    return candidates


# ============================================================
#  Главная функция
# ============================================================


def build_pipeline() -> Pipeline:
    """Собрать pipeline с обученными моделями (как в main.py)."""
    crf = "models/crf_model.joblib" if os.path.exists("models/crf_model.joblib") else None
    neu = "models/neural_pos" if os.path.exists("models/neural_pos") else None
    lem = "models/neural_lemma" if os.path.exists("models/neural_lemma") else None
    return Pipeline(
        crf_model_path=crf,
        neural_model_path=neu,
        lemma_model_path=lem,
    )


def load_articles(date_str: str) -> list[dict]:
    """Загрузить все статьи за дату из data/raw_news/{date}/*.json."""
    day_dir = RAW_NEWS_DIR / date_str
    if not day_dir.exists():
        return []
    articles = []
    for path in sorted(day_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                articles.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Не удалось прочитать %s: %s", path, exc)
    return articles


def run_select(
    date_str: str | None = None,
    *,
    max_articles: int | None = None,
    threshold: float = ACTIVE_LEARNING_THRESHOLD,
    pipeline: Pipeline | None = None,
) -> dict:
    """Отобрать кандидатов за дату. Возвращает статистику + пишет JSONL."""
    if date_str is None:
        date_str = datetime.now(KZ_TZ).strftime("%Y-%m-%d")
    if pipeline is None:
        pipeline = build_pipeline()

    articles = load_articles(date_str)
    if max_articles:
        articles = articles[:max_articles]
    logger.info("Дата %s: загружено %d статей", date_str, len(articles))
    if not articles:
        logger.warning("Нет статей за %s — запустите harvest_sources.py", date_str)
        return {"date": date_str, "n_articles": 0, "n_candidates": 0, "n_tokens": 0}

    all_candidates: list[dict] = []
    n_tokens_total = 0
    for i, article in enumerate(articles, 1):
        text = article.get("text", "")
        if not text:
            continue
        tokens = pipeline.process(text)
        n_tokens_total += len(tokens)
        cands = extract_candidates(
            tokens, threshold=threshold, source_url=article.get("url", "")
        )
        all_candidates.extend(cands)
        if i % 5 == 0 or i == len(articles):
            logger.info("  обработано %d/%d статей, кандидатов накоплено %d",
                        i, len(articles), len(all_candidates))

    # Запись JSONL.
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CANDIDATES_DIR / f"{date_str}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for cand in all_candidates:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")
    logger.info("Кандидаты: %d (из %d токенов) → %s",
                len(all_candidates), n_tokens_total, out_path)

    return {
        "date": date_str,
        "n_articles": len(articles),
        "n_candidates": len(all_candidates),
        "n_tokens": n_tokens_total,
        "candidate_rate": round(len(all_candidates) / max(n_tokens_total, 1), 4),
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", default=None,
                    help="Дата статей (YYYY-MM-DD). По умолчанию сегодня (KZ tz).")
    ap.add_argument("--max-articles", type=int, default=None,
                    help="Лимит статей (для тестирования).")
    ap.add_argument("--threshold", type=float, default=ACTIVE_LEARNING_THRESHOLD,
                    help=f"Порог confidence (default {ACTIVE_LEARNING_THRESHOLD}).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = run_select(
        date_str=args.date,
        max_articles=args.max_articles,
        threshold=args.threshold,
    )
    print("\n" + "=" * 50)
    print("Отбор кандидатов")
    print("=" * 50)
    print(f"  Дата:           {stats['date']}")
    print(f"  Статей:         {stats['n_articles']}")
    print(f"  Токенов всего:  {stats['n_tokens']}")
    print(f"  Кандидатов:     {stats['n_candidates']}")
    if stats["n_tokens"]:
        print(f"  Доля кандидатов:{stats['candidate_rate']:.2%}  (conf<{args.threshold} OR diff!=NONE)")
    print(f"  Записано в:     {stats.get('out_path', '-')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
