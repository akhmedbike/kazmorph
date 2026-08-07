#!/usr/bin/env python3
"""
RSS/HTML харвестер для Active Learning.

Каждый прогон:
  1. Читает config/sources.yaml — список источников (inform.kz и др.).
  2. Для type=rss тянет ленту через feedparser; для type=html —
     requests + trafilatura (извлечение основного текста).
  3. Дедуплицирует по SHA256 текста (чтобы не гонять одну новость дважды
     — лента inform.kz держит 200 последних, многие перекрываются между
     прогонами).
  4. Сохраняет в data/raw_news/{YYYY-MM-DD}/{source}_{hash}.json с
     полями {text, source, url, title, fetched_at, expires_at}.

Юридическая защита: сырой текст хранится RAW_TEXT_RETENTION_DAYS (default 7),
затем автоматически удаляется (cleanup_old_files). Публикуется только
морфоразметка CoNLL-U, не сам текст — это снимает вопрос авторских прав
независимо от источника (см. idea-документ, Фаза 0).

Запуск:
  python scripts/harvest_sources.py                    # все источники
  python scripts/harvest_sources.py --source inform_kz # один источник
  python scripts/harvest_sources.py --cleanup-only     # только чистка старых
  python scripts/harvest_sources.py --dry-run          # без записи на диск
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("harvest_sources")

CONFIG_PATH = "config/sources.yaml"
OUTPUT_DIR = Path("data/raw_news")
SEEN_HASHES_PATH = Path("data/raw_news/.seen_hashes.json")
DEFAULT_RETENTION_DAYS = 7
# Таймзона Казахстана (Almaty/Astana, UTC+05, без перевода часов).
# Имена папок raw_news/{date} группируются по ней, чтобы «вечерняя сводка
# 4 августа» попала в 2026-08-04, а не в 2026-08-03 (как было бы по UTC).
KZ_TZ = timezone(timedelta(hours=5))


def local_today() -> str:
    """Текущая дата в таймзоне Казахстана (для имени папки)."""
    return datetime.now(KZ_TZ).strftime("%Y-%m-%d")


# ============================================================
#  Загрузка конфига источников
# ============================================================


def load_sources(config_path: str = CONFIG_PATH) -> list[dict]:
    """Прочитать config/sources.yaml → список описаний источников."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sources = cfg.get("sources", [])
    if not sources:
        raise ValueError(f"В {config_path} нет источников (поле sources пусто)")
    return sources


# ============================================================
#  Дедупликация по хэшу текста
# ============================================================


def text_hash(text: str) -> str:
    """SHA256 нормализованного текста (для дедупликации)."""
    norm = " ".join(text.split()).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def load_seen_hashes() -> set[str]:
    """Загрузить индекс уже gesehenных хэшей (для дедупликации между прогонами)."""
    if not SEEN_HASHES_PATH.exists():
        return set()
    try:
        with open(SEEN_HASHES_PATH, encoding="utf-8") as f:
            return set(json.load(f).get("hashes", []))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Не удалось прочитать %s: %s", SEEN_HASHES_PATH, exc)
        return set()


def save_seen_hashes(hashes: set[str]) -> None:
    """Сохранить индекс хэшей (атомарная запись)."""
    SEEN_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SEEN_HASHES_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"hashes": sorted(hashes), "updated_at": datetime.now(timezone.utc).isoformat()}, f)
    tmp.replace(SEEN_HASHES_PATH)


# ============================================================
#  Извлечение текста
# ============================================================


def _strip_html(html: str) -> str:
    """Грубая очистка HTML из RSS summary (удаляет теги, склеивая текст).

    feedparser возвращает summary с <p>, <a>, <br> и т.п. Для морфоанализа
    нужен чистый текст; тяжёлый HTML-парсер здесь избыточен — summary
    информагентства простой и предсказуемый.
    """
    import re

    # Удаляем теги, заменяя <br>/<p> на пробел/перенос.
    text = re.sub(r"<br\s*/?>", " ", html, flags=re.I)
    text = re.sub(r"</p>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    # Декодируем базовые HTML-сущности.
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'"))
    return " ".join(text.split()).strip()


def _fetch_url(url: str, *, timeout: float = 30.0) -> bytes:
    """Скачать URL как байты (с retry + exponential backoff).

    feedparser.parse(url) сам тянет ленту, но его сетевой слой нестабилен с
    kaz.inform.kz (то 200 записей, то 0 из-за bozo на кодировке). Поэтому
    качаем через urllib надёжно, а feedparser используем только для парсинга
    уже скачанных байтов — это разделение делает процесс детерминированным.
    """
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "kazmorph-research/1.0"})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning("  fetch %s: попытка %d провалена (%s), жду %ds", url, attempt + 1, exc, wait)
                time.sleep(wait)
    raise RuntimeError(f"Не удалось загрузить {url} после 3 попыток: {last_exc}")


def harvest_rss(source: dict, *, timeout: float = 30.0) -> list[dict]:
    """Собрать статьи из RSS-ленты одного источника.

    Возвращает список {title, url, text, published}. text уже очищен от HTML.
    """
    url = source["url"]
    max_articles = source.get("max_articles", 50)
    # Per-source порог минимальной длины текста. inform.kz отдаёт полные
    # тексты в summary (порог 200), а 24.kz — только короткие анонсы (87-150
    # символов); для него ставят min_text_length=100, чтобы пропускать
    # однострочные заголовки, но брать анонсы-абзацы.
    min_text_length = source.get("min_text_length", 200)
    logger.info("RSS: %s (%s) — запрос…", source["name"], url)
    # Качаем байты сами (надёжно), парсим через feedparser.
    raw_bytes = _fetch_url(url, timeout=timeout)
    parsed = feedparser.parse(raw_bytes)
    if parsed.bozo and parsed.bozo_exception:
        # bozo на кодировке (us-ascii vs utf-8) — не блокер, записи парсятся.
        logger.debug("  RSS %s: предупреждение парсера (не блокер): %s", source["name"], parsed.bozo_exception)
    entries = parsed.entries[:max_articles] if max_articles else parsed.entries
    logger.info("  RSS %s: получено %d записей (лимит %s)", source["name"], len(entries), max_articles)

    articles = []
    for entry in entries:
        # Текст: summary (есть всегда у inform.kz), fallback на title.
        raw_text = entry.get("summary") or entry.get("description") or ""
        text = _strip_html(raw_text) if raw_text else ""
        # Если summary слишком короткий, текст неполный — лучше пропустить,
        # чем гонять пайплайн по заголовку. Порог per-source (см. выше).
        if len(text) < min_text_length:
            continue
        articles.append({
            "title": entry.get("title", "").strip(),
            "url": entry.get("link", "").strip(),
            "text": text,
            "published": entry.get("published", ""),
        })
    logger.info("  RSS %s: годных статей (text≥%d символов): %d",
                source["name"], min_text_length, len(articles))
    return articles


def harvest_html(source: dict, *, timeout: float = 30.0) -> list[dict]:
    """Собрать статьи из HTML-источника через trafilatura.

    TODO: для akorda/primeminister/adilet — нужна логика обхода списка
    статей (каждый сайт своей структуры). Сейчас общая реализация:
    тянет ОДИН url (source['url']) и извлекает основной текст.
    Для реальных источников расширить списком URL или sitemap-парсингом.
    """
    try:
        import trafilatura
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("trafilatura не установлен: pip install trafilatura") from exc

    url = source["url"]
    logger.info("HTML: %s — запрос %s", source["name"], url)
    try:
        raw = _fetch_url(url, timeout=timeout)
    except RuntimeError as exc:
        logger.error("  HTML %s: %s", source["name"], exc)
        return []
    html = raw.decode("utf-8", errors="replace")

    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    if len(text) < 200:
        logger.warning("  HTML %s: извлечённый текст слишком короткий (%d символов)", source["name"], len(text))
        return []
    return [{"title": source.get("description", source["name"]), "url": url, "text": text, "published": ""}]


# ============================================================
#  Сохранение и очистка
# ============================================================


def save_article(article: dict, source_name: str, today: str, retention_days: int) -> Path | None:
    """Сохранить одну статью в JSON. Возвращает путь или None если путь пустой."""
    h = text_hash(article["text"])
    out_dir = OUTPUT_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_name}_{h}.json"
    fetched_at = datetime.now(KZ_TZ)
    expires_at = fetched_at + timedelta(days=retention_days)
    payload = {
        "source": source_name,
        "title": article["title"],
        "url": article["url"],
        "text": article["text"],
        "text_hash": h,
        "published": article.get("published", ""),
        "fetched_at": fetched_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "retention_days": retention_days,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def cleanup_old_files(retention_days: int | None = None) -> int:
    """Удалить файлы старше retention_days. Возвращает число удалённых.

    Защита по expires_at (записанному в каждом файле) и/или по возрасту файла.
    """
    if retention_days is None:
        retention_days = int(os.environ.get("RAW_TEXT_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
    if not OUTPUT_DIR.exists():
        return 0
    now = datetime.now(timezone.utc)
    removed = 0
    for json_path in OUTPUT_DIR.rglob("*.json"):
        if json_path.name.startswith("."):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
            expires_str = meta.get("expires_at")
            if expires_str:
                expires = datetime.fromisoformat(expires_str)
                if now > expires:
                    json_path.unlink()
                    removed += 1
                    continue
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # повреждённый файл — удаляем по возрасту ниже
        # Fallback: возраст файла.
        age_days = (now.timestamp() - json_path.stat().st_mtime) / 86400
        if age_days > retention_days:
            json_path.unlink()
            removed += 1
    if removed:
        logger.info("Очистка: удалено %d файлов старше %d дней", removed, retention_days)
    return removed


# ============================================================
#  Главный цикл
# ============================================================


def run_harvest(
    config_path: str | None = None,
    *,
    source_filter: str | None = None,
    dry_run: bool = False,
    retention_days: int | None = None,
) -> dict:
    """Собрать статьи со всех источников. Возвращает summary-статистику.

    config_path=None → читает модульный глобал CONFIG_PATH в момент вызова
    (важно для тестов, которые monkeypatch'ят hv.CONFIG_PATH; дефолт-аргумент
    захватил бы значение при определении функции и не увидел бы подмену).
    """
    if config_path is None:
        config_path = CONFIG_PATH
    if retention_days is None:
        retention_days = int(os.environ.get("RAW_TEXT_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
    sources = load_sources(config_path)
    if source_filter:
        sources = [s for s in sources if s["name"] == source_filter]
        if not sources:
            raise ValueError(f"Источник {source_filter!r} не найден в {config_path}")

    seen = load_seen_hashes() if not dry_run else set()
    today = local_today()
    stats = {"sources": [], "total_new": 0, "total_skipped_dup": 0, "dry_run": dry_run}

    for source in sources:
        try:
            if source["type"] == "rss":
                articles = harvest_rss(source)
            elif source["type"] == "html":
                articles = harvest_html(source)
            else:
                logger.warning("Источник %s: неизвестный type=%s, пропуск", source["name"], source["type"])
                continue
        except Exception as exc:  # noqa: BLE001
            logger.error("Источник %s: провал: %s", source["name"], exc)
            stats["sources"].append({"name": source["name"], "error": str(exc), "new": 0, "dup": 0})
            continue

        new_count = 0
        dup_count = 0
        for article in articles:
            h = text_hash(article["text"])
            if h in seen:
                dup_count += 1
                continue
            seen.add(h)
            new_count += 1
            if not dry_run:
                save_article(article, source["name"], today, retention_days)

        logger.info("Источник %s: новых=%d, дубликатов=%d", source["name"], new_count, dup_count)
        stats["sources"].append({"name": source["name"], "new": new_count, "dup": dup_count})
        stats["total_new"] += new_count
        stats["total_skipped_dup"] += dup_count

    if not dry_run:
        save_seen_hashes(seen)
        cleanup_old_files(retention_days)

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--source", default=None, help="Имя одного источника (иначе все).")
    ap.add_argument("--dry-run", action="store_true", help="Не записывать на диск, только статистика.")
    ap.add_argument("--cleanup-only", action="store_true", help="Только очистка старых файлов.")
    ap.add_argument("--retention-days", type=int, default=None,
                    help=f"Срок хранения сырого текста в днях (default {DEFAULT_RETENTION_DAYS}).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cleanup_only:
        n = cleanup_old_files(args.retention_days)
        print(f"Очистка завершена: удалено {n} файлов.")
        return

    stats = run_harvest(
        config_path=args.config,
        source_filter=args.source,
        dry_run=args.dry_run,
        retention_days=args.retention_days,
    )
    print("\n" + "=" * 50)
    print("Сводка харвеста")
    print("=" * 50)
    for s in stats["sources"]:
        if "error" in s:
            print(f"  {s['name']}: ОШИБКА — {s['error']}")
        else:
            print(f"  {s['name']}: новых={s['new']}, дубликатов={s['dup']}")
    print(f"  Итого: новых={stats['total_new']}, дубликатов={stats['total_skipped_dup']}")
    if stats["dry_run"]:
        print("  (dry-run — ничего не записано на диск)")
    print("=" * 50)


if __name__ == "__main__":
    main()
