"""
Нагрузочный тест стабильности работы KazMorph (ТЗ, ЧАСТЬ C, Этап 5:
«система для тестирования», «стабильная работа при реальной нагрузке»).

Отправляет конкурентные запросы к /api/analyze и измеряет:
  - успешность (доля ответов HTTP 200);
  - латентность (min / avg / p50 / p95 / p99 / max);
  - пропускную способность (запросов в секунду);
  - стабильность при разной нагрузке (нарастающие уровни конкуренции).

Использование:
    python scripts/load_test.py --host 127.0.0.1 --port 8090 \
        --total 200 --concurrency 10
    python scripts/load_test.py --ramp   # нарастающая нагрузка 1→2→5→10→20
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TEXT = (
    "Қазақстан Республикасының Конституциясы бойынша әрбір адамның "
    "өмір сүруге, жеке басының бостандығына және қорғалуына құқығы бар."
)


def _percentile(values: list[float], pct: float) -> float:
    """Перцентиль (0..100) без внешних зависимостей."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _do_request(url: str, text: str, timeout: float) -> tuple[bool, float]:
    """Один запрос. Возвращает (успех, latency_мс)."""
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            ok = resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        ok = False
    return ok, (time.perf_counter() - t0) * 1000.0


def run_load(
    url: str,
    text: str,
    total: int,
    concurrency: int,
    timeout: float = 30.0,
) -> dict:
    """
    Прогнать total запросов с уровнем конкуренции concurrency.
    Возвращает сводку результатов.
    """
    latencies: list[float] = []
    success = 0
    errors = 0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_do_request, url, text, timeout) for _ in range(total)]
        for fut in as_completed(futures):
            ok, lat = fut.result()
            latencies.append(lat)
            if ok:
                success += 1
            else:
                errors += 1
    wall = time.perf_counter() - t_start

    return {
        "concurrency": concurrency,
        "total": total,
        "success": success,
        "errors": errors,
        "success_rate": success / total if total else 0.0,
        "wall_time_s": wall,
        "rps": total / wall if wall else 0.0,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
    }


def _print_result(r: dict) -> None:
    print(f"  Конкуренция {r['concurrency']:>3} | запросов {r['total']:>4} | "
          f"успех {r['success_rate']*100:5.1f}% ({r['success']}/{r['total']}) | "
          f"{r['rps']:6.1f} req/s | "
          f"latency avg={r['latency_ms']['avg']:6.1f}мс "
          f"p95={r['latency_ms']['p95']:7.1f}мс "
          f"max={r['latency_ms']['max']:8.1f}мс"
          + (f" | ошибок: {r['errors']}" if r['errors'] else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Нагрузочный тест KazMorph")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Текст для анализа")
    parser.add_argument("--total", type=int, default=200, help="Всего запросов")
    parser.add_argument("--concurrency", type=int, default=10, help="Уровень конкуренции")
    parser.add_argument("--timeout", type=float, default=30.0, help="Таймаут запроса, с")
    parser.add_argument("--ramp", action="store_true",
                        help="Нарастающая нагрузка: 1→2→5→10→20")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/api/analyze"

    # Проверка готовности
    try:
        with urllib.request.urlopen(
            f"http://{args.host}:{args.port}/api/health", timeout=5
        ) as resp:
            health = json.loads(resp.read())
    except Exception as e:
        print(f"Ошибка: сервер на {args.host}:{args.port} недоступен ({e})")
        sys.exit(1)

    print("=" * 78)
    print("  KazMorph — нагрузочный тест стабильности (Этап 5)")
    print("=" * 78)
    print(f"  URL: {url}")
    print(f"  Этапы: {', '.join(health.get('stages', []))}")
    print(f"  Текст: {args.text[:50]}...")
    print()

    levels = [1, 2, 5, 10, 20] if args.ramp else [args.concurrency]
    results = []
    for c in levels:
        r = run_load(url, args.text, args.total, c, args.timeout)
        results.append(r)
        _print_result(r)

    # Итог
    print()
    print("-" * 78)
    total_req = sum(r["total"] for r in results)
    total_err = sum(r["errors"] for r in results)
    overall_rate = (total_req - total_err) / total_req if total_req else 0.0
    print(f"  ИТОГО: {total_req} запросов, успех {overall_rate*100:.2f}%, "
          f"ошибок {total_err}")
    status = "СТАБИЛЬНА" if total_err == 0 else "ЕСТЬ ОШИБКИ"
    print(f"  Стабильность: {status}")
    print("=" * 78)

    # JSON для включения в отчёт
    print("\n[JSON для отчёта]")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
