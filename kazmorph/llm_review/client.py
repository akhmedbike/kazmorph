"""
Клиент LLM-ревьюера: проверяет отдельный токен/разбор через Anthropic API.

Точка входа — ``review_token(...)`` (функциональный стиль, для скриптов)
или ``LLMReviewer`` (класс, переиспользует HTTP-клиент между вызовами).

Дизайн:
  • Промпт требует строго JSON-ответ с полем verdict ∈ {correct, incorrect,
    uncertain}. Это трёхсторонний вердикт — explicit "uncertain" защищает от
    галлюцинаций LLM в low-resource казахском (риск №1 из idea-документа).
  • Парсинг устойчив к markdown-fencing (модели часто оборачивают JSON в
    ```json ... ```), к лишнему тексту до/после, и к сломанному JSON — в
    последнем случае возвращается verdict=uncertain, чтобы не молча
    проглотить мусор в обучение.
  • Стоимость оценивается из usage-полей ответа и тарифов модели; пишется в
    ReviewResult для логирования расходов (риск №4 — не удивиться счёту).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Провайдеры и модели ─────────────────────────────────────────────────────
# Claude Haiku 4.5: $1/$5 за млн токенов (in/out), 200k контекст — оптимальный
# выбор для cost-sensitive классификации разборов. Aliases подтверждены в
# docs.anthropic.com (2026).
PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    "claude": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5",
        # USD за 1 млн токенов (input, output) — для оценки расхода.
        "price_in_per_mtok": 1.0,
        "price_out_per_mtok": 5.0,
    },
    # Заглушка: подключается при появлении отдельного pay-per-token ключа
    # ZAI_API_KEY (НЕ GLM Coding Plan). Anthropic-совместимый ли endpoint
    # standalone Z.ai API — проверяется отдельно при активации.
    "glm": {
        "api_key_env": "ZAI_API_KEY",
        "default_model": "glm-4.5",
        "price_in_per_mtok": 0.5,
        "price_out_per_mtok": 0.5,
    },
}

VALID_VERDICTS = frozenset({"correct", "incorrect", "uncertain"})

# Максимальное число токенов в ответе. JSON-вердикт укладывается в ~150
# токенов с объяснением; запас на случай многословия модели.
MAX_OUTPUT_TOKENS = 512

# Промпт ревьюера. Формулировка «источник официальный, орфография корректна» —
# из idea-документа: ценность не в поиске опечаток, а в проверке, где сам
# анализатор не уверен. Явное «не гадай, скажи uncertain» — защита от
# уверенных галлюцинаций на агглютинативных формах.
_REVIEW_SYSTEM = """\
Ты — эксперт по казахской морфологии (Universal Dependencies). Тебе дано \
предложение и один токен из него, а также разбор, предложенный автомати-\
ческим анализатором. Источник текста — официальный (государственный сайт), \
орфография предполагается корректной.

Проверь разбор анализатора (лемма и часть речи, UPOS). Если разбор верен — \
подтверди его. Если неверен — дай исправленную пару лемма/UPOS и коротко \
(1 предложение) объясни ошибку. Если ты не уверен — так и скажи; не гадай, \
особенно на агглютинативных формах с длинными аффиксными цепочками и \
чередованиями согласных (қ↔ғ, к↔г, б↔п, д↔т).

UPOS-алфавит (используй только его): NOUN, VERB, ADJ, ADV, PRON, DET, NUM, \
ADP, CCONJ, SCONJ, PART, INTJ, AUX, PROPN, PUNCT, SYM, X.

Ответ СТРОГО в JSON, без markdown-обёртки и лишнего текста:
{"verdict": "correct" | "incorrect" | "uncertain",
 "corrected_lemma": "...",   # только при verdict=incorrect; иначе null
 "corrected_pos": "...",     # только при verdict=incorrect; иначе null
 "explanation": "..."}       # 1 предложение
"""


@dataclass
class ReviewResult:
    """Результат проверки одного токена ревьюером."""

    token: str
    verdict: str  # correct | incorrect | uncertain
    corrected_lemma: str | None = None
    corrected_pos: str | None = None
    explanation: str = ""
    # Метаданные для логирования и аналитики.
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    # Флаги проблем парсинга — полезны для отладки промпта.
    parse_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "verdict": self.verdict,
            "corrected_lemma": self.corrected_lemma,
            "corrected_pos": self.corrected_pos,
            "explanation": self.explanation,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "parse_error": self.parse_error,
        }


@dataclass
class _Stats:
    """Накопительная статистика по сессии вызовов (для отчёта запуска)."""

    n_calls: int = 0
    n_errors: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    verdicts: dict[str, int] = field(default_factory=lambda: {"correct": 0, "incorrect": 0, "uncertain": 0})


class LLMReviewer:
    """Ревьюер с переиспользуемым HTTP-клиентом. Один экземпляр на запуск."""

    def __init__(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        provider = (provider or os.getenv("LLM_PROVIDER", "claude")).lower()
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Неизвестный LLM_PROVIDER={provider!r}. "
                f"Доступно: {sorted(PROVIDER_CONFIG)}"
            )
        cfg = PROVIDER_CONFIG[provider]
        self.provider = provider
        self.model = model or os.getenv("LLM_REVIEW_MODEL") or cfg["default_model"]
        self._price_in = cfg["price_in_per_mtok"]
        self._price_out = cfg["price_out_per_mtok"]
        key_env = cfg["api_key_env"]
        self.api_key = api_key or os.environ.get(key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"Не задан {key_env}. Добавьте ключ в .env "
                f"(провайдер={provider})."
            )
        self.timeout = timeout
        self.stats = _Stats()
        # Ленивый импорт: если anthropic не установлен, модуль всё равно
        # импортируется (нужно для чтения ReviewResult в других местах).
        self._client = self._make_client()

    def _make_client(self):
        if self.provider == "claude":
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Пакет anthropic не установлен. Установите: pip install anthropic"
                ) from exc
            return anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        if self.provider == "glm":
            # TODO: активировать при появлении ZAI_API_KEY (pay-per-token).
            # Пока — явная ошибка, чтобы не дойти до продакшена с заглушкой.
            raise NotImplementedError(
                "GLM-провайдер не активирован. Нужен отдельный pay-per-token "
                "ключ ZAI_API_KEY (не GLM Coding Plan) и сверка совместимости "
                "endpoint'а с docs.z.ai."
            )
        raise ValueError(f"Провайдер {self.provider!r} не реализован")

    # ── Публичный API ───────────────────────────────────────────────────────

    def review_token(
        self,
        sentence: str,
        token: str,
        pipeline_lemma: str,
        pipeline_pos: str,
        *,
        difficulty_class: str | None = None,
        confidence: float | None = None,
    ) -> ReviewResult:
        """Проверить один токен и его разбор. Возвращает ReviewResult.

        sentence      — предложение целиком (контекст для LLM).
        token         — проверяемый токен (должен встречаться в sentence).
        pipeline_*    — разбор, выданный основным пайплайном.
        """
        user_prompt = _build_user_prompt(
            sentence=sentence,
            token=token,
            pipeline_lemma=pipeline_lemma,
            pipeline_pos=pipeline_pos,
            difficulty_class=difficulty_class,
            confidence=confidence,
        )
        t0 = time.monotonic()
        raw, usage = self._call_llm(user_prompt)
        latency_ms = int((time.monotonic() - t0) * 1000)
        result = _parse_response(
            raw,
            token=token,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            usage=usage,
            price_in=self._price_in,
            price_out=self._price_out,
        )
        self._record(result, usage)
        return result

    def review_sentence_tokens(
        self,
        sentence: str,
        tokens: list[dict[str, Any]],
    ) -> list[ReviewResult]:
        """Проверить набор токенов одного предложения.

        Сейчас вызывает LLM по разу на токен (последовательные запросы).
        Пакетная оптимизация (несколько токенов в одном вызове) оставлена
        на потом — она усложняет парсинг и валидацию, а при ~50–150
        кандидатах в день стоимость и так копеечная.
        """
        out: list[ReviewResult] = []
        for t in tokens:
            out.append(
                self.review_token(
                    sentence=sentence,
                    token=t["token"],
                    pipeline_lemma=t.get("lemma", ""),
                    pipeline_pos=t.get("pos", ""),
                    difficulty_class=t.get("difficulty_class"),
                    confidence=t.get("confidence"),
                )
            )
        return out

    # ── Внутреннее ──────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> tuple[str, dict[str, int]]:
        """Один вызов API. Возвращает (сырой текст ответа, usage)."""
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=_REVIEW_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Извлекаем текст из content-блоков.
        text_parts = []
        for block in resp.content:
            # Anthropic SDK: text-блок имеет .text, у других типов его нет.
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
        raw = "".join(text_parts).strip()
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
        }
        return raw, usage

    def _record(self, result: ReviewResult, usage: dict[str, int]) -> None:
        self.stats.n_calls += 1
        self.stats.total_cost_usd += result.cost_usd
        self.stats.total_input_tokens += usage.get("input_tokens", 0)
        self.stats.total_output_tokens += usage.get("output_tokens", 0)
        if result.parse_error:
            self.stats.n_errors += 1
        v = result.verdict if result.verdict in VALID_VERDICTS else "uncertain"
        self.stats.verdicts[v] = self.stats.verdicts.get(v, 0) + 1


# ── Промпт и парсинг ────────────────────────────────────────────────────────


def _build_user_prompt(
    *,
    sentence: str,
    token: str,
    pipeline_lemma: str,
    pipeline_pos: str,
    difficulty_class: str | None,
    confidence: float | None,
) -> str:
    """Собрать user-сообщение из данных одного токена."""
    conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "н/д"
    diff_str = difficulty_class or "н/д"
    return (
        f"Предложение: {sentence}\n"
        f"Токен: {token}\n"
        f"Разбор анализатора: лемма={pipeline_lemma}, POS={pipeline_pos}, "
        f"уверенность={conf_str}, класс_сложности={diff_str}\n\n"
        f"Проверь разбор. Ответ в JSON."
    )


# Регэксп для извлечения JSON-объекта из ответа, который модель могла
# обернуть в ```json ... ``` или сопроводить лишним текстом.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(
    raw: str,
    *,
    token: str,
    provider: str,
    model: str,
    latency_ms: int,
    usage: dict[str, int],
    price_in: float,
    price_out: float,
) -> ReviewResult:
    """Распарсить сырой ответ LLM в ReviewResult с защитой от мусора."""
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = (in_tok * price_in + out_tok * price_out) / 1_000_000

    base = dict(
        token=token,
        verdict="uncertain",  # безопасный дефолт при любой ошибке парсинга
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )

    if not raw:
        return ReviewResult(**base, parse_error="empty_response")

    # Попытка 1: весь ответ — валидный JSON.
    obj: dict[str, Any] | None = None
    parse_err: str | None = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Попытка 2: вытащить первый {...} блок.
        m = _JSON_OBJECT_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                parse_err = f"json_extract_failed: {exc}"
        else:
            parse_err = "no_json_object_found"

    if obj is None:
        return ReviewResult(**base, parse_error=parse_err)

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        return ReviewResult(
            **base,
            parse_error=f"invalid_verdict:{verdict!r}",
            explanation=str(obj.get("explanation", "")),
        )

    corrected_lemma = obj.get("corrected_lemma")
    corrected_pos = obj.get("corrected_pos")
    # Нормализация: при verdict != incorrect поля исправления игнорируем
    # (модель иногда пишет туда копию pipeline-разбора — это не правка).
    if verdict != "incorrect":
        corrected_lemma = None
        corrected_pos = None
    else:
        # Приводим pos к верхнему регистру и проверяем.
        if isinstance(corrected_pos, str):
            corrected_pos = corrected_pos.strip().upper() or None
        if isinstance(corrected_lemma, str):
            corrected_lemma = corrected_lemma.strip() or None

    explanation = str(obj.get("explanation", "")).strip()
    return ReviewResult(
        token=token,
        verdict=verdict,
        corrected_lemma=corrected_lemma,
        corrected_pos=corrected_pos,
        explanation=explanation,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
        # None при чистом json.loads; иначе — пометка, что сработал
        # fallback-парсинг (полезно для отладки промпта).
        parse_error=parse_err,
    )


# ── Фабрика и функциональная точка входа ───────────────────────────────────

_reviewer_singleton: LLMReviewer | None = None


def get_reviewer() -> LLMReviewer:
    """Вернуть (и закэшировать) синглтон-ревьюер из настроек окружения.

    Используется скриптами (llm_review.py, poc_llm_eval.py): создаётся один
    HTTP-клиент на весь запуск.
    """
    global _reviewer_singleton
    if _reviewer_singleton is None:
        _reviewer_singleton = LLMReviewer()
    return _reviewer_singleton


def reset_reviewer() -> None:
    """Сбросить синглтон — нужно тестам, чтобы менять env между кейсами."""
    global _reviewer_singleton
    _reviewer_singleton = None


def review_token(
    sentence: str,
    token: str,
    pipeline_lemma: str,
    pipeline_pos: str,
    **kwargs: Any,
) -> ReviewResult:
    """Функциональный shortcut: get_reviewer().review_token(...)."""
    return get_reviewer().review_token(
        sentence=sentence,
        token=token,
        pipeline_lemma=pipeline_lemma,
        pipeline_pos=pipeline_pos,
        **kwargs,
    )
