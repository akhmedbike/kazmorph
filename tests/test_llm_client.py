"""
Тесты LLM-ревьюера (kazmorph/llm_review/client.py).

Anthropic SDK полностью подменяется — реальные API-вызовы не делаются.
Фокус на парсинге ответа (самое хрупкое место) и на расчёте стоимости.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazmorph.llm_review import client as llm_client
from kazmorph.llm_review.client import (
    LLMReviewer,
    ReviewResult,
    _parse_response,
)


# ============================================================
#  Фикстуры: фейковый Anthropic-клиент
# ============================================================


class _FakeUsage(SimpleNamespace):
    """Имитация resp.usage из Anthropic SDK."""


class _FakeResponse(SimpleNamespace):
    """Имитация объекта ответа messages.create."""


class _FakeMessages:
    """Записывает вызовы и возвращает предзаготовленный ответ."""

    def __init__(self, raw: str, in_tok: int = 100, out_tok: int = 50):
        self._raw = raw
        self._in_tok = in_tok
        self._out_tok = out_tok
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003 — имитация SDK-сигнатуры
        self.calls.append(kwargs)
        return _FakeResponse(
            content=[SimpleNamespace(text=self._raw)],
            usage=_FakeUsage(input_tokens=self._in_tok, output_tokens=self._out_tok),
        )


class _FakeAnthropic:
    """Подмена класса anthropic.Anthropic: отдаёт _FakeMessages."""

    instances: list[_FakeMessages] = []

    def __init__(self, **kwargs):  # noqa: ANN003
        # Не используем pre-set ответ здесь — тест задаёт его через _set_reply.
        self.kwargs = kwargs
        self.messages = _FakeMessages(raw="")
        _FakeAnthropic.instances.append(self.messages)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    @classmethod
    def last(cls) -> _FakeMessages:
        return cls.instances[-1]


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Подменить модуль anthropic так, чтобы Anthropic() возвращал фейк."""
    _FakeAnthropic.reset()
    monkeypatch.setattr(llm_client, "_reviewer_singleton", None, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-fake")
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    # monkeypatch модуля anthropic: LLMReviewer._make_client делает import
    # anthropic внутри метода — подменяем sys.modules.
    import sys

    fake_module = SimpleNamespace(Anthropic=_FakeAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    yield _FakeAnthropic
    _FakeAnthropic.reset()


def _make_reviewer(fake_anthropic, reply: str, **usage) -> LLMReviewer:
    """Создать LLMReviewer и задать ответ фейка."""
    r = LLMReviewer()
    fake_anthropic.last()._raw = reply
    if usage:
        m = fake_anthropic.last()
        m._in_tok = usage.get("in_tok", 100)
        m._out_tok = usage.get("out_tok", 50)
    return r


# ============================================================
#  Парсинг ответа (_parse_response) — чистая функция, без моков SDK
# ============================================================


def _parse(raw: str, *, in_tok: int = 100, out_tok: int = 50) -> ReviewResult:
    return _parse_response(
        raw,
        token="токен",
        provider="claude",
        model="claude-haiku-4-5",
        latency_ms=42,
        usage={"input_tokens": in_tok, "output_tokens": out_tok},
        price_in=1.0,
        price_out=5.0,
    )


class TestParseResponse:
    def test_correct_verdict(self):
        raw = json.dumps(
            {"verdict": "correct", "corrected_lemma": None, "corrected_pos": None,
             "explanation": "Разбор верный."}
        )
        r = _parse(raw)
        assert r.verdict == "correct"
        assert r.corrected_lemma is None
        assert r.corrected_pos is None
        assert r.explanation == "Разбор верный."
        assert r.parse_error is None

    def test_incorrect_with_correction(self):
        raw = json.dumps(
            {"verdict": "incorrect", "corrected_lemma": "құқық",
             "corrected_pos": "noun", "explanation": "Қ/ғ-чередование."}
        )
        r = _parse(raw)
        assert r.verdict == "incorrect"
        assert r.corrected_lemma == "құқық"
        # POS нормализуется к верхнему регистру.
        assert r.corrected_pos == "NOUN"

    def test_uncertain_verdict(self):
        raw = json.dumps(
            {"verdict": "uncertain", "corrected_lemma": None,
             "corrected_pos": None, "explanation": "Сложная аффиксная цепочка."}
        )
        r = _parse(raw)
        assert r.verdict == "uncertain"

    def test_markdown_fenced_json(self):
        # Модель часто оборачивает JSON в ```json ... ``` — должно вытащить.
        raw = 'Какой-то текст ```json\n{"verdict": "correct", "corrected_lemma": null, "corrected_pos": null, "explanation": "ок"}\n``` ещё текст'
        r = _parse(raw)
        assert r.verdict == "correct"
        assert r.parse_error is None  # fallback-парсинг через регэксп сработал чисто

    def test_invalid_verdict_value(self):
        raw = json.dumps({"verdict": "maybe", "explanation": "хз"})
        r = _parse(raw)
        assert r.verdict == "uncertain"  # безопасный дефолт
        assert r.parse_error == "invalid_verdict:'maybe'"

    def test_malformed_json(self):
        r = _parse("{ это не json }}}")
        assert r.verdict == "uncertain"
        assert r.parse_error is not None
        assert "json" in r.parse_error.lower()

    def test_empty_response(self):
        r = _parse("")
        assert r.verdict == "uncertain"
        assert r.parse_error == "empty_response"

    def test_correct_verdict_ignores_correction_fields(self):
        # Если verdict != incorrect, поля исправления игнорируются (модель
        # иногда копирует туда pipeline-разбор — это не правка).
        raw = json.dumps(
            {"verdict": "correct", "corrected_lemma": "что-то", "corrected_pos": "NOUN"}
        )
        r = _parse(raw)
        assert r.verdict == "correct"
        assert r.corrected_lemma is None
        assert r.corrected_pos is None

    def test_incorrect_strips_empty_correction(self):
        raw = json.dumps(
            {"verdict": "incorrect", "corrected_lemma": "  ", "corrected_pos": ""}
        )
        r = _parse(raw)
        assert r.verdict == "incorrect"
        assert r.corrected_lemma is None  # пустая строка → None
        assert r.corrected_pos is None

    def test_cost_calculation(self):
        # 100 in × $1/M + 50 out × $5/M = 0.0001 + 0.00025 = 0.00035
        r = _parse('{"verdict":"correct"}', in_tok=100, out_tok=50)
        assert r.cost_usd == pytest.approx(0.00035, rel=1e-6)

    def test_cost_zero_when_no_usage(self):
        r = _parse('{"verdict":"correct"}', in_tok=0, out_tok=0)
        assert r.cost_usd == 0.0


# ============================================================
#  Интеграция: LLMReviewer + фейковый Anthropic SDK
# ============================================================


class TestLLMReviewer:
    def test_review_token_correct(self, fake_anthropic):
        reply = json.dumps({"verdict": "correct", "explanation": "Верно."})
        reviewer = _make_reviewer(fake_anthropic, reply)
        result = reviewer.review_token(
            sentence="Адам келді.", token="келді",
            pipeline_lemma="кел", pipeline_pos="VERB",
        )
        assert result.verdict == "correct"
        assert result.provider == "claude"
        assert result.model == "claude-haiku-4-5"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd > 0

    def test_review_token_incorrect_normalizes_pos(self, fake_anthropic):
        reply = json.dumps(
            {"verdict": "incorrect", "corrected_lemma": "құқық",
             "corrected_pos": "noun", "explanation": "чередование"}
        )
        reviewer = _make_reviewer(fake_anthropic, reply)
        result = reviewer.review_token(
            sentence="...құқығы...", token="құқығы",
            pipeline_lemma="құқы", pipeline_pos="NOUN",
        )
        assert result.verdict == "incorrect"
        assert result.corrected_lemma == "құқық"
        assert result.corrected_pos == "NOUN"

    def test_prompt_contains_sentence_and_token(self, fake_anthropic):
        reviewer = _make_reviewer(fake_anthropic, '{"verdict":"correct"}')
        reviewer.review_token(
            sentence="Менің атым Максим.", token="Максим",
            pipeline_lemma="Максим", pipeline_pos="PROPN",
        )
        sent_payload = fake_anthropic.last().calls[-1]
        user_msg = sent_payload["messages"][0]["content"]
        assert "Менің атым Максим." in user_msg
        assert "Максим" in user_msg

    def test_stats_accumulate(self, fake_anthropic):
        reviewer = _make_reviewer(fake_anthropic, '{"verdict":"correct"}')
        for _ in range(3):
            reviewer.review_token("с", "т", "л", "NOUN")
        assert reviewer.stats.n_calls == 3
        assert reviewer.stats.verdicts["correct"] == 3
        assert reviewer.stats.total_input_tokens == 300
        assert reviewer.stats.total_cost_usd > 0

    def test_stats_record_parse_error(self, fake_anthropic):
        reviewer = _make_reviewer(fake_anthropic, "не json вообще")
        reviewer.review_token("с", "т", "л", "NOUN")
        assert reviewer.stats.n_errors == 1
        assert reviewer.stats.verdicts["uncertain"] == 1

    def test_max_tokens_in_request(self, fake_anthropic):
        reviewer = _make_reviewer(fake_anthropic, '{"verdict":"correct"}')
        reviewer.review_token("с", "т", "л", "NOUN")
        call = fake_anthropic.last().calls[-1]
        assert call["max_tokens"] == llm_client.MAX_OUTPUT_TOKENS
        assert call["model"] == "claude-haiku-4-5"

    def test_system_prompt_present(self, fake_anthropic):
        reviewer = _make_reviewer(fake_anthropic, '{"verdict":"correct"}')
        reviewer.review_token("с", "т", "л", "NOUN")
        call = fake_anthropic.last().calls[-1]
        assert "expert" in call["system"].lower() or "эксперт" in call["system"].lower()
        # Промпт упоминает все три вердикта и казахские чередования.
        assert "uncertain" in call["system"]
        assert "қ↔ғ" in call["system"]


# ============================================================
#  Конфигурация провайдера / ошибки инициализации
# ============================================================


class TestProviderConfig:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "claude")
        llm_client.reset_reviewer()
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            LLMReviewer()

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(ValueError, match="Неизвестный LLM_PROVIDER"):
            LLMReviewer()

    def test_glm_provider_not_implemented(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "x")
        monkeypatch.setenv("LLM_PROVIDER", "glm")
        with pytest.raises(NotImplementedError, match="ZAI_API_KEY"):
            LLMReviewer()

    def test_custom_model_override(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("LLM_REVIEW_MODEL", "claude-sonnet-5")
        llm_client.reset_reviewer()
        reviewer = LLMReviewer()
        assert reviewer.model == "claude-sonnet-5"

    def test_get_reviewer_singleton(self, fake_anthropic):
        llm_client.reset_reviewer()
        a = llm_client.get_reviewer()
        b = llm_client.get_reviewer()
        assert a is b


# ============================================================
#  review_sentence_tokens: несколько токенов одного предложения
# ============================================================


class TestReviewSentenceTokens:
    def test_returns_one_result_per_token(self, fake_anthropic):
        reviewer = _make_reviewer(fake_anthropic, '{"verdict":"correct"}')
        tokens = [
            {"token": "Адам", "lemma": "адам", "pos": "NOUN", "confidence": 0.5},
            {"token": "келді", "lemma": "кел", "pos": "VERB", "confidence": 0.6},
        ]
        results = reviewer.review_sentence_tokens("Адам келді.", tokens)
        assert len(results) == 2
        assert all(r.verdict == "correct" for r in results)
        assert reviewer.stats.n_calls == 2


# ============================================================
#  Локальный провайдер Ollama (OpenAI-совместимый endpoint)
# ============================================================


class _FakeUsageOA(SimpleNamespace):
    """Имитация resp.usage из OpenAI SDK."""


class _FakeMessageOA(SimpleNamespace):
    pass


class _FakeChoiceOA(SimpleNamespace):
    pass


class _FakeRespOA(SimpleNamespace):
    """Имитация объекта ответа chat.completions.create."""


class _FakeCompletions:
    """Записывает вызовы и возвращает предзаготовленный ответ."""

    def __init__(self, raw: str, in_tok: int = 120, out_tok: int = 40):
        self._raw = raw
        self._in_tok = in_tok
        self._out_tok = out_tok
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003 — имитация SDK-сигнатуры
        self.calls.append(kwargs)
        return _FakeRespOA(
            choices=[_FakeChoiceOA(message=_FakeMessageOA(content=self._raw))],
            usage=_FakeUsageOA(prompt_tokens=self._in_tok, completion_tokens=self._out_tok),
        )


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions(raw="")


class _FakeOpenAI:
    """Подмена класса openai.OpenAI: отдаёт _FakeChat."""

    instances: list[_FakeOpenAI] = []

    def __init__(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        self.chat = _FakeChat()
        _FakeOpenAI.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    @classmethod
    def last(cls) -> _FakeOpenAI:
        return cls.instances[-1]


@pytest.fixture
def fake_openai(monkeypatch):
    """Подменить модуль openai так, чтобы OpenAI() возвращал фейк."""
    _FakeOpenAI.reset()
    monkeypatch.setattr(llm_client, "_reviewer_singleton", None, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # monkeypatch модуля openai: _make_client делает import openai внутри —
    # подменяем sys.modules.
    import sys

    fake_module = SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    yield _FakeOpenAI
    _FakeOpenAI.reset()


def _make_reviewer_ollama(fake_openai, reply: str, **usage) -> LLMReviewer:
    """Создать LLMReviewer (провайдер ollama) и задать ответ фейка."""
    r = LLMReviewer()
    c = fake_openai.last().chat.completions
    c._raw = reply
    if usage:
        c._in_tok = usage.get("in_tok", 120)
        c._out_tok = usage.get("out_tok", 40)
    return r


class TestOllamaProvider:
    def test_creates_without_api_key(self, fake_openai):
        # Локальный провайдер не требует ключа — RuntimeError быть не должно.
        reviewer = LLMReviewer()
        assert reviewer.provider == "ollama"
        assert reviewer.model == "qwen2.5:7b"  # дефолт

    def test_uses_openai_chat_completions(self, fake_openai):
        reviewer = _make_reviewer_ollama(
            fake_openai, '{"verdict":"correct","explanation":"ок"}'
        )
        result = reviewer.review_token("с", "т", "л", "NOUN")
        # Вызов шёл через chat.completions.create (openai-путь).
        call = fake_openai.last().chat.completions.calls[-1]
        assert call["model"] == "qwen2.5:7b"
        assert call["max_tokens"] == llm_client.MAX_OUTPUT_TOKENS
        # System + user roles присутствуют.
        roles = [m["role"] for m in call["messages"]]
        assert roles == ["system", "user"]

    def test_parses_verdict_and_zero_cost(self, fake_openai):
        # Локальная модель — цена 0, cost_usd всегда 0.
        reviewer = _make_reviewer_ollama(
            fake_openai,
            '{"verdict":"incorrect","corrected_lemma":"құқық",'
            '"corrected_pos":"noun","explanation":"чередование"}',
        )
        result = reviewer.review_token("...құқығы...", "құқығы", "құқы", "NOUN")
        assert result.verdict == "incorrect"
        assert result.corrected_lemma == "құқық"
        assert result.corrected_pos == "NOUN"
        assert result.provider == "ollama"
        assert result.cost_usd == 0.0
        assert result.input_tokens == 120
        assert result.output_tokens == 40

    def test_base_url_passed_to_client(self, fake_openai, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-host:1234/v1")
        llm_client.reset_reviewer()
        LLMReviewer()
        kwargs = _FakeOpenAI.last().kwargs
        assert kwargs["base_url"] == "http://my-host:1234/v1"

    def test_custom_model_override(self, fake_openai, monkeypatch):
        monkeypatch.setenv("LLM_REVIEW_MODEL", "qwen2.5:14b")
        llm_client.reset_reviewer()
        reviewer = LLMReviewer()
        assert reviewer.model == "qwen2.5:14b"
