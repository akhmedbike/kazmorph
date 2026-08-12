"""
LLM-ревьюер морфологических разборов (этап Active Learning).

Архитектурно это «teacher-модель» в схеме weak supervision: для токена,
в котором основной пайплайн (FST→CRF→Neural) не уверен, LLM выступает
вторым арбитром и предлагает либо подтверждение, либо исправленную пару
лемма/POS. Предложения LLM **не** идут в автообучение напрямую — они
попадают в админку для ручного подтверждения (см. app/routes/admin.py).

Поддерживаемые провайдеры (переключаются через LLM_PROVIDER):
  • claude — Anthropic API (claude-haiku-4-5 по умолчанию). Платный pay-per-
             token; пример внешнего ревьюера.
  • glm    — Z.ai standalone API. Заглушка; подключается при появлении
             отдельного pay-per-token ключа (не путать с GLM Coding Plan,
             который работает только внутри coding-инструментов).
  • ollama — локальный LLM через Ollama (OpenAI-совместимый endpoint /v1),
             по умолчанию qwen2.5:7b. Бесплатно, без внешних вызовов; ключ
             не требуется. OLLAMA_BASE_URL можно перенаправить на любой
             OpenAI-совместимый сервер (vLLM, LM Studio, llama.cpp).

Ключевая защита от галлюцинаций LLM в low-resource казахском — промпт
явно разрешает вердикт "uncertain", и такие ответы не должны идти в
авто-retrain, а показываются в админке с пометкой «низкая уверенность LLM».
"""

from kazmorph.llm_review.client import (
    ReviewResult,
    LLMReviewer,
    get_reviewer,
    review_token,
)

__all__ = ["ReviewResult", "LLMReviewer", "get_reviewer", "review_token"]
