"""
REST API v1 — интеграционные эндпоинты для сторонних сервисов (ТЗ, ЧАСТЬ B
Этап 4: единая интегрированная система; Этап 5: доступ пользователей).

Эндпоинты:
  POST /api/v1/correct        — исправить текст (бинарный корректор, главная фича)
  POST /api/v1/correct3       — трёхсторонний корректор KEEP/CORRECT/REVIEW
  POST /api/v1/analyze/batch  — пакетный анализ нескольких текстов
  POST /api/v1/evaluate       — оценить качество разметки (ЧАСТЬ A как сервис)
  GET  /api/v1/info           — метаданные сервиса (этапы, модели, лексикон)

Защита: если задана переменная окружения KAZMORPH_API_KEY, все эндпоинты
требуют заголовок X-API-Key. Локально (без переменной) — ключ не нужен.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from kazmorph.corrector import correct
from kazmorph.evaluation import QualityReport, parse_conllu_string
from kazmorph.pipeline import Pipeline, token_spans
from kazmorph.correction.corrector_3way import (
    correct_sentence_3way,
    TAU_E as _3W_TAU_E,
    TAU_C as _3W_TAU_C,
    TAU_M as _3W_TAU_M,
)

router = APIRouter(prefix="/api/v1", tags=["integration-v1"])

VERSION = "1.0.0"


# ============================================================
#  Зависимости
# ============================================================

def get_pipeline(request: Request) -> Pipeline:
    """
    Достать экземпляр Pipeline из app.state (инициализируется в lifespan).
    Объявлена как зависимость — переопределяется в тестах через
    app.dependency_overrides.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """
    Опциональная защита API-ключом.

    Если KAZMORPH_API_KEY не задана — защита отключена (локальная разработка).
    Если задана — требует совпадающего заголовка X-API-Key.
    """
    expected = os.environ.get("KAZMORPH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# ============================================================
#  Модели запросов/ответов
# ============================================================

class CorrectRequest(BaseModel):
    text: str
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Correct3Request(BaseModel):
    """Запрос к трёхстороннему корректору KEEP/CORRECT/REVIEW."""
    text: str
    tau_e: float = Field(default=_3W_TAU_E, ge=0.0, le=1.0,
                         description="Порог p_err: ниже — KEEP (тронуто не будет).")
    tau_c: float = Field(default=_3W_TAU_C, ge=0.0, le=1.0,
                         description="Вероятность топ-кандидата должна достичь этого для CORRECT.")
    tau_m: float = Field(default=_3W_TAU_M, ge=0.0, le=1.0,
                         description="Зазор топ-кандидат − второй должен достичь этого для CORRECT.")
    candidate_mode: str = Field(
        default="dl_fst_filter",
        description="Режим генерации кандидатов: dl | dl_freq | dl_fst_filter | dl_freq_morph.",
    )


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EvaluateRequest(BaseModel):
    gold: str = Field(..., description="Эталонная разметка в формате CoNLL-U")
    pred: str = Field(..., description="Системная разметка в формате CoNLL-U")
    min_support: int = Field(default=5, ge=0)


# ============================================================
#  Эндпоинты
# ============================================================

@router.post("/correct", summary="Исправить казахский текст")
def correct_text(
    req: CorrectRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    _: bool = Depends(require_api_key),
) -> dict:
    """
    Анализирует текст и возвращает исправленную версию: нераспознанные и
    низкоуверенные токены заменяются на лучший вариант исправления
    (Levenshtein), форматирование сохраняется.
    """
    return correct(pipeline, req.text, req.confidence_threshold)


@router.post("/correct3", summary="Трёхсторонний корректор (KEEP/CORRECT/REVIEW)")
def correct_text_3way(
    req: Correct3Request,
    pipeline: Pipeline = Depends(get_pipeline),
    _: bool = Depends(require_api_key),
) -> dict:
    """Анализирует текст и принимает по каждому токену одно из трёх решений:

      * KEEP    — высокая уверенность, токен оставлен как есть;
      * CORRECT — найден надёжный кандидат, токен заменён;
      * REVIEW  — токен подозрителен, но кандидаты недостаточно уверены;
                   оригинал сохраняется, кандидаты возвращаются для проверки.

    Возвращает исправленный текст (применены только CORRECT-решения, пробелы
    и пунктуация сохранены), список решений с доказательствами и сводку
    статистики. В отличие от бинарного /correct, REVIEW никогда не портит
    сомнительные, но возможно-правильные формы (нижний FCR на чистом тексте).
    """
    decisions = correct_sentence_3way(
        pipeline, req.text,
        tau_e=req.tau_e, tau_c=req.tau_c, tau_m=req.tau_m,
        candidate_mode=req.candidate_mode,
    )

    # Реконструкция строки с сохранением пробелов/пунктуации (тот же подход,
    # что и в kazmorph.corrector.correct): между токенами переносим исходный
    # текст дословно, а сам токен заменяем только при decision == CORRECT.
    # Границы — от общего токенизатора pipeline (копия регулярки здесь когда-то
    # разошлась с Pipeline._tokenize и теряла хвост текста).
    spans = token_spans(req.text)
    out_parts: list[str] = []
    last_end = 0
    n = min(len(decisions), len(spans))
    n_keep = n_correct = n_review = 0
    decision_rows: list[dict] = []

    for i in range(n):
        tok, start, end = spans[i]
        d = decisions[i]
        out_parts.append(req.text[last_end:start])
        last_end = end

        emitted = d.predicted  # == token на KEEP/REVIEW, кандидат на CORRECT
        out_parts.append(emitted)

        if d.decision == "KEEP":
            n_keep += 1
        elif d.decision == "CORRECT":
            n_correct += 1
        else:
            n_review += 1

        decision_rows.append({
            "position": start,
            "token": d.token,
            "predicted": emitted,
            "decision": d.decision,
            "pos": d.pos,
            "confidence": d.confidence,
            "p_err": d.p_err,
            "top_candidate": d.top_candidate,
            "top_prob": d.top_prob,
            "margin": d.margin,
            "n_candidates": d.n_candidates,
            "candidates": [{"word": w, "prob": p} for w, p in d.candidates[:5]],
        })

    out_parts.append(req.text[last_end:])

    return {
        "original": req.text,
        "corrected": "".join(out_parts),
        "decisions": decision_rows,
        "stats": {
            "tokens": len(decisions),
            "keep": n_keep,
            "correct": n_correct,
            "review": n_review,
        },
    }


@router.post("/analyze/batch", summary="Пакетный анализ текстов")
def analyze_batch(
    req: BatchRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    _: bool = Depends(require_api_key),
) -> dict:
    """
    Анализирует несколько текстов за один запрос (эффективнее отдельных вызовов
    при интеграции). Ошибка одного текста не валит весь запрос.
    """
    results = []
    for text in req.texts:
        try:
            tokens = pipeline.process(text)
        except Exception as exc:  # noqa: BLE001 — изолируем один текст
            tokens = [{"error": str(exc)}]
        results.append({"text": text, "tokens": tokens})
    return {"count": len(results), "results": results}


@router.post("/evaluate", summary="Оценить качество морфологической разметки")
def evaluate_annotation(
    req: EvaluateRequest,
    _: bool = Depends(require_api_key),
) -> dict:
    """
    Сравнивает системную разметку с эталонной (обе в CoNLL-U) и возвращает
    метрики P/R/F, матрицу ошибок, анализ ошибок и рекомендации
    (ТЗ, ЧАСТЬ A как сервис).
    """
    gold = parse_conllu_string(req.gold)
    pred = parse_conllu_string(req.pred)
    if not gold:
        raise HTTPException(status_code=400, detail="gold: не разобрано ни одного предложения")
    if not pred:
        raise HTTPException(status_code=400, detail="pred: не разобрано ни одного предложения")

    report = QualityReport()
    report.evaluate_from_sentences(gold, pred, min_support=req.min_support)
    return report.to_dict()


@router.get("/info", summary="Метаданные сервиса")
def service_info(
    pipeline: Pipeline = Depends(get_pipeline),
    _: bool = Depends(require_api_key),
) -> dict:
    """Возвращает версию, активные этапы пайплайна, наличие моделей и размер лексикона."""
    from kazmorph.lexicons import KNOWN_NOUNS, KNOWN_VERBS, KNOWN_ADJ, BORROWINGS

    stages = ["RMA"]
    if pipeline.crf.model is not None:
        stages.append("CRF")
    if pipeline.neural.model is not None:
        stages.append("Neural")
    stages.append("MFRN")

    return {
        "name": "KazMorphCorpus-2026",
        "version": VERSION,
        "stages": stages,
        "models": {
            "crf": pipeline.crf.model is not None,
            "neural": pipeline.neural.model is not None,
        },
        "lexicon": {
            "nouns": len(KNOWN_NOUNS),
            "verbs": len(KNOWN_VERBS),
            "adjectives": len(KNOWN_ADJ),
            "borrowings": len(BORROWINGS),
        },
    }
