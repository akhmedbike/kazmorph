"""
Quality-assessment tool for morphological annotation of Kazakh text.

Реализует требования ТЗ, ЧАСТЬ A:
  - Сравнение системной разметки с эталонной (обе в формате CoNLL-U),
    либо сравнение вывода пайплайна KazMorph с эталоном.
  - Precision / Recall / F-measure по каждой части речи, по каждому
    тексту и по корпусу в целом.
  - Анализ ошибок (матрица ошибок, категоризация, примеры).
  - Автоматически формируемые рекомендации по улучшению качества разметки.

Основные измеряемые характеристики:
  - POS-тегирование (UPOS): P/R/F по тегам, macro/weighted, точность.
  - Лемматизация: точность + анализ ошибок.

Эталонная и системная разметки читаются из CoNLL-U; токены выравниваются
позиционно внутри каждого предложения (тот же принцип выравнивания, что в
scripts/benchmark.py и scripts/evaluate.py).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Iterable


# ============================================================
#  CoNLL-U parser
# ============================================================

@dataclass
class ConlluToken:
    """One token row of a CoNLL-U file."""
    token_id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str


@dataclass
class ConlluSentence:
    """One sentence (text) from a CoNLL-U file."""
    sent_id: str
    text: str
    tokens: list[ConlluToken]


def parse_conllu_string(content: str) -> list[ConlluSentence]:
    """
    Разобрать содержимое CoNLL-U (строка) в список предложений.

    Многословные токены (id содержит '-' или '.') пропускаются —
    оцениваются только атомарные токены.
    """
    sentences: list[ConlluSentence] = []
    current_tokens: list[ConlluToken] = []
    sent_id = ""
    text = ""

    for raw in content.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("#"):
            low = line.lower()
            if low.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip() if "=" in line else ""
            elif low.startswith("# text"):
                text = line.split("=", 1)[1].strip() if "=" in line else ""
            continue
        if not line:
            if current_tokens:
                sentences.append(ConlluSentence(sent_id, text, current_tokens))
            current_tokens = []
            sent_id = ""
            text = ""
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            continue
        tid = parts[0]
        if "-" in tid or "." in tid:
            continue
        current_tokens.append(ConlluToken(
            token_id=tid,
            form=parts[1],
            lemma=parts[2],
            upos=parts[3],
            xpos=parts[4],
            feats=parts[5],
        ))

    if current_tokens:
        sentences.append(ConlluSentence(sent_id, text, current_tokens))
    return sentences


def parse_conllu(path: str) -> list[ConlluSentence]:
    """Разобрать файл CoNLL-U в список предложений (делегирует в parse_conllu_string)."""
    with open(path, encoding="utf-8") as f:
        return parse_conllu_string(f.read())


# ============================================================
#  Alignment: gold vs system annotation
# ============================================================

@dataclass
class AlignedToken:
    """Позиционно выравненный токен с эталонной и системной разметкой."""
    sent_id: str
    token_id: str
    form: str
    gold_pos: str
    pred_pos: str
    gold_lemma: str
    pred_lemma: str
    gold_feats: str
    pred_feats: str

    @property
    def pos_correct(self) -> bool:
        return self.gold_pos == self.pred_pos

    @property
    def lemma_correct(self) -> bool:
        return self.gold_lemma.lower() == self.pred_lemma.lower()


def align_sentences(
    gold: list[ConlluSentence],
    pred: list[ConlluSentence],
) -> list[AlignedToken]:
    """
    Выравнять эталонные и системные предложения позиционно.

    Предложения сопоставляются по порядку; токены внутри предложения —
    также по порядку (позиционное выравнивание, как в benchmark.py).
    """
    aligned: list[AlignedToken] = []
    n = min(len(gold), len(pred))
    for si in range(n):
        g_sent = gold[si]
        p_sent = pred[si]
        sent_id = g_sent.sent_id or p_sent.sent_id or f"sent_{si + 1}"
        m = min(len(g_sent.tokens), len(p_sent.tokens))
        for ti in range(m):
            gt = g_sent.tokens[ti]
            pt = p_sent.tokens[ti]
            aligned.append(AlignedToken(
                sent_id=sent_id,
                token_id=gt.token_id,
                form=gt.form,
                gold_pos=gt.upos,
                pred_pos=pt.upos,
                gold_lemma=gt.lemma,
                pred_lemma=pt.lemma,
                gold_feats=gt.feats,
                pred_feats=pt.feats,
            ))
    return aligned


def aligned_from_dicts(
    forms: list[str],
    gold_positions: list[dict],
    pred_positions: list[dict],
    sent_id: str = "",
) -> list[AlignedToken]:
    """
    Построить AlignedToken из уже разобранных позиций (для случая, когда
    системная разметка получена напрямую из пайплайна, а не из CoNLL-U).
    """
    out: list[AlignedToken] = []
    m = min(len(gold_positions), len(pred_positions))
    for i in range(m):
        g = gold_positions[i]
        p = pred_positions[i]
        out.append(AlignedToken(
            sent_id=sent_id,
            token_id=str(i + 1),
            form=forms[i],
            gold_pos=g.get("upos", ""),
            pred_pos=p.get("upos", ""),
            gold_lemma=g.get("lemma", ""),
            pred_lemma=p.get("lemma", ""),
            gold_feats=g.get("feats", "_"),
            pred_feats=p.get("feats", "_"),
        ))
    return out


# ============================================================
#  Metrics: Precision / Recall / F-measure
# ============================================================

@dataclass
class ClassMetrics:
    """P/R/F для одного тега (класса)."""
    tag: str
    precision: float
    recall: float
    f1: float
    support: int  # число эталонных вхождений тега


@dataclass
class AggregateMetrics:
    """Сводные метрики по всем тегам."""
    accuracy: float           # доля верно предсказанных меток (== micro-F1)
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_tag: dict[str, ClassMetrics]
    support: int              # общее число токенов

    def class_table(self, min_support: int = 0) -> list[ClassMetrics]:
        """Классы, отсортированные по убыванию support, с фильтром."""
        rows = [m for m in self.per_tag.values() if m.support >= min_support]
        return sorted(rows, key=lambda m: -m.support)


def compute_prf(gold: list[str], pred: list[str]) -> AggregateMetrics:
    """
    Вычислить P/R/F по тегам + macro/weighted/accuracy из двух
    выровненных списков меток.
    """
    assert len(gold) == len(pred), "gold and pred must be the same length"
    all_tags = sorted(set(gold) | set(pred))

    per_tag: dict[str, ClassMetrics] = {}
    for tag in all_tags:
        tp = sum(1 for g, p in zip(gold, pred) if g == tag and p == tag)
        fp = sum(1 for g, p in zip(gold, pred) if g != tag and p == tag)
        fn = sum(1 for g, p in zip(gold, pred) if g == tag and p != tag)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_tag[tag] = ClassMetrics(tag, precision, recall, f1, tp + fn)

    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    total = len(gold)
    accuracy = correct / total if total else 0.0

    f1_vals = [per_tag[t].f1 for t in all_tags]
    p_vals = [per_tag[t].precision for t in all_tags]
    r_vals = [per_tag[t].recall for t in all_tags]
    macro_p = sum(p_vals) / len(p_vals) if p_vals else 0.0
    macro_r = sum(r_vals) / len(r_vals) if r_vals else 0.0
    macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0

    total_support = sum(per_tag[t].support for t in all_tags)
    weighted_f1 = (
        sum(per_tag[t].f1 * per_tag[t].support for t in all_tags) / total_support
        if total_support else 0.0
    )

    return AggregateMetrics(
        accuracy=accuracy,
        micro_precision=accuracy,
        micro_recall=accuracy,
        micro_f1=accuracy,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_tag=per_tag,
        support=total,
    )


def lemma_accuracy(tokens: Iterable[AlignedToken]) -> float:
    """Точность лемматизации по списку выравненных токенов."""
    toks = list(tokens)
    if not toks:
        return 0.0
    correct = sum(1 for t in toks if t.lemma_correct)
    return correct / len(toks)


# ============================================================
#  Confusion matrix
# ============================================================

class ConfusionMatrix:
    """Матрица ошибок gold → pred для POS-тегов."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def add(self, gold: str, pred: str) -> None:
        self._counts[(gold, pred)] = self._counts.get((gold, pred), 0) + 1

    def at(self, gold: str, pred: str) -> int:
        return self._counts.get((gold, pred), 0)

    @property
    def all_tags(self) -> list[str]:
        tags: set[str] = set()
        for g, p in self._counts:
            tags.add(g)
            tags.add(p)
        return sorted(tags)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    @property
    def diagonal(self) -> int:
        return sum(c for (g, p), c in self._counts.items() if g == p)

    @property
    def errors(self) -> int:
        return self.total - self.diagonal

    def top_confusions(self, n: int = 10) -> list[tuple[str, str, int]]:
        """Топ-N пар (gold, pred) с ошибками, без диагонали, по убыванию."""
        off = [(g, p, c) for (g, p), c in self._counts.items() if g != p]
        off.sort(key=lambda x: -x[2])
        return off[:n]

    def as_table(self, min_support: int = 1) -> str:
        """Текстовая матрица (только теги с достаточной частотой)."""
        tags = [t for t in self.all_tags
                if sum(self._counts.get((t, p), 0) for p in self.all_tags) >= min_support]
        if not tags:
            return "(пусто)"
        w = max((len(t) for t in tags), default=4) + 1
        header = " " * (w + 2) + "".join(f"{t:>{w}}" for t in tags)
        lines = [header]
        for g in tags:
            row = f"{g:>{w}} │"
            total_g = sum(self._counts.get((g, p), 0) for p in tags)
            for p in tags:
                c = self._counts.get((g, p), 0)
                cell = str(c) if c else "."
                row += f"{cell:>{w}}"
            lines.append(row)
        return "\n".join(lines)


# ============================================================
#  Error analysis
# ============================================================

@dataclass
class ErrorExample:
    sent_id: str
    form: str
    gold: str
    pred: str


@dataclass
class ConfusionPair:
    """Конкретная пара тегов (gold→pred), спутанных системой."""
    gold: str
    pred: str
    count: int
    examples: list[ErrorExample] = field(default_factory=list)


@dataclass
class LemmaError:
    sent_id: str
    form: str
    gold_lemma: str
    pred_lemma: str


@dataclass
class ErrorAnalysis:
    """Результат анализа ошибок по всему корпусу."""
    total_tokens: int
    pos_errors: int
    pos_error_rate: float
    lemma_errors: int
    lemma_error_rate: float
    coverage: float                      # выравнено / эталонных токенов
    pos_confusions: list[ConfusionPair]  # отсортировано по убыванию count
    lemma_examples: list[LemmaError]     # выборка ошибок лемматизации
    by_gold_tag: dict[str, int]          # число ошибок, где эталон = тег

    @property
    def coverage_loss(self) -> float:
        return 1.0 - self.coverage


def analyze_errors(
    tokens: list[AlignedToken],
    gold_total: int,
    confusion: ConfusionMatrix,
    max_examples: int = 5,
    max_lemma_examples: int = 30,
) -> ErrorAnalysis:
    """
    Категоризировать ошибки: спутанные пары POS-тегов (с примерами),
    ошибки лемматизации, потери покрытия.
    """
    # Примеры на каждую пару gold→pred
    pair_examples: dict[tuple[str, str], list[ErrorExample]] = {}
    by_gold_tag: dict[str, int] = {}

    for t in tokens:
        if not t.pos_correct:
            key = (t.gold_pos, t.pred_pos)
            ex = pair_examples.setdefault(key, [])
            if len(ex) < max_examples:
                ex.append(ErrorExample(t.sent_id, t.form, t.gold_pos, t.pred_pos))
            by_gold_tag[t.gold_pos] = by_gold_tag.get(t.gold_pos, 0) + 1

    confusions: list[ConfusionPair] = []
    for gold_tag, pred_tag, count in confusion.top_confusions(50):
        ex = pair_examples.get((gold_tag, pred_tag), [])
        confusions.append(ConfusionPair(gold_tag, pred_tag, count, ex))

    # Ошибки лемматизации
    lemma_errs: list[LemmaError] = []
    for t in tokens:
        if not t.lemma_correct and t.gold_pos != "PUNCT":
            lemma_errs.append(LemmaError(t.sent_id, t.form, t.gold_lemma, t.pred_lemma))
            if len(lemma_errs) >= max_lemma_examples:
                break

    pos_errors = confusion.errors
    pos_total = confusion.total
    lemma_total = sum(1 for t in tokens if t.gold_pos != "PUNCT") or 1
    lemma_err_count = sum(1 for t in tokens if not t.lemma_correct and t.gold_pos != "PUNCT")

    return ErrorAnalysis(
        total_tokens=len(tokens),
        pos_errors=pos_errors,
        pos_error_rate=pos_errors / pos_total if pos_total else 0.0,
        lemma_errors=lemma_err_count,
        lemma_error_rate=lemma_err_count / lemma_total if lemma_total else 0.0,
        coverage=len(tokens) / gold_total if gold_total else 1.0,
        pos_confusions=confusions,
        lemma_examples=lemma_errs,
        by_gold_tag=by_gold_tag,
    )


# ============================================================
#  Recommendations
# ============================================================

# Лингвистически осмысленные комментарии для типичных казахских спутанных пар.
_KAZ_CONFUSION_HINTS: dict[tuple[str, str], str] = {
    ("NOUN", "PROPN"): "имена собственные часто теряют тег PROPN; "
                       "учитывайте заглавную букву и словарь топонимов/антропонимов.",
    ("PROPN", "NOUN"): "часть нарицательных существительных помечается как PROPN; "
                       "расширьте список имён собственных.",
    ("VERB", "AUX"): "вспомогательные глаголы (жат, отыр, жүр, тұр, ет, ал, бер) "
                     "часто помечаются как полноценные VERB.",
    ("AUX", "VERB"): "некоторые самостоятельные глаголы уходят в AUX.",
    ("ADJ", "NOUN"): "прилагательные, образованные аффиксами -ды/-ді/-ты/-ті, "
                     "иногда размечаются как существительные.",
    ("NOUN", "ADJ"): "отадъективные существительные путаются с прилагательными.",
    ("ADV", "ADJ"): "наречия и прилагательные на -ша/-ше смешиваются.",
    ("NUM", "ADJ"): "порядковые числительные (-ншы/-нші) путаются с прилагательными.",
    ("CCONJ", "SCONJ"): "союзы және/да/де vs өйткені/себебі путаются.",
    ("X", "NOUN"): "нераспознанные словоформы (X) часто оказываются существительными.",
}


@dataclass
class Recommendation:
    severity: str   # "high" | "medium" | "low"
    category: str   # "pos" | "lemma" | "coverage" | "overall"
    title: str
    detail: str
    evidence: str


def _severity_from_score(score: float) -> str:
    if score < 0.50:
        return "high"
    if score < 0.75:
        return "medium"
    return "low"


def _plural_cases(n: int) -> str:
    """Правильное согласование слова «случай» с числом (1 случай / 2 случая / 5 случаев)."""
    n10 = n % 10
    n100 = n % 100
    if n10 == 1 and n100 != 11:
        word = "случай"
    elif 2 <= n10 <= 4 and not (12 <= n100 <= 14):
        word = "случая"
    else:
        word = "случаев"
    return f"{n} {word}"


def generate_recommendations(
    metrics: AggregateMetrics,
    confusion: ConfusionMatrix,
    errors: ErrorAnalysis,
    min_support: int = 5,
) -> list[Recommendation]:
    """
    Сформировать рекомендации по улучшению качества разметки на основе
    выявленных паттернов ошибок (ТЗ A.1.7).
    """
    recs: list[Recommendation] = []

    # 1. Общая оценка макро-F1
    if metrics.macro_f1 < 0.85:
        recs.append(Recommendation(
            severity=_severity_from_score(metrics.macro_f1),
            category="overall",
            title="Повысить макро-F1 POS-тегирования",
            detail=f"Macro-F1 = {metrics.macro_f1:.1%}, Accuracy = {metrics.accuracy:.1%}. "
                   f"Целевой уровень — ≥85%. Усильте теги с низким recall.",
            evidence=f"macro_f1={metrics.macro_f1:.3f}, weighted_f1={metrics.weighted_f1:.3f}",
        ))

    # 2. Теги с низким recall (недопризнание)
    for cm in metrics.class_table(min_support):
        if cm.recall >= 0.70:
            continue
        top_target = next(
            ((g, p, c) for g, p, c in confusion.top_confusions(50)
             if g == cm.tag),
            None,
        )
        hint = ""
        if top_target:
            _, pred_tag, cnt = top_target
            hint = _KAZ_CONFUSION_HINTS.get(
                (cm.tag, pred_tag),
                f"чаще всего ошибочно помечается как {pred_tag} ({_plural_cases(cnt)}).",
            )
        recs.append(Recommendation(
            severity=_severity_from_score(cm.recall),
            category="pos",
            title=f"Низкий recall тега {cm.tag} (R={cm.recall:.1%})",
            detail=f"Из {cm.support} эталонных вхождений распознано верно "
                   f"{round(cm.recall * cm.support)}. {hint}",
            evidence=f"recall={cm.recall:.3f}, f1={cm.f1:.3f}, support={cm.support}",
        ))

    # 3. Теги с низким precision (ложные срабатывания)
    for cm in metrics.class_table(min_support):
        if cm.precision >= 0.70:
            continue
        top_source = next(
            ((g, p, c) for g, p, c in confusion.top_confusions(50)
             if p == cm.tag),
            None,
        )
        src = ""
        if top_source:
            gold_tag, _, cnt = top_source
            src = f"Источником ложных срабатываний чаще всего служит {gold_tag} ({_plural_cases(cnt)}). "
        recs.append(Recommendation(
            severity=_severity_from_score(cm.precision),
            category="pos",
            title=f"Низкий precision тега {cm.tag} (P={cm.precision:.1%})",
            detail=f"{src}Уточните признаки, отличающие {cm.tag} от соседних классов.",
            evidence=f"precision={cm.precision:.3f}, support={cm.support}",
        ))

    # 4. Лемматизация
    if errors.lemma_error_rate > 0.20:
        recs.append(Recommendation(
            severity=_severity_from_score(1.0 - errors.lemma_error_rate),
            category="lemma",
            title="Улучшить лемматизацию",
            detail=f"Доля ошибок лемматизации = {errors.lemma_error_rate:.1%}. "
                   f"Проверьте отрезание продуктивных аффиксов "
                   f"(-дық/-дік, -лық/-лік, -сыз/-сіз, -лы/-лі).",
            evidence=f"lemma_error_rate={errors.lemma_error_rate:.3f}",
        ))

    # 5. Потеря покрытия (невыравненные токены)
    if errors.coverage < 0.99:
        recs.append(Recommendation(
            severity="high" if errors.coverage < 0.95 else "medium",
            category="coverage",
            title="Потеря покрытия токенов",
            detail=f"Выравнено {errors.coverage:.1%} эталонных токенов. "
                   f"Разница в токенизации между эталоном и системой приводит к "
                   f"сдвигу позиций и ложным ошибкам. Унифицируйте токенизацию.",
            evidence=f"coverage={errors.coverage:.3f}",
        ))

    # Сортировка: high → medium → low, затем по категории
    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: (order.get(r.severity, 9), r.category, r.title))
    return recs


# ============================================================
#  Per-text metrics
# ============================================================

@dataclass
class TextMetrics:
    """Метрики для отдельного текста (предложения)."""
    sent_id: str
    text: str
    tokens: int
    pos_correct: int
    pos_accuracy: float
    pos_f1: float          # macro-F1 по тегам, присутствующим в тексте
    lemma_accuracy: float


def compute_per_text(tokens: list[AlignedToken]) -> list[TextMetrics]:
    """Посчитать P/R/F-стиль метрики для каждого текста (предложения)."""
    by_sent: dict[str, list[AlignedToken]] = {}
    text_by_sent: dict[str, str] = {}
    for t in tokens:
        by_sent.setdefault(t.sent_id, []).append(t)
        text_by_sent.setdefault(t.sent_id, t.form)

    results: list[TextMetrics] = []
    for sid, toks in by_sent.items():
        gold = [t.gold_pos for t in toks]
        pred = [t.pred_pos for t in toks]
        agg = compute_prf(gold, pred)
        pos_correct = sum(1 for t in toks if t.pos_correct)
        results.append(TextMetrics(
            sent_id=sid,
            text=" ".join(t.form for t in toks),
            tokens=len(toks),
            pos_correct=pos_correct,
            pos_accuracy=agg.accuracy,
            pos_f1=agg.macro_f1,
            lemma_accuracy=lemma_accuracy(toks),
        ))
    return results


# ============================================================
#  QualityReport — агрегат + рендеринг отчёта
# ============================================================

class QualityReport:
    """
    Полный отчёт по качеству разметки: сравнение эталона и системной
    разметки с метриками, анализом ошибок и рекомендациями.

    Пример::

        report = QualityReport("data/test.conllu", "data/system.conllu")
        report.evaluate()
        print(report.to_markdown())
        report.save("reports/quality.md")
    """

    def __init__(self, gold_path: str = "", pred_path: str = "") -> None:
        self.gold_path = gold_path
        self.pred_path = pred_path
        self.metrics: AggregateMetrics | None = None
        self.confusion: ConfusionMatrix | None = None
        self.errors: ErrorAnalysis | None = None
        self.recommendations: list[Recommendation] = []
        self.per_text: list[TextMetrics] = []
        self.gold_sentence_count = 0
        self.gold_token_count = 0

    def evaluate(self, min_support: int = 5) -> None:
        """Разобрать оба файла, выровнять, вычислить все метрики."""
        gold = parse_conllu(self.gold_path)
        pred = parse_conllu(self.pred_path)
        self.evaluate_from_sentences(gold, pred, min_support=min_support)

    def evaluate_from_sentences(
        self,
        gold: list[ConlluSentence],
        pred: list[ConlluSentence],
        min_support: int = 5,
    ) -> None:
        """
        Вычислить все метрики по уже разобранным предложениям
        (без чтения файлов — для программного/API использования).
        """
        self.gold_sentence_count = len(gold)
        self.gold_token_count = sum(len(s.tokens) for s in gold)

        tokens = align_sentences(gold, pred)

        gold_pos = [t.gold_pos for t in tokens]
        pred_pos = [t.pred_pos for t in tokens]
        self.metrics = compute_prf(gold_pos, pred_pos)

        confusion = ConfusionMatrix()
        for t in tokens:
            confusion.add(t.gold_pos, t.pred_pos)
        self.confusion = confusion

        self.errors = analyze_errors(
            tokens, self.gold_token_count, confusion
        )
        self.recommendations = generate_recommendations(
            self.metrics, confusion, self.errors, min_support=min_support
        )
        self.per_text = compute_per_text(tokens)

    def to_dict(self, max_texts: int = 100) -> dict:
        """
        JSON-сериализуемое представление отчёта (для REST API).

        Возвращает сводные метрики, P/R/F по тегам, топ спутанных пар,
        рекомендации и метрики по текстам.
        """
        if self.metrics is None or self.errors is None:
            raise RuntimeError("Сначала вызовите evaluate()/evaluate_from_sentences()")
        m = self.metrics
        e = self.errors
        return {
            "summary": {
                "gold_sentences": self.gold_sentence_count,
                "gold_tokens": self.gold_token_count,
                "aligned_tokens": e.total_tokens,
                "coverage": round(e.coverage, 4),
                "pos_accuracy": round(m.accuracy, 4),
                "macro_precision": round(m.macro_precision, 4),
                "macro_recall": round(m.macro_recall, 4),
                "macro_f1": round(m.macro_f1, 4),
                "weighted_f1": round(m.weighted_f1, 4),
                "lemma_accuracy": round(1.0 - e.lemma_error_rate, 4),
                "pos_errors": e.pos_errors,
                "lemma_errors": e.lemma_errors,
            },
            "per_tag": [
                {
                    "tag": cm.tag,
                    "precision": round(cm.precision, 4),
                    "recall": round(cm.recall, 4),
                    "f1": round(cm.f1, 4),
                    "support": cm.support,
                }
                for cm in m.class_table()
            ],
            "top_confusions": [
                {"gold": g, "pred": p, "count": c}
                for g, p, c in (self.confusion.top_confusions(15) if self.confusion else [])
            ],
            "recommendations": [
                {
                    "severity": r.severity,
                    "category": r.category,
                    "title": r.title,
                    "detail": r.detail,
                    "evidence": r.evidence,
                }
                for r in self.recommendations
            ],
            "per_text": [
                {
                    "sent_id": tm.sent_id,
                    "tokens": tm.tokens,
                    "pos_accuracy": round(tm.pos_accuracy, 4),
                    "pos_f1": round(tm.pos_f1, 4),
                    "lemma_accuracy": round(tm.lemma_accuracy, 4),
                }
                for tm in self.per_text[:max_texts]
            ],
        }

    # ── Доступ ────────────────────────────────────────────────
    @property
    def lemma_accuracy(self) -> float:
        return self.errors.lemma_error_rate if self.errors else 0.0

    # ── Рендеринг в Markdown ──────────────────────────────────
    def to_markdown(self) -> str:
        if self.metrics is None:
            raise RuntimeError("Сначала вызовите evaluate()")
        m = self.metrics
        e = self.errors
        lines: list[str] = []
        lines.append("# Отчёт по качеству морфологической разметки")
        lines.append("")
        lines.append(f"- Эталон: `{self.gold_path}`")
        lines.append(f"- Системная разметка: `{self.pred_path}`")
        lines.append(f"- Предложений (эталон): {self.gold_sentence_count}")
        lines.append(f"- Токенов (эталон): {self.gold_token_count}")
        lines.append(f"- Выровнено токенов: {e.total_tokens} (покрытие {e.coverage:.1%})")
        lines.append("")
        lines.append("## 1. Сводные метрики по корпусу")
        lines.append("")
        lines.append("| Метрика | Значение |")
        lines.append("|---|---|")
        lines.append(f"| POS Accuracy | **{m.accuracy:.1%}** |")
        lines.append(f"| Macro-F1 | {m.macro_f1:.1%} |")
        lines.append(f"| Weighted-F1 | {m.weighted_f1:.1%} |")
        lines.append(f"| Macro-Precision | {m.macro_precision:.1%} |")
        lines.append(f"| Macro-Recall | {m.macro_recall:.1%} |")
        lines.append(f"| Lemma Accuracy | {(1 - e.lemma_error_rate):.1%} |")
        lines.append("")
        lines.append("## 2. P/R/F по частям речи")
        lines.append("")
        lines.append("| UPOS | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for cm in m.class_table():
            lines.append(
                f"| {cm.tag} | {cm.precision:.3f} | {cm.recall:.3f} | "
                f"{cm.f1:.3f} | {cm.support} |"
            )
        lines.append("")
        lines.append("## 3. Анализ ошибок")
        lines.append("")
        lines.append(f"- Всего ошибок POS: **{e.pos_errors}** ({e.pos_error_rate:.1%})")
        lines.append(f"- Всего ошибок лемм: **{e.lemma_errors}** ({e.lemma_error_rate:.1%})")
        lines.append("")
        lines.append("### 3.1 Матрица ошибок (gold → pred)")
        lines.append("")
        lines.append("```")
        lines.append(self.confusion.as_table(min_support=3))
        lines.append("```")
        lines.append("")
        lines.append("### 3.2 Частые спутанные пары тегов")
        lines.append("")
        lines.append("| Эталон | Система | Кол-во | Примеры |")
        lines.append("|---|---|---:|---|")
        for cp in e.pos_confusions[:10]:
            exs = ", ".join(f"{ex.form} ({ex.sent_id})" for ex in cp.examples[:3]) or "—"
            lines.append(f"| {cp.gold} | {cp.pred} | {cp.count} | {exs} |")
        lines.append("")
        lines.append("### 3.3 Примеры ошибок лемматизации")
        lines.append("")
        if e.lemma_examples:
            lines.append("| Форма | Эталон | Система | Предложение |")
            lines.append("|---|---|---|---|")
            for le in e.lemma_examples[:15]:
                lines.append(f"| {le.form} | {le.gold_lemma} | {le.pred_lemma} | {le.sent_id} |")
        else:
            lines.append("_Ошибок лемматизации не обнаружено._")
        lines.append("")
        lines.append("## 4. Рекомендации по улучшению")
        lines.append("")
        if self.recommendations:
            for i, r in enumerate(self.recommendations, 1):
                lines.append(f"{i}. **[{r.severity.upper()}] {r.title}**")
                lines.append(f"   - {r.detail}")
                lines.append(f"   - _Обоснование:_ {r.evidence}")
                lines.append("")
        else:
            lines.append("_Существенных проблем не выявлено._")
        lines.append("")
        lines.append("## 5. Метрики по каждому тексту")
        lines.append("")
        lines.append("| # | Текст | Токенов | POS Acc | POS F1 | Lemma Acc |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, tm in enumerate(self.per_text[:50], 1):
            preview = tm.text[:60].replace("|", "\\|")
            if len(tm.text) > 60:
                preview += "…"
            lines.append(
                f"| {i} | {preview} | {tm.tokens} | "
                f"{tm.pos_accuracy:.1%} | {tm.pos_f1:.1%} | {tm.lemma_accuracy:.1%} |"
            )
        if len(self.per_text) > 50:
            lines.append(f"\n_…и ещё {len(self.per_text) - 50} текстов_")
        lines.append("")
        return "\n".join(lines)

    # ── Рендеринг в HTML ──────────────────────────────────────
    def to_html(self) -> str:
        md = self.to_markdown()
        body = _markdown_to_html(md)
        return (
            "<!DOCTYPE html>\n<html lang=\"ru\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Отчёт по качеству разметки — KazMorph</title>\n"
            "<style>\n"
            "body{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:980px;"
            "margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}\n"
            "th,td{border:1px solid #ccc;padding:.35rem .55rem;text-align:left}\n"
            "th{background:#f4f4f4}\n"
            "pre{background:#f6f6f6;padding:.8rem;overflow-x:auto;border-radius:6px}\n"
            "code{background:#f0f0f0;padding:.1rem .3rem;border-radius:3px}\n"
            "h1,h2,h3{color:#222}\n"
            "</style>\n"
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
        )

    def save(self, path: str, fmt: str = "md") -> None:
        """Сохранить отчёт в файл. fmt: 'md' или 'html'."""
        content = self.to_html() if fmt == "html" else self.to_markdown()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def _markdown_to_html(md: str) -> str:
    """
    Минимальный Markdown→HTML: заголовки, таблицы, списки, абзацы,
    inline-код. Достаточно для отчёта без внешних зависимостей.
    """
    import re

    def esc(s: str) -> str:
        return html.escape(s, quote=False)

    out: list[str] = []
    lines = md.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Заголовки
        if line.startswith("# "):
            out.append(f"<h1>{_inline(esc(line[2:]))}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(esc(line[3:]))}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{_inline(esc(line[4:]))}</h3>")
        elif line.startswith("```"):
            # блок кода
            i += 1
            code: list[str] = []
            while i < n and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(f"<pre>{esc(chr(10).join(code))}</pre>")
        elif line.startswith("|") and i + 1 < n and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            # таблица
            header = [c.strip() for c in _split_row(line)]
            i += 2
            rows: list[list[str]] = []
            while i < n and lines[i].startswith("|"):
                rows.append([c.strip() for c in _split_row(lines[i])])
                i += 1
            out.append("<table>")
            out.append("<tr>" + "".join(f"<th>{_inline(esc(h))}</th>" for h in header) + "</tr>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{_inline(esc(c))}</td>" for c in row) + "</tr>")
            out.append("</table>")
            i -= 1  # компенсация +=1 в заголовке цикла
        elif re.match(r"^\d+\. ", line):
            out.append(f"<p>{_inline(esc(line))}</p>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{_inline(esc(line))}</p>")
        i += 1
    return "\n".join(out)


def _split_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return inner.split("|")


def _inline(text: str) -> str:
    """Обработка **bold**, _italic_, `code` в уже эскейпнутом тексте."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
    return text
