"""
Генератор отчёта по качеству морфологической разметки (ТЗ, ЧАСТЬ A).

Два режима работы:

  1. Сравнение двух файлов разметки (эталон vs системная):
       python scripts/quality_report.py \
           --gold data/test.conllu --pred data/system.conllu \
           --output reports/quality.md

  2. Прогон пайплайна KazMorph на тексте эталона и оценка его вывода:
       python scripts/quality_report.py \
           --gold data/test.conllu --run-pipeline \
           --output reports/quality.html --format html

Отчёт содержит (ТЗ A.1.5–A.1.7):
  - P/R/F по каждой части речи и по корпусу в целом;
  - P/R/F-стиль метрики по каждому тексту;
  - анализ ошибок (матрица ошибок, частые спутанные пары, примеры);
  - рекомендации по улучшению качества разметки.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazmorph.evaluation import (
    ConlluSentence,
    ConlluToken,
    QualityReport,
    parse_conllu,
)


# ============================================================
#  Прогон пайплайна → системная разметка в формате CoNLL-U
# ============================================================

def run_pipeline_to_conllu(gold_path: str, out_path: str) -> str:
    """
    Прогнать пайплайн KazMorph на текстах эталонного CoNLL-U и записать
    системную разметку в CoNLL-U. Возвращает путь к записанному файлу.

    Токены выравниваются позиционно: i-й токен пайплайна ↔ i-й эталонный
    токен. Поля features/deprel в системном файле не заполняются
    (оцениваются POS и лемма).
    """
    from kazmorph.pipeline import Pipeline

    crf_path = "models/crf_model.joblib"
    neural_path = "models/neural_pos"
    lemma_path = "models/neural_lemma"
    crf_model = crf_path if os.path.exists(crf_path) else None
    neural_model = neural_path if os.path.exists(neural_path) else None
    lemma_model = lemma_path if os.path.exists(lemma_path) else None
    pipeline = Pipeline(
        crf_model_path=crf_model,
        neural_model_path=neural_model,
        lemma_model_path=lemma_model,
    )

    sentences = parse_conllu(gold_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(f"# sent_id = {sent.sent_id}\n")
            f.write(f"# text = {sent.text}\n")
            text = sent.text or " ".join(t.form for t in sent.tokens)
            results = pipeline.process(text)
            for i, gold_tok in enumerate(sent.tokens):
                if i < len(results):
                    pred = results[i]
                    upos = pred["pos"]
                    lemma = pred["lemma"]
                else:
                    # Токен, не охваченный пайплайном — помечаем как пропуск.
                    upos = "_"
                    lemma = "_"
                f.write(
                    f"{gold_tok.token_id}\t{gold_tok.form}\t{lemma}\t"
                    f"{upos}\t_\t_\t_\t_\t_\t_\n"
                )
            f.write("\n")

    return out_path


# ============================================================
#  Сводная печать в консоль
# ============================================================

def print_summary(report: QualityReport) -> None:
    m = report.metrics
    e = report.errors
    bar = "=" * 70
    print(bar)
    print("  KazMorph — Отчёт по качеству разметки (ТЗ, ЧАСТЬ A)")
    print(bar)
    print(f"  Эталон:        {report.gold_path}")
    print(f"  Система:       {report.pred_path}")
    print(f"  Предложений:   {report.gold_sentence_count}")
    print(f"  Токенов:       {report.gold_token_count} "
          f"(выровнено {e.total_tokens}, покрытие {e.coverage:.1%})")
    print()
    print(f"  POS Accuracy : {m.accuracy:.1%}")
    print(f"  Macro-F1     : {m.macro_f1:.1%}")
    print(f"  Weighted-F1  : {m.weighted_f1:.1%}")
    print(f"  Lemma Acc    : {(1 - e.lemma_error_rate):.1%}")
    print()
    n_high = sum(1 for r in report.recommendations if r.severity == "high")
    n_med = sum(1 for r in report.recommendations if r.severity == "medium")
    print(f"  Рекомендаций: {len(report.recommendations)} "
          f"(high: {n_high}, medium: {n_med})")
    print(bar)


# ============================================================
#  Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Отчёт по качеству морфологической разметки (ТЗ, ЧАСТЬ A)"
    )
    parser.add_argument(
        "--gold", required=True,
        help="Эталонный CoNLL-U файл",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pred", help="Системный CoNLL-U файл (готовая разметка)",
    )
    src.add_argument(
        "--run-pipeline", action="store_true",
        help="Прогнать пайплайн KazMorph на тексте эталона",
    )
    parser.add_argument(
        "--write-pred",
        help="(с --run-pipeline) сохранить вывод пайплайна в этот CoNLL-U",
        default=None,
    )
    parser.add_argument(
        "--output", "-o", default="reports/quality_report.md",
        help="Куда сохранить отчёт (по умолчанию reports/quality_report.md)",
    )
    parser.add_argument(
        "--format", choices=["md", "html"], default="md",
        help="Формат отчёта",
    )
    parser.add_argument(
        "--min-support", type=int, default=5,
        help="Минимальная частота тега для рекомендаций (по умолчанию 5)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"Ошибка: эталонный файл не найден: {args.gold}")
        sys.exit(1)

    # Определяем источник системной разметки
    if args.run_pipeline:
        pred_path = args.write_pred or "reports/pipeline_output.conllu"
        print(f"Прогон пайплайна над эталоном → {pred_path} ...")
        run_pipeline_to_conllu(args.gold, pred_path)
        print("Готово.\n")
    else:
        pred_path = args.pred
        if not os.path.exists(pred_path):
            print(f"Ошибка: системный файл не найден: {pred_path}")
            sys.exit(1)

    # Оценка
    report = QualityReport(args.gold, pred_path)
    report.evaluate(min_support=args.min_support)
    print_summary(report)

    # Сохранение отчёта
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report.save(args.output, fmt=args.format)
    print(f"\nОтчёт сохранён: {args.output} (формат: {args.format})")


if __name__ == "__main__":
    main()
