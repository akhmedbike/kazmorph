#!/usr/bin/env python3
"""
Retrain CRF-модели на одобренных правках из Active Learning контура.

Пайплайн:
  1. Загружает одобренные аннотации (status IN approved/edited) из БД.
  2. Группирует токены в «предложения» по полю sentence аннотации.
  3. Decontamination: убирает предложения, дублирующие data/test.conllu
     (по нормализованному тексту — защита от утечки silver→test, которая
     уже однажды случалась, см. reports/EVALUATION_REPORT.md).
  4. Мерджит с data/train.conllu → data/versions/train_v{N}.conllu.
  5. Замер ДО: evaluate_pipeline на data/test.conllu с текущей моделью.
  6. Обучает: subprocess train_crf.py на merged-файле → временный joblib.
  7. Замер ПОСЛЕ: те же метрики с новой моделью.
  8. McNemar: если POS accuracy значимо упала → mark_rejected, pointer
     остаётся на старой. Если выросла/стабильна → promote (новая current).

ГЕЙТ retrain (защита от model collapse, риск №2 из idea-документа):
  promote только если new_pos_acc >= baseline_pos_acc (с допустимым
  шумом через McNemar: если разница не значима, promote всё равно —
  новые данные не вредят, а покрытие расширяется).

Запуск:
  DATABASE_URL=... python scripts/retrain.py
  DATABASE_URL=... python scripts/retrain.py --dry-run  # без обучения, только отчёт
  DATABASE_URL=... python scripts/retrain.py --min-corrections 50  # порог
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kazmorph.model_versioning import (  # noqa: E402
    mark_rejected,
    next_version_number,
    promote,
    read_pointer,
    resolve_current_model_path,
    save_version,
    version_meta_path,
)
from scripts.evaluate import (  # noqa: E402
    bootstrap_ci,
    evaluate_pipeline,
    mcnemar_test,
    parse_conllu,
)

logger = logging.getLogger("retrain")

TRAIN_FILE = "data/train.conllu"
TEST_FILE = "data/test.conllu"
DEV_FILE = "data/dev.conllu"
REPORT_DIR = Path("reports/retrain")
MERGED_DIR = Path("data/versions")

# Порог числа правок для запуска retrain (идея-документ: 500 для CRF).
DEFAULT_MIN_CORRECTIONS = 50  # снижен для раннего тестирования; поднять до 500 в прод


# ============================================================
#  Загрузка одобренных аннотаций → CoNLL-U
# ============================================================


def load_approved_annotations() -> list[dict]:
    """Загрузить approved/edited аннотации из БД.

    Возвращает список словарей в формате, удобном для группировки по
    предложениям: {sentence, token, final_lemma, final_pos, source_url}.
    """
    from app.db import init_db, get_session_factory
    from app.models import Annotation

    if not init_db():
        logger.warning("БД недоступна — нет правок для retrain")
        return []
    factory = get_session_factory()
    if factory is None:
        return []
    db = factory()
    try:
        rows = (
            db.query(Annotation)
            .filter(Annotation.status.in_(["approved", "edited"]))
            .filter(Annotation.final_lemma.isnot(None))
            .filter(Annotation.final_pos.isnot(None))
            .all()
        )
        return [
            {
                "sentence": r.sentence,
                "token": r.token,
                "final_lemma": r.final_lemma,
                "final_pos": r.final_pos,
                "source_url": r.source_url,
                "annotation_id": r.id,
            }
            for r in rows
        ]
    finally:
        db.close()


def _normalize_sentence(text: str) -> str:
    """Нормализация для decontamination: lower + схлопывание пробелов."""
    return " ".join(text.split()).strip().lower()


def annotations_to_conllu(annotations: list[dict]) -> list[list[dict]]:
    """Сгруппировать аннотации в предложения (CoNLL-U-формат).

    Группировка по полю sentence — токены одной строки-предложения
    собираются в одно предложение. Внутри предложения токены
    переупорядочиваются по позиции в исходной строке.
    """
    # Группировка по предложению.
    by_sentence: dict[str, list[dict]] = {}
    for ann in annotations:
        key = _normalize_sentence(ann["sentence"])
        by_sentence.setdefault(key, []).append(ann)

    sentences = []
    for sent_text, anns in by_sentence.items():
        # Сортировка токенов по позиции в исходном предложении.
        original = anns[0]["sentence"]
        anns_sorted = sorted(anns, key=lambda a: original.find(a["token"]))
        tokens = []
        for idx, ann in enumerate(anns_sorted, 1):
            tokens.append({
                "id": str(idx),
                "form": ann["token"],
                "lemma": ann["final_lemma"],
                "upos": ann["final_pos"],
            })
        if tokens:
            sentences.append(tokens)
    return sentences


# ============================================================
#  Decontamination vs test.conllu
# ============================================================


def load_test_sentence_fingerprints(test_file: str = TEST_FILE) -> set[str]:
    """Множество нормализованных текстов предложений из test.conllu.

    Используется для decontamination — новые данные не должны дублировать
    тестовые предложения (label leak).
    """
    sentences = parse_conllu(test_file)
    fps = set()
    for sent in sentences:
        text = " ".join(t["form"] for t in sent)
        fps.add(_normalize_sentence(text))
    return fps


def decontaminate(sentences: list[list[dict]], test_file: str = TEST_FILE) -> list[list[dict]]:
    """Убрать предложения, дублирующие test.conllu по тексту."""
    test_fps = load_test_sentence_fingerprints(test_file)
    kept = []
    removed = 0
    for sent in sentences:
        text = _normalize_sentence(" ".join(t["form"] for t in sent))
        if text in test_fps:
            removed += 1
            continue
        kept.append(sent)
    if removed:
        logger.info("Decontamination: удалено %d дубликатов с test.conllu", removed)
    return kept


# ============================================================
#  Мердж с train.conllu
# ============================================================


def write_conllu(sentences: list[list[dict]], path: Path) -> None:
    """Записать предложения в CoNLL-U файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(sentences, 1):
            f.write(f"# retrain_sentence = {i}\n")
            for tok in sent:
                # 10-колоночный CoNLL-U: id, form, lemma, upos, _, _, _, _, _, _
                cols = [
                    tok["id"], tok["form"], tok["lemma"], tok["upos"],
                    "_", "_", "_", "_", "_", "_",
                ]
                f.write("\t".join(cols) + "\n")
            f.write("\n")


def merge_train(new_sentences: list[list[dict]],
                train_file: str = TRAIN_FILE) -> tuple[list[list[dict]], int]:
    """Merje новых предложений с train.conllu. Возвращает (merged, n_train_orig)."""
    original = parse_conllu(train_file)
    n_orig = len(original)
    merged = original + new_sentences
    return merged, n_orig


# ============================================================
#  Замер качества на gold-set
# ============================================================


def evaluate_model(model_path: str, test_file: str = TEST_FILE) -> dict:
    """Оценить CRF-модель на test.conllu. Возвращает {pos_acc, ci, per_token_correct}."""
    from kazmorph.crf_tagger import CRFTagger
    from kazmorph.pipeline import Pipeline

    crf = CRFTagger(model_path=model_path)
    pipeline = Pipeline(crf_model_path=model_path, neural_model_path=None,
                        lemma_model_path=None)
    sentences = parse_conllu(test_file)
    tokens = evaluate_pipeline(pipeline, sentences)
    pos_correct = [t["pred_pos"] == t["gold_pos"] for t in tokens]
    pos_acc, ci_lo, ci_hi = bootstrap_ci(pos_correct, n_iterations=2000)
    return {
        "pos_accuracy": round(pos_acc / 100, 4),
        "pos_ci_lower": round(ci_lo / 100, 4),
        "pos_ci_upper": round(ci_hi / 100, 4),
        "per_token_correct": pos_correct,
        "n_tokens": len(tokens),
    }


# ============================================================
#  Обучение через subprocess
# ============================================================


def train_new_model(train_file: str, dev_file: str, out_path: str,
                    seed: int = 42) -> subprocess.CompletedProcess:
    """Запустить train_crf.py через subprocess на merged train-файле."""
    cmd = [
        sys.executable, "scripts/train_crf.py",
        "--train-file", train_file,
        "--dev-file", dev_file,
        "--seed", str(seed),
    ]
    logger.info("Обучение: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("train_crf.py провален (exit %d):\n%s",
                     result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"train_crf.py failed: exit {result.returncode}")
    # train_crf.py пишет в models/crf_model.joblib — копируем в out_path.
    if os.path.exists("models/crf_model.joblib"):
        import shutil
        shutil.copy2("models/crf_model.joblib", out_path)
    return result


# ============================================================
#  Главный цикл retrain
# ============================================================


def run_retrain(
    *,
    min_corrections: int = DEFAULT_MIN_CORRECTIONS,
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    """Полный цикл: правки → decontaminate → merge → eval → train → gate."""
    # 1. Загрузка правок.
    annotations = load_approved_annotations()
    logger.info("Одобренных правок: %d", len(annotations))
    if len(annotations) < min_corrections:
        logger.info("Недостаточно правок (%d < %d порога) — retrain отменён",
                    len(annotations), min_corrections)
        return {"skipped": True, "reason": "below_min_corrections",
                "n_corrections": len(annotations), "min": min_corrections}

    # 2. Группировка + decontamination + merge.
    new_sentences = annotations_to_conllu(annotations)
    new_sentences = decontaminate(new_sentences)
    if not new_sentences:
        logger.warning("После decontamination не осталось новых предложений")
        return {"skipped": True, "reason": "all_decontaminated",
                "n_corrections": len(annotations)}
    logger.info("Новых предложений после decontam: %d", len(new_sentences))

    merged, n_train_orig = merge_train(new_sentences)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = MERGED_DIR / f"train_merged_{datetime.now(timezone.utc).strftime('%Y%m%d')}.conllu"
    write_conllu(merged, merged_path)
    logger.info("Merged train: %d предложений (%d оригинал + %d новые) → %s",
                len(merged), n_train_orig, len(new_sentences), merged_path)

    # 3. Baseline: оценка текущей модели.
    current_path = resolve_current_model_path()
    if current_path is None:
        return {"skipped": True, "reason": "no_current_model"}
    logger.info("Baseline eval на текущей модели: %s", current_path)
    baseline = evaluate_model(current_path)
    logger.info("Baseline POS acc: %.4f [%.4f, %.4f]",
                baseline["pos_accuracy"], baseline["pos_ci_lower"], baseline["pos_ci_upper"])

    if dry_run:
        logger.info("Dry-run: обучение пропущено")
        return {"dry_run": True, "baseline": baseline,
                "n_new_sentences": len(new_sentences), "merged_path": str(merged_path)}

    # 4. Обучение новой модели во временный файл.
    next_n = next_version_number()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model = os.path.join(tmpdir, "crf_new.joblib")
        train_new_model(str(merged_path), DEV_FILE, tmp_model, seed=seed)

        # 5. Оценка новой модели.
        logger.info("New model eval на test.conllu")
        new_metrics = evaluate_model(tmp_model)
        logger.info("New POS acc: %.4f [%.4f, %.4f]",
                    new_metrics["pos_accuracy"], new_metrics["pos_ci_lower"],
                    new_metrics["pos_ci_upper"])

        # 6. McNemar + решение гейта.
        mcnemar = mcnemar_test(baseline["per_token_correct"], new_metrics["per_token_correct"])
        # ГЕЙТ: promote если новая не хуже (acc не упала значимо).
        # Если значимо хуже → reject. Если не значимо или лучше → promote.
        baseline_acc = baseline["pos_accuracy"]
        new_acc = new_metrics["pos_accuracy"]
        # Значимое ухудшение: new < baseline И mcnemar значим в худшую сторону.
        # mcnemar.significant означает разницу вообще; направление по n01 vs n10.
        # n01 = baseline correct, new wrong (ухудшение). n10 = наоборот.
        significantly_worse = (new_acc < baseline_acc
                               and mcnemar["significant"]
                               and mcnemar["n01"] > mcnemar["n10"])

        # Сохраняем версию в любом случае (для истории).
        meta = {
            "baseline_pos_acc": baseline_acc,
            "new_pos_acc": new_acc,
            "delta_pos_acc": round(new_acc - baseline_acc, 4),
            "mcnemar": mcnemar,
            "n_corrections": len(annotations),
            "n_new_sentences": len(new_sentences),
            "n_train_orig": n_train_orig,
            "source": "active_learning_rss_llm",
            "merged_train": str(merged_path),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        n, model_path = save_version(tmp_model, meta=meta)
        logger.info("Сохранена версия v%d → %s", n, model_path)

        if significantly_worse:
            # Отклонить: модель хуже, pointer не трогаем.
            reason = (f"POS acc упала {baseline_acc:.4f}→{new_acc:.4f}, "
                      f"McNemar p={mcnemar['p_value']} (значимо)")
            rejected_path = mark_rejected(n, reason=reason)
            logger.warning("REJECTED v%d: %s → %s", n, reason, rejected_path.name)
            decision = "rejected"
        else:
            # Promote: новая модель не хуже → делаем текущей.
            promote(n, baseline_pos_acc=new_acc,
                    extra={"source": "active_learning_rss_llm",
                           "n_corrections": len(annotations)})
            logger.info("PROMOTED v%d: new current (acc %.4f→%.4f)", n, baseline_acc, new_acc)
            decision = "promoted"

        report = {
            "version": f"v{n}",
            "decision": decision,
            "baseline": {k: v for k, v in baseline.items() if k != "per_token_correct"},
            "new": {k: v for k, v in new_metrics.items() if k != "per_token_correct"},
            "delta_pos_acc": round(new_acc - baseline_acc, 4),
            "mcnemar": mcnemar,
            "n_corrections": len(annotations),
            "n_new_sentences": len(new_sentences),
            "merged_path": str(merged_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # 7. Отчёт.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Отчёт: %s", report_path)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-corrections", type=int, default=DEFAULT_MIN_CORRECTIONS,
                    help=f"Минимум правок для запуска retrain (default {DEFAULT_MIN_CORRECTIONS}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Без обучения — только отчёт о правках и baseline.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_retrain(
        min_corrections=args.min_corrections,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    print("\n" + "=" * 55)
    print("Retrain CRF")
    print("=" * 55)
    if report.get("skipped"):
        print(f"  Пропущено: {report['reason']} ({report.get('n_corrections', 0)} правок)")
    elif report.get("dry_run"):
        print(f"  Dry-run. Baseline POS acc: {report['baseline']['pos_accuracy']:.4f}")
        print(f"  Новых предложений: {report['n_new_sentences']}")
        print(f"  Merged train: {report['merged_path']}")
    else:
        print(f"  Версия:       {report['version']}")
        print(f"  Решение:      {report['decision']}")
        print(f"  Baseline POS: {report['baseline']['pos_accuracy']:.4f}")
        print(f"  New POS:      {report['new']['pos_accuracy']:.4f}")
        print(f"  Delta:        {report['delta_pos_acc']:+.4f}")
        print(f"  McNemar:      p={report['mcnemar']['p_value']} "
              f"(significant={report['mcnemar']['significant']})")
        print(f"  Правок:       {report['n_corrections']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
