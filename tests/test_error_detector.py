"""
Тесты поверхностного детектора ошибок (error_detector) и детерминизма порчи.

Запуск: pytest tests/test_error_detector.py -v
"""

import json
import os

import pytest

from kazmorph.correction.error_detector import (
    extract_features, estimate_p_err, FEATURES, DEFAULT_MODEL_PATH,
)


class TestFeatures:
    def test_all_features_present_and_bounded(self):
        feats = extract_features("кітапты", 0.9)
        assert set(feats) == set(FEATURES)
        for v in feats.values():
            assert -0.01 <= v <= 10.0

    def test_valid_form_differs_from_garbage(self):
        good = extract_features("кітап", 0.9)
        bad = extract_features("кытаппп", 0.9)
        assert good["in_form_index"] >= bad["in_form_index"]
        # κ для мусора вакуумно равен 1.0 (RMA не находит разбор — штрафовать
        # нечему), поэтому монотонности по κ нет — эту роль несёт rma_unk:
        assert good["rma_unk"] <= bad["rma_unk"]

    def test_pos_conf_clamped(self):
        feats = extract_features("сөз", 5.0)   # выход за [0,1] обрезается
        assert feats["pos_conf"] == 1.0


class TestEstimate:
    def test_fallback_without_model(self, tmp_path):
        # Без файла модели — легаси-поведение: p_err = 1 - pos_conf
        p = estimate_p_err("сөз", 0.8, model_path=str(tmp_path / "missing.json"))
        assert p == pytest.approx(0.2)

    def test_logistic_with_model(self, tmp_path):
        weights = {f: 0.0 for f in FEATURES}
        weights["in_form_index"] = -10.0     # членство в форм-индексе сильно снижает p_err
        payload = {"model": "test", "features": list(FEATURES),
                   "weights": weights, "bias": 5.0, "metrics": {}}
        path = tmp_path / "det.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        # форма из корпуса → низкий p_err; мусор → высокий
        p_good = estimate_p_err("адам", 1.0, model_path=str(path))
        p_bad = estimate_p_err("xyzqwerty", 1.0, model_path=str(path))
        assert p_good < 0.5 < p_bad

    def test_model_feature_mismatch_rejected(self, tmp_path):
        payload = {"model": "test", "features": ["other"], "weights": {}, "bias": 0.0}
        path = tmp_path / "det.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            estimate_p_err("сөз", 0.5, model_path=str(path))

    @pytest.mark.skipif(not os.path.exists(DEFAULT_MODEL_PATH),
                        reason="нет обученной модели models/error_detector.json")
    def test_corrupted_scores_above_clean(self):
        # Живые кейсы: испорченная форма должна получать больший p_err
        for clean, corrupted in [("бүгін", "бүгн"), ("кітапты", "китапты"),
                                 ("халық", "халкы")]:
            assert estimate_p_err(corrupted, 0.9) > estimate_p_err(clean, 0.9), \
                f"{corrupted} должен подозреваться сильнее, чем {clean}"


class TestCorruptionDeterminism:
    def test_corrupt_fold_reproducible_same_seed(self):
        """Регрессия: пулы на frozenset и итерация set-литерала делали вывод
        зависимым от PYTHONHASHSEED; теперь вывод байтово воспроизводим."""
        from kazmorph.corpus import parse_conllu
        from kazmorph.correction.corrupt import corrupt_fold
        sents = parse_conllu("data/dev.conllu")
        a = corrupt_fold(sents, 11)
        b = corrupt_fold(sents, 11)
        assert [x.corrupted_forms for x in a] == [x.corrupted_forms for x in b]
