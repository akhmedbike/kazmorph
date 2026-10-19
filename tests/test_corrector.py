"""
Тесты для модуля корректора текста.

Запуск: pytest tests/test_corrector.py -v
"""

import pytest

from kazmorph.corrector import correct
from kazmorph.pipeline import Pipeline


@pytest.fixture
def pipeline():
    return Pipeline(crf_model_path=None, neural_model_path=None)


class TestCorrector:
    def test_output_structure(self, pipeline):
        result = correct(pipeline, "адам сөз")
        assert set(result.keys()) == {"original", "corrected", "corrections", "stats"}
        assert result["original"] == "адам сөз"
        assert isinstance(result["corrected"], str)
        assert isinstance(result["corrections"], list)
        assert set(result["stats"].keys()) == {"tokens", "corrected"}

    def test_token_count(self, pipeline):
        result = correct(pipeline, "адам сөз ақ")
        assert result["stats"]["tokens"] == 3

    def test_preserves_formatting(self, pipeline):
        text = "адам, сөз!"
        result = correct(pipeline, text)
        # знаки препинания и пробелы сохранены на тех же позициях
        assert "," in result["corrected"]
        assert "!" in result["corrected"]

    def test_preserves_whitespace(self, pipeline):
        text = "адам   сөз"
        result = correct(pipeline, text)
        # множественные пробелы сохранены
        assert "   " in result["corrected"]

    def test_empty_input(self, pipeline):
        result = correct(pipeline, "")
        assert result["corrected"] == ""
        assert result["corrections"] == []
        assert result["stats"]["tokens"] == 0

    def test_correction_has_position(self, pipeline):
        # заведомо «битое» слово должно либо получить исправление, либо остаться
        result = correct(pipeline, "адам xyzqwerty сөз")
        for c in result["corrections"]:
            assert "position" in c
            assert 0 <= c["position"] <= len(result["original"])
            assert "token" in c and "suggestion" in c

    def test_corrected_count_matches(self, pipeline):
        result = correct(pipeline, "адам сөз")
        assert result["stats"]["corrected"] == len(result["corrections"])

    def test_threshold_param(self, pipeline):
        # высокий порог — меньше замен; низкий — больше (границы допустимы)
        r_high = correct(pipeline, "адам сөз", confidence_threshold=0.9)
        r_low = correct(pipeline, "адам сөз", confidence_threshold=0.1)
        assert r_high["stats"]["corrected"] >= r_low["stats"]["corrected"]

    def test_multisentence_text_not_truncated(self, pipeline):
        # Регрессия: копия регулярки токенизации в corrector.py когда-то
        # расходилась с Pipeline._tokenize (склеивала «слово+точка»), из-за
        # чего многословный текст терял хвост при выравнивании результатов
        # пайплайна с матчами регулярки. Токенизация теперь общая
        # (pipeline.token_spans), хвост обязан сохраняться.
        text = "Мен бүгін мектепке бардым. Ол кітапты оқып жатыр. Қазақстан халқы бірлікті жақсы көреді."
        result = correct(pipeline, text)
        assert result["stats"]["tokens"] >= 16
        # хвост обязан сохраняться независимо от того, были замены или нет
        # (голый RMA-пайплайн может заменить часть слов — это отдельная
        # особенность эвристической уверенности, а не баг реконструкции)
        assert result["corrected"].rstrip().endswith("көреді.")
        assert " ." not in result["corrected"]

    def test_periods_stay_attached_to_preceding_word(self, pipeline):
        # Регрессия той же рассинхронизации: точки не должны «отрываться»
        # пробелом от предыдущего слова в исправленной строке.
        text = "Бірінші сөйлем. Екінші сөйлем."
        result = correct(pipeline, text)
        assert " ." not in result["corrected"]
