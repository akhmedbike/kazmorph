"""
Тесты для модуля оценки качества разметки (ТЗ, ЧАСТЬ A).

Запуск: pytest tests/test_evaluation.py -v
"""

import os
import tempfile

import pytest

from kazmorph.evaluation import (
    AlignedToken,
    ClassMetrics,
    ConfusionMatrix,
    ErrorAnalysis,
    QualityReport,
    align_sentences,
    analyze_errors,
    compute_per_text,
    compute_prf,
    generate_recommendations,
    lemma_accuracy,
    parse_conllu,
)

# ============================================================
#  CoNLL-U parsing
# ============================================================

SAMPLE_CONLLU = """\
# sent_id = test:1
# text = Адам келді .
1	адам	адам	NOUN	n	Case=Nom	2	nsubj	_	_
2	келді	кел	VERB	v	Mood=Ind|Tense=Past	0	root	_	SpaceAfter=No
3	.	.	PUNCT	sent	_	2	punct	_	_

# sent_id = test:2
# text = Ол үйге барды .
1	ол	ол	PRON	prn	Case=Nom|PronType=Prs	3	nsubj	_	_
2	үйге	үй	NOUN	n	Case=Dat	3	obl	_	_
3	барды	бар	VERB	v	Mood=Ind|Tense=Past	0	root	_	SpaceAfter=No
4	.	.	PUNCT	sent	_	3	punct	_	_
"""


def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".conllu")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestParser:
    def test_parse_conllu_sentences(self):
        sentences = parse_conllu(_write_tmp(SAMPLE_CONLLU))
        assert len(sentences) == 2
        assert sentences[0].sent_id == "test:1"
        assert sentences[0].text == "Адам келді ."
        assert len(sentences[0].tokens) == 3
        assert sentences[0].tokens[0].form == "адам"
        assert sentences[0].tokens[0].upos == "NOUN"
        assert sentences[0].tokens[0].lemma == "адам"
        assert sentences[0].tokens[0].feats == "Case=Nom"

    def test_parse_skips_multiword(self):
        conllu = (
            "# text = a b\n"
            "1-2\tab\t_	_	_	_	_	_	_	_\n"
            "1\ta\ta\tNOUN\t_	_	_	_	_	_\n"
            "2\tb\tb\tVERB\t_	_	_	_	_	_\n\n"
        )
        sentences = parse_conllu(_write_tmp(conllu))
        assert len(sentences[0].tokens) == 2  # multiword row skipped


# ============================================================
#  Alignment
# ============================================================

class TestAlignment:
    def test_align_positional(self):
        gold = parse_conllu(_write_tmp(SAMPLE_CONLLU))
        pred = parse_conllu(_write_tmp(SAMPLE_CONLLU))  # identical
        aligned = align_sentences(gold, pred)
        assert len(aligned) == 7
        assert aligned[0].gold_pos == "NOUN"
        assert aligned[0].pred_pos == "NOUN"
        assert aligned[0].pos_correct

    def test_align_detects_errors(self):
        gold = parse_conllu(_write_tmp(SAMPLE_CONLLU))
        # Искажаем предсказание: первый токен — VERB вместо NOUN
        pred = parse_conllu(_write_tmp(SAMPLE_CONLLU))
        pred[0].tokens[0].upos = "VERB"
        aligned = align_sentences(gold, pred)
        assert aligned[0].gold_pos == "NOUN"
        assert aligned[0].pred_pos == "VERB"
        assert not aligned[0].pos_correct

    def test_align_truncates_extra(self):
        gold = parse_conllu(_write_tmp(SAMPLE_CONLLU))
        pred = parse_conllu(_write_tmp(SAMPLE_CONLLU))
        # добавляем лишний токен в предсказание
        from kazmorph.evaluation import ConlluToken
        pred[0].tokens.append(ConlluToken("99", "x", "x", "X", "_", "_"))
        aligned = align_sentences(gold, pred)
        # лишний токен не попадает в выравнивание
        assert len([a for a in aligned if a.sent_id == "test:1"]) == 3


# ============================================================
#  Metrics: P/R/F
# ============================================================

class TestMetrics:
    def test_perfect_prediction(self):
        m = compute_prf(["NOUN", "VERB", "NOUN"], ["NOUN", "VERB", "NOUN"])
        assert m.accuracy == 1.0
        assert m.macro_f1 == 1.0
        assert m.per_tag["NOUN"].precision == 1.0
        assert m.per_tag["NOUN"].recall == 1.0

    def test_precision_recall_basic(self):
        # gold: NOUN NOUN VERB; pred: NOUN VERB VERB
        m = compute_prf(["NOUN", "NOUN", "VERB"], ["NOUN", "VERB", "VERB"])
        # NOUN: tp=1, fp=0, fn=1 → P=1.0, R=0.5
        assert m.per_tag["NOUN"].precision == 1.0
        assert m.per_tag["NOUN"].recall == 0.5
        assert m.per_tag["NOUN"].f1 == pytest.approx(2 / 3)
        assert m.accuracy == pytest.approx(2 / 3)

    def test_empty_labels(self):
        m = compute_prf([], [])
        assert m.accuracy == 0.0
        assert m.support == 0

    def test_macro_vs_weighted(self):
        # несбалансированный класс: VERB×1 (угадан), NOUN×9 (1 ошибка)
        gold = ["VERB"] + ["NOUN"] * 9
        pred = ["VERB"] + ["NOUN"] * 8 + ["VERB"]
        m = compute_prf(gold, pred)
        # weighted ближе к accuracy (доминирует NOUN)
        assert m.weighted_f1 >= m.macro_f1

    def test_micro_equals_accuracy(self):
        m = compute_prf(["NOUN", "VERB"], ["NOUN", "NOUN"])
        assert m.micro_f1 == m.accuracy

    def test_class_table_sorted_by_support(self):
        m = compute_prf(["NOUN"] * 5 + ["VERB"] * 2, ["NOUN"] * 5 + ["VERB"] * 2)
        rows = m.class_table()
        assert rows[0].tag == "NOUN"
        assert rows[0].support == 5

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_prf(["NOUN"], ["NOUN", "VERB"])


# ============================================================
#  Confusion matrix
# ============================================================

class TestConfusion:
    def test_counts_and_diagonal(self):
        cm = ConfusionMatrix()
        for g, p in [("NOUN", "NOUN"), ("NOUN", "PROPN"), ("VERB", "VERB")]:
            cm.add(g, p)
        assert cm.total == 3
        assert cm.diagonal == 2
        assert cm.errors == 1
        assert cm.at("NOUN", "PROPN") == 1

    def test_top_confusions_excludes_diagonal(self):
        cm = ConfusionMatrix()
        for g, p in [("NOUN", "NOUN")] * 5 + [("NOUN", "VERB")] * 3:
            cm.add(g, p)
        confs = cm.top_confusions(5)
        assert confs == [("NOUN", "VERB", 3)]

    def test_all_tags(self):
        cm = ConfusionMatrix()
        cm.add("A", "B")
        assert set(cm.all_tags) == {"A", "B"}

    def test_table_render(self):
        cm = ConfusionMatrix()
        cm.add("NOUN", "NOUN")
        table = cm.as_table()
        assert "NOUN" in table


# ============================================================
#  Error analysis
# ============================================================

def _make_aligned(gold_pos, pred_pos, form="x", lemma="x", gold_lemma="x"):
    return AlignedToken(
        sent_id="s1", token_id="1", form=form,
        gold_pos=gold_pos, pred_pos=pred_pos,
        gold_lemma=gold_lemma, pred_lemma=lemma,
        gold_feats="_", pred_feats="_",
    )


class TestErrorAnalysis:
    def test_analyze_counts(self):
        tokens = [
            _make_aligned("NOUN", "NOUN", "адам"),
            _make_aligned("VERB", "NOUN", "келді"),
            _make_aligned("NOUN", "PROPN", "Алматы"),
        ]
        cm = ConfusionMatrix()
        for t in tokens:
            cm.add(t.gold_pos, t.pred_pos)
        ea = analyze_errors(tokens, gold_total=3, confusion=cm)
        assert ea.pos_errors == 2
        assert ea.pos_error_rate == pytest.approx(2 / 3)
        assert ea.coverage == 1.0
        # частые спутанные пары отсортированы
        assert ea.pos_confusions[0].count >= ea.pos_confusions[-1].count

    def test_lemma_errors(self):
        tokens = [
            _make_aligned("NOUN", "NOUN", form="адамдар", gold_lemma="адам", lemma="адамд"),
        ]
        cm = ConfusionMatrix()
        cm.add("NOUN", "NOUN")
        ea = analyze_errors(tokens, gold_total=1, confusion=cm)
        assert ea.lemma_errors == 1
        assert len(ea.lemma_examples) == 1

    def test_punct_excluded_from_lemma(self):
        tokens = [_make_aligned("PUNCT", "PUNCT", form=".", lemma="_", gold_lemma=".")]
        cm = ConfusionMatrix()
        cm.add("PUNCT", "PUNCT")
        ea = analyze_errors(tokens, gold_total=1, confusion=cm)
        # пунктуация не считается ошибкой леммы
        assert ea.lemma_errors == 0

    def test_coverage_loss(self):
        tokens = [_make_aligned("NOUN", "NOUN")]
        cm = ConfusionMatrix()
        cm.add("NOUN", "NOUN")
        ea = analyze_errors(tokens, gold_total=5, confusion=cm)
        assert ea.coverage == pytest.approx(1 / 5)
        assert ea.coverage_loss == pytest.approx(4 / 5)


# ============================================================
#  Recommendations
# ============================================================

class TestRecommendations:
    def test_generates_for_low_recall(self):
        # VERB всегда ошибается (recall=0)
        gold = ["VERB", "VERB", "NOUN", "NOUN", "NOUN", "NOUN", "NOUN"]
        pred = ["NOUN", "NOUN", "NOUN", "NOUN", "NOUN", "NOUN", "NOUN"]
        m = compute_prf(gold, pred)
        cm = ConfusionMatrix()
        for g, p in zip(gold, pred):
            cm.add(g, p)
        ea = analyze_errors(
            [_make_aligned(g, p) for g, p in zip(gold, pred)],
            gold_total=len(gold), confusion=cm,
        )
        recs = generate_recommendations(m, cm, ea, min_support=2)
        titles = [r.title for r in recs]
        assert any("VERB" in t and "recall" in t.lower() for t in titles)

    def test_no_recommendations_when_perfect(self):
        gold = ["NOUN"] * 10
        pred = ["NOUN"] * 10
        m = compute_prf(gold, pred)
        cm = ConfusionMatrix()
        for g, p in zip(gold, pred):
            cm.add(g, p)
        ea = analyze_errors(
            [_make_aligned(g, p) for g, p in zip(gold, pred)],
            gold_total=10, confusion=cm,
        )
        recs = generate_recommendations(m, cm, ea, min_support=2)
        assert recs == []

    def test_recommendations_sorted_by_severity(self):
        gold = ["VERB", "NOUN", "NOUN", "NOUN", "NOUN"]
        pred = ["X", "NOUN", "NOUN", "NOUN", "NOUN"]
        m = compute_prf(gold, pred)
        cm = ConfusionMatrix()
        for g, p in zip(gold, pred):
            cm.add(g, p)
        ea = analyze_errors(
            [_make_aligned(g, p) for g, p in zip(gold, pred)],
            gold_total=5, confusion=cm,
        )
        recs = generate_recommendations(m, cm, ea, min_support=1)
        order = {"high": 0, "medium": 1, "low": 2}
        sev = [order[r.severity] for r in recs]
        assert sev == sorted(sev)

    def test_severity_thresholds(self):
        from kazmorph.evaluation import _severity_from_score
        assert _severity_from_score(0.3) == "high"
        assert _severity_from_score(0.6) == "medium"
        assert _severity_from_score(0.9) == "low"


# ============================================================
#  Per-text metrics
# ============================================================

class TestPerText:
    def test_per_text_groups_by_sentence(self):
        tokens = [
            AlignedToken("s1", "1", "адам", "NOUN", "NOUN", "адам", "адам", "_", "_"),
            AlignedToken("s1", "2", "келді", "VERB", "NOUN", "кел", "келді", "_", "_"),
            AlignedToken("s2", "1", "үй", "NOUN", "NOUN", "үй", "үй", "_", "_"),
        ]
        per_text = compute_per_text(tokens)
        assert len(per_text) == 2
        by_id = {tm.sent_id: tm for tm in per_text}
        assert by_id["s1"].tokens == 2
        assert by_id["s1"].pos_accuracy == 0.5
        assert by_id["s2"].pos_accuracy == 1.0

    def test_lemma_accuracy_helper(self):
        tokens = [
            AlignedToken("s", "1", "x", "NOUN", "NOUN", "адам", "адам", "_", "_"),
            AlignedToken("s", "2", "x", "NOUN", "NOUN", "адам", "бала", "_", "_"),
        ]
        assert lemma_accuracy(tokens) == 0.5


# ============================================================
#  QualityReport integration
# ============================================================

class TestQualityReport:
    def test_full_report_pipeline(self):
        gold_path = _write_tmp(SAMPLE_CONLLU)
        # системная разметка с одной ошибкой
        sys_conllu = SAMPLE_CONLLU.replace(
            "1\tадам\tадам\tNOUN", "1\tадам\tадам\tVERB"
        )
        pred_path = _write_tmp(sys_conllu)

        report = QualityReport(gold_path, pred_path)
        report.evaluate(min_support=1)

        assert report.metrics is not None
        assert report.metrics.support == 7
        # одна ошибка из 7 токенов
        assert report.metrics.accuracy == pytest.approx(6 / 7)
        assert report.errors.pos_errors == 1
        assert len(report.recommendations) >= 0
        assert len(report.per_text) == 2

    def test_markdown_contains_sections(self):
        gold_path = _write_tmp(SAMPLE_CONLLU)
        pred_path = _write_tmp(SAMPLE_CONLLU)
        report = QualityReport(gold_path, pred_path)
        report.evaluate()
        md = report.to_markdown()
        assert "Сводные метрики" in md
        assert "P/R/F по частям речи" in md
        assert "Анализ ошибок" in md
        assert "Рекомендации" in md
        assert "по каждому тексту" in md

    def test_save_markdown_and_html(self):
        gold_path = _write_tmp(SAMPLE_CONLLU)
        pred_path = _write_tmp(SAMPLE_CONLLU)
        report = QualityReport(gold_path, pred_path)
        report.evaluate()

        md_path = tempfile.mktemp(suffix=".md")
        html_path = tempfile.mktemp(suffix=".html")
        report.save(md_path, fmt="md")
        report.save(html_path, fmt="html")

        assert os.path.exists(md_path)
        assert os.path.exists(html_path)
        html_content = open(html_path, encoding="utf-8").read()
        assert "<table>" in html_content
        assert "<html" in html_content

    def test_evaluate_before_render_raises(self):
        report = QualityReport("x", "y")
        with pytest.raises(RuntimeError):
            report.to_markdown()
