"""
Тесты интеграционного API v1.

Запуск: pytest tests/test_api_v1.py -v

Pipeline подменяется лёгким экземпляром (без моделей) через dependency_overrides,
чтобы тесты были быстрыми и не зависели от наличия обученных моделей.
"""

import pytest
from fastapi.testclient import TestClient

import main
from kazmorph.api_v1 import get_pipeline
from kazmorph.pipeline import Pipeline

SAMPLE_CONLLU = (
    "# sent_id = test:1\n# text = Адам келді .\n"
    "1\tадам\tадам\tNOUN\tn\tCase=Nom\t2\tnsubj\t_\t_\n"
    "2\tкелді\tкел\tVERB\tv\tMood=Ind|Tense=Past\t0\troot\t_\tSpaceAfter=No\n"
    "3\t.\t.\tPUNCT\tsent\t_\t2\tpunct\t_\t_\n\n"
)


@pytest.fixture
def client():
    """TestClient с переопределённым pipeline (без моделей, без запуска lifespan)."""
    test_pipeline = Pipeline(crf_model_path=None, neural_model_path=None)
    main.app.dependency_overrides[get_pipeline] = lambda: test_pipeline
    # Легаси-эндпоинт /api/analyze читает глобальный _pipeline — задаём и его.
    main._pipeline = test_pipeline
    # Без контекстного менеджера lifespan не запускается — модели не грузятся.
    c = TestClient(main.app)
    yield c
    main.app.dependency_overrides.clear()
    main._pipeline = None


# ============================================================
#  /api/v1/info
# ============================================================

class TestInfo:
    def test_info_fields(self, client):
        resp = client.get("/api/v1/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "KazMorphCorpus-2026"
        assert "version" in data
        assert isinstance(data["stages"], list)
        assert "RMA" in data["stages"] and "MFRN" in data["stages"]
        assert "models" in data
        assert "lexicon" in data
        assert data["lexicon"]["nouns"] > 0

    def test_info_reflects_no_models(self, client):
        # в тесте модели не загружены
        data = client.get("/api/v1/info").json()
        assert data["models"] == {"crf": False, "neural": False}


# ============================================================
#  /api/v1/correct
# ============================================================

class TestCorrect:
    def test_correct_returns_corrected(self, client):
        resp = client.post("/api/v1/correct", json={"text": "адам сөз"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["original"] == "адам сөз"
        assert "corrected" in data
        assert isinstance(data["corrections"], list)

    def test_correct_empty(self, client):
        resp = client.post("/api/v1/correct", json={"text": ""})
        assert resp.status_code == 200
        assert resp.json()["corrected"] == ""

    def test_correct_threshold_validation(self, client):
        # порог вне [0,1] → 422
        resp = client.post("/api/v1/correct", json={"text": "x", "confidence_threshold": 2.0})
        assert resp.status_code == 422


# ============================================================
#  /api/v1/analyze/batch
# ============================================================

class TestBatch:
    def test_batch_multiple(self, client):
        resp = client.post("/api/v1/analyze/batch",
                           json={"texts": ["адам сөз", "мектеп"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["tokens"][0]["token"] == "адам"

    def test_batch_requires_nonempty(self, client):
        resp = client.post("/api/v1/analyze/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_batch_text_field(self, client):
        data = client.post("/api/v1/analyze/batch", json={"texts": ["бір"]}).json()
        assert data["results"][0]["text"] == "бір"


# ============================================================
#  /api/v1/evaluate
# ============================================================

class TestEvaluate:
    def test_evaluate_identical(self, client):
        resp = client.post("/api/v1/evaluate",
                           json={"gold": SAMPLE_CONLLU, "pred": SAMPLE_CONLLU})
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["pos_accuracy"] == 1.0
        assert data["summary"]["gold_tokens"] == 3
        assert data["summary"]["aligned_tokens"] == 3
        assert data["summary"]["coverage"] == 1.0
        assert isinstance(data["per_tag"], list)
        assert isinstance(data["recommendations"], list)

    def test_evaluate_with_error(self, client):
        broken = SAMPLE_CONLLU.replace("1\tадам\tадам\tNOUN", "1\tадам\tадам\tVERB")
        resp = client.post("/api/v1/evaluate",
                           json={"gold": SAMPLE_CONLLU, "pred": broken})
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["pos_accuracy"] < 1.0
        assert data["summary"]["pos_errors"] >= 1

    def test_evaluate_empty_gold(self, client):
        resp = client.post("/api/v1/evaluate", json={"gold": "", "pred": SAMPLE_CONLLU})
        assert resp.status_code == 400

    def test_evaluate_top_confusions(self, client):
        broken = SAMPLE_CONLLU.replace("1\tадам\tадам\tNOUN", "1\tадам\tадам\tVERB")
        data = client.post("/api/v1/evaluate",
                           json={"gold": SAMPLE_CONLLU, "pred": broken}).json()
        assert any(c["gold"] == "NOUN" and c["pred"] == "VERB"
                   for c in data["top_confusions"])


# ============================================================
#  API key (env KAZMORPH_API_KEY)
# ============================================================

class TestApiKey:
    def test_no_key_required_by_default(self, client):
        resp = client.get("/api/v1/info")
        assert resp.status_code == 200

    def test_key_required_when_set(self, client, monkeypatch):
        monkeypatch.setenv("KAZMORPH_API_KEY", "secret123")
        resp = client.get("/api/v1/info")
        assert resp.status_code == 401

    def test_correct_key_passes(self, client, monkeypatch):
        monkeypatch.setenv("KAZMORPH_API_KEY", "secret123")
        resp = client.get("/api/v1/info", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200

    def test_wrong_key_rejected(self, client, monkeypatch):
        monkeypatch.setenv("KAZMORPH_API_KEY", "secret123")
        resp = client.get("/api/v1/info", headers={"X-API-Key": "nope"})
        assert resp.status_code == 401


# ============================================================
#  Обратная совместимость: старые эндпоинты живы
# ============================================================

class TestBackwardCompat:
    def test_legacy_analyze(self, client):
        resp = client.post("/api/analyze", json={"text": "адам сөз"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tokens"][0]["token"] == "адам"

    def test_legacy_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
