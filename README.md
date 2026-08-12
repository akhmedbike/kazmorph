# KazMorphCorpus-2026

**A hybrid morphological analyzer and corrector for the Kazakh language.**

KazMorphCorpus-2026 is a morphological analysis pipeline for Kazakh that
combines a rule-based lexicon analyzer, statistical and neural taggers, a
morphophonological rule engine, and a text-correction layer. It provides a web
interface (with dark/light theme and Kazakh/Russian/English UI), a REST API for
programmatic access, and an interactive CLI. The system is designed to be
reproducible and evaluable end-to-end on the Universal Dependencies
Kazakh-KTB corpus.

---

## How to cite

### TODO. Will add citation after publication in journal.

---

## Architecture

The pipeline (`kazmorph/pipeline.py`) processes a text through up to five
stages, then classifies each token and generates correction candidates. It
runs in two modes: a **linear cascade** (default) and an **error-taxonomy-driven
fusion** (`routing=True`).

```
text → tokenize → RMA → [CRF] → [Neural POS] → [Neural lemma] → MFRN
                     │                                         │
                     └── routed mode: per-class weighted vote ──┘
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                  difficulty class            correction candidates
                  (BOR/AFC/SEG/AGR)           (SymSpell suggestions)
```

| Stage | Module | Description |
|-------|--------|-------------|
| **RMA** | `rma.py` | Reverse Morphological Analyzer — trie-based suffix stripping with lexicon and UD-lemma lookup |
| **CRF** | `crf_tagger.py` | sklearn-crfsuite POS disambiguator (~30 features, context window ±2). Bypassed in the cascade when a neural model is loaded |
| **Neural POS** | `neural.py` | Transformer token classifier (KazBERT / XLM-RoBERTa) for 17 UPOS tags, confidence-calibrated via temperature scaling |
| **Neural lemma** | `neural_lemma.py` | KazBERT classifier predicting the lemma prefix length, with deterministic repair of final-consonant devoicing (қ↔ғ, к↔г) |
| **MFRN** | `mfrn.py` | Morphophonological rule engine — 64 rules in 8 groups (vowel harmony, assimilation, devoicing, suffix selection, …) |

Each tagged token is classified into a **difficulty category** that drives
routing and surfaces in the output:

- **BOR** — borrowing (international vocabulary, consonant clusters)
- **AFC** — affix-chain complexity (3+ suffixes)
- **SEG** — segmentation ambiguity (multiple valid morpheme splits)
- **AGR** — agreement complexity (possessive + case stacking)

Every component degrades gracefully: if a model file is absent, that stage is
skipped and a bare `Pipeline()` runs RMA-only with a warning.

### Operating modes

- **Cascade** (`routing=False`, default) — stages run in sequence; CRF is
  skipped once a neural POS model is available (on this corpus the small-data
  CRF hurts accuracy once a transformer is in place).
- **Routed** (`routing=True`) — all POS components run, the per-token error
  class is computed, and the final POS/confidence is a confidence-weighted
  vote across RMA/CRF/Neural using per-class weights (`CLASS_WEIGHTS`).

### Correction layer

`kazmorph/corrector.py` rewrites a text by replacing tokens that are either
low-confidence (`< 0.85` by default) or tagged `UNK` with their top SymSpell
suggestion (`kazmorph/suggestor.py`), preserving whitespace and punctuation.
Exposed as `POST /api/v1/correct`.

### Confidence calibration

Neural POS confidence is temperature-scaled on a dev set
(`kazmorph/calibration.py`, Guo 2017) and read back at inference time. By
default the MFRN runs in diagnostic mode and leaves confidence intact, so the
calibrated probabilities are preserved end-to-end.

### Fully local analysis & pluggable reviewer

The analysis pipeline (`/api/analyze`, `/api/v1/correct`) is **fully local** —
it never calls any external service. The optional Active-Learning reviewer is a
**pluggable** LLM: Claude (`LLM_PROVIDER=claude`) is just one example. Set
`LLM_PROVIDER=ollama`, `LLM_REVIEW_MODEL=qwen2.5:7b`, and `OLLAMA_BASE_URL` to
run the reviewer on a **local** model (Qwen + Ollama), free of charge and with
zero external calls. Any OpenAI-compatible backend (Ollama, vLLM, LM Studio)
works — no code changes needed (`kazmorph/llm_review/client.py`).

---

## Reproducing the results

All metrics below are measured on the held-out `data/test.conllu` (162
sentences, 1594 tokens) after a silver-to-test **decontamination** step that
removed 127 exact-duplicate sentences. Bootstrap 95% CIs use 2000 iterations;
pairwise comparisons use McNemar's test (α = 0.05).

```bash
python scripts/evaluate.py          # POS/lemma accuracy, CI, McNemar, ablations A–E
python scripts/benchmark.py         # per-stage coverage and confidence
```

### Results

| Configuration | POS Acc | 95% CI | Lemma | Macro-F1 | W-F1 |
|---|---:|:---:|---:|---:|---:|
| **A** RMA-only (lexicon) | 92.53% | [91.28, 93.79] | 95.04% | 0.8083 | 0.9258 |
| **B** RMA + CRF | **93.10%** | [91.84, 94.29] | 95.04% | **0.8853** | **0.9310** |
| **C** RMA + Neural | 89.08% | [87.52, 90.59] | 90.84% | 0.8079 | 0.8898 |
| **D** Hybrid (cascade) | 89.08% | [87.52, 90.59] | 90.84% | 0.8079 | 0.8898 |
| **E** Hybrid + routing | 89.34% | [87.83, 90.84] | 90.84% | 0.8174 | 0.8925 |

### Key findings

1. **CRF outperforms the neural tagger on POS** under an honest,
   decontaminated evaluation: RMA+CRF (93.10%) beats RMA+Neural (89.08%) by
   ~4 points and is ~57× faster. The difference is statistically significant
   (McNemar χ² = 24.5, p = 10⁻⁶).
2. **The neural model adds a better lemmatizer** (90.84% vs. RMA's lexicon
   lookup), motivating the hybrid RMA+CRF (POS) + Neural (lemma) configuration.
3. **Routing (D vs. E) is a small, non-significant gain** (+0.26 pp, p = 0.13)
   on 1594 tokens — reported as a limitation.
4. Prior silver-augmented runs that reported POS ≈ 95% were **contaminated**
   by sentence overlap between silver data and the test split; closing that
   leak dropped neural POS by 5.7 points.

---

## Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
git clone https://github.com/akhmedbike/kazmorph.git
cd kazmorph

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Running

The app runs **with a database** (accounts, request history, active learning)
or **without** (stateless analysis only).

**Docker (recommended — includes PostgreSQL):**

```bash
cp .env.example .env      # fill in real values
docker compose up -d --build
```

**Local development (with a database):**

```bash
export DATABASE_URL="postgresql+psycopg2://kazmorph:kazmorph@localhost:5432/kazmorph"
export SESSION_SECRET="some-long-random-string"
alembic upgrade head
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

**Stateless mode (no database — analysis only):**

```bash
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

Open http://127.0.0.1:8080 in your browser, or use the CLI:

```bash
python demo.py
```

---

## API

### Analyze text

```
POST /api/analyze
Content-Type: application/json

{"text": "Қазақстан Республикасының Конституциясы"}
```

```json
{
  "tokens": [
    {
      "token": "Қазақстан",
      "lemma": "Қазақстан",
      "pos": "NOUN",
      "suffixes": [],
      "confidence": 0.65,
      "difficulty_class": "BOR",
      "description": "Borrowing detected",
      "suggestions": []
    }
  ]
}
```

### Versioned integration API (`/api/v1`)

Interactive Swagger docs at `/docs`. All v1 endpoints are guarded by an
optional API key (`X-API-Key`, set `KAZMORPH_API_KEY`).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/correct` | Correct text — returns corrected text plus per-token edits |
| `POST` | `/api/v1/analyze/batch` | Analyze several texts in one request |
| `POST` | `/api/v1/evaluate` | Evaluate gold vs. system CoNLL-U: P/R/F, confusion, error analysis |
| `GET`  | `/api/v1/info` | Service info: version, active stages, lexicon size |

```
GET /api/health
```

Returns active pipeline stages and server status.

---

## Project structure

```
main.py                  # FastAPI app: pages + /api/analyze + /api/health
kazmorph/
    pipeline.py          # 5-stage orchestrator (cascade / routed modes)
    rma.py               # Stage 1 — reverse-trie suffix stripping + lexicon
    crf.py               # PyTorch linear-chain CRF head
    crf_tagger.py        # Stage 2 — sklearn-crfsuite POS tagger
    neural.py            # Stage 3 — transformer UPOS tagger (calibrated)
    neural_lemma.py      # Stage 3b — KazBERT lemma classifier
    neural_lemma_seq2seq.py  # alternative mt5-small seq2seq lemmatizer
    mfrn.py              # Stage 4 — 64 morphophonological rules
    taxonomy.py          # BOR/AFC/SEG/AGR difficulty classifier
    corrector.py         # text-level correction
    suggestor.py         # SymSpell correction candidates
    calibration.py       # temperature scaling (Guo 2017)
    silver.py            # Apertium-kaz → silver CoNLL-U
    augment.py           # training-time data augmentation
    losses.py            # uncertainty-weighted multi-task loss (Kendall 2018)
    model_versioning.py  # versioned CRF store (active-learning retrain loop)
    api_v1.py            # REST API v1 router
app/                     # web layer: DB (SQLAlchemy), auth, dashboard, admin
scripts/                 # training, evaluation, data prep, active learning
data/                    # UD splits, silver CoNLL-U, lemma pairs
models/                  # trained artifacts (weights external — see below)
tests/                   # ~23 pytest suites (pipeline, taggers, AL loop, web)
```

---

## Training

```bash
# CRF POS tagger → models/crf_model.joblib
python scripts/train_crf.py

# Neural POS tagger (KazBERT, 5-fold CV, class weighting)
python scripts/train_neural.py --base-model Eraly-ml/KazBERT --folds 5 --epochs 12

# Neural lemmatizer (prefix-length classifier)
python scripts/train_lemma.py --folds 5 --epochs 12
```

The UD corpus is small (754 train sentences). Silver annotations from
[Apertium-kaz](https://github.com/apertium/apertium-kaz) can be mixed in via
`--silver`; they are first decontaminated against the test split to prevent
label leakage.

```bash
# Convert Apertium-kaz → silver CoNLL-U
python scripts/make_silver.py --tagged /path/to/apertium-kaz/corpus/kaz.tagged \
    --output data/silver_apertium.conllu --drop-unknown 0.5

# Calibrate neural POS confidence
python scripts/calibrate.py
```

---

## Data sources

| Source | Used for | License |
|--------|----------|---------|
| [UD Kazakh-KTB](https://github.com/UniversalDependencies/UD_Kazakh-KTB) | train/dev/test splits, evaluation | CC BY-SA 4.0 |
| [Apertium-kaz](https://github.com/apertium/apertium-kaz) | lexicons, silver UPOS & lemma data | GPLv2+ |
| [KazBERT](https://huggingface.co/Eraly-ml/KazBERT) (`Eraly-ml/KazBERT`) | base transformer for the neural models | see model card |

See [`DATA_LICENSE`](DATA_LICENSE) for full attribution and per-source terms.

---

## Large files / model weights

The two neural checkpoints are too large for Git (~420 MB each) and are
**excluded** via `.gitignore`:

| File | Size | Description |
|------|------|-------------|
| `models/neural_pos/model.safetensors` | ~420 MB | Fine-tuned KazBERT POS tagger |
| `models/neural_lemma/model.safetensors` | ~420 MB | Fine-tuned KazBERT lemmatizer |

**Download:** [Download model weights (Zenodo / Google Drive)](https://drive.google.com/drive/folders/1bwqtml8R47vs5oEVxAgOF8YckfsQ5umw?usp=sharing)

#TODO. First verson on google drive after all reviews replace to Zenodo.

Place the weights under `models/`. The CRF model (`models/crf_model.joblib`,
176 KB) and all small config files are included in the repository. Without the
neural weights, the pipeline degrades gracefully to RMA + CRF + MFRN.

---

## License

- **Source code** — [MIT License](LICENSE).
- **Data & model artifacts** — derived from UD Kazakh-KTB (CC BY-SA 4.0),
  Apertium-kaz (GPLv2+), and KazBERT. See [`DATA_LICENSE`](DATA_LICENSE).

---

## Funding

**Funding:** This research has been/was/is funded by the Science Committee of
the Ministry of Science and Higher Education of the Republic of Kazakhstan
with the grant “Innovative technologies for automated correction of Kazakh
language texts: machine learning and morphological analysis”
(Grant No. AP23487753).
