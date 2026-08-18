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
runs in three modes: a **linear cascade** (default), an **error-taxonomy-driven
fusion** (`routing=True`), and a **hard neural fallback** (`fallback=True`).

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
| **Neural POS** | `neural.py` | Transformer token classifier (KazBERT / XLM-RoBERTa / Kazakh RoBERTa) for 17 UPOS tags, confidence-calibrated via temperature scaling |
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

- **Cascade** (`routing=False`, default) — stages run in sequence; the CRF is
  bypassed once a neural POS model is loaded. On this corpus that is a
  liability, not a benefit: the cascade (D) coincides with RMA+Neural (C,
  89.59%) and stays 3.5 pp below RMA+CRF (B, 93.10%) — see configuration
  D vs. B in the results table.
- **Routed** (`routing=True`) — all POS components run, the per-token error
  class is computed, and the final POS/confidence is a confidence-weighted
  vote across RMA/CRF/Neural using per-class weights (`CLASS_WEIGHTS`).
- **Fallback** (`fallback=True`, ablation F) — CRF is the default POS source;
  the neural tagger contributes only for tokens the CRF leaves unknown or
  labels below `fallback_conf_threshold` (τ = 0.50, selected on dev). Lemma
  stays with the RMA lexicon, so F differs from RMA+CRF only in the POS
  source. Evaluate with `scripts/evaluate.py --config fallback
  --fallback-threshold T`.

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
removed 127 exact-duplicate sentences. Bootstrap 95% CIs use 10,000 iterations;
pairwise comparisons use McNemar's test (α = 0.05).

```bash
python scripts/evaluate.py          # POS/lemma accuracy, CI, McNemar, ablations A–F
python scripts/benchmark.py         # per-stage coverage and confidence
python scripts/benchmark.py --configs  # throughput/RAM per configuration (Table 7)
```

### Results

| Configuration | POS Acc | 95% CI | Lemma | Macro-F1 | W-F1 |
|---|---:|:---:|---:|---:|---:|
| **A** RMA-only (lexicon) | 92.53% | [91.22, 93.79] | 95.04% | 0.8588 | 0.9258 |
| **B** RMA + CRF | **93.10%** | [91.84, 94.35] | **95.04%** | **0.8853** | **0.9310** |
| **C** RMA + Neural | 89.59% | [88.08, 91.03] | 94.10% | 0.8300 | 0.8962 |
| **D** Hybrid (cascade) | 89.59% | [88.08, 91.03] | 94.10% | 0.8300 | 0.8962 |
| **E** Hybrid + routing | 89.84% | [88.33, 91.28] | 94.17% | 0.8332 | 0.8989 |
| **F** Hybrid + fallback | 93.04% | [91.78, 94.29] | 95.04% | 0.8872 | 0.9299 |
| **G** RMA + XLM-R-base | 89.65% | [88.14, 91.15] | 94.17% | 0.8330 | 0.8993 |
| **H** RMA + KazRoBERTa | 88.02% | [86.39, 89.59] | 94.17% | 0.8335 | 0.8804 |

All three transformer encoders are trained on data **identical to the CRF's**
(full 754-sentence `train.conllu`, selection on `dev.conllu`, LR grid
{2, 3, 5}×10⁻⁵ — 5×10⁻⁵ won for every encoder), with all RNGs seed-controlled.

### Key findings

1. **CRF outperforms every fine-tuned transformer tested, on every training
   seed**, under an honest, decontaminated evaluation: RMA+CRF (93.10%) beats
   the monolingual KazBERT (seed mean 89.17% ± 0.43 over 3 seeds), XLM-R-base
   (89.44% ± 0.25), and a Kazakh RoBERTa (87.26% ± 0.55), while running
   30–38× faster (median 6,304 vs 163–210 tok/s on CPU, 5 repetitions).
   Even the best individual seed run of each encoder stays significantly
   below B (p ≤ 1.4×10⁻⁵); encoders are seed-stable (sd ≤ 0.55 pp). The CRF's
   own gain over RMA-only (+0.6 pp) is not significant (p = 0.29).
2. **The RMA lexicon also beats the neural lemmatizer** (95.04% vs 94.10–94.17%),
   so the recommended configuration B is best on both POS and lemma.
3. **Routing (D vs. E) is a null result** (+0.25 pp, p = 0.22) — reported as
   a limitation; no nominal significance even before Holm correction.
4. **Hard neural fallback (F) does not improve on CRF-only** (93.04% vs
   93.10%, p = 1.00): the neural tagger is adopted for only ~0.5% of tokens
   (τ = 0.50) and moves 2 tokens either way; across τ ∈ {0.50–0.70} the
   fallback tracks B within ±0.2 pp — reinforcing the low-resource thesis.
5. Prior silver-augmented runs that reported POS ≈ 95% were **contaminated**
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
    crf.py               # experimental PyTorch linear-chain CRF head (not
                         #   used in the reported experiments)
    crf_tagger.py        # Stage 2 — sklearn-crfsuite POS tagger (the CRF
                         #   behind configurations B/D/E/F in the paper)
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

# Neural POS tagger — paper protocol (§3.5): aligned training on the FULL
# 754-sentence train.conllu, selection on dev.conllu, LR grid {2,3,5}e-5,
# all RNGs seed-controlled (XLM-R-base: "FacebookAI/xlm-roberta-base",
# Kazakh RoBERTa: "kz-transformers/kaz-roberta-conversational")
for lr in 2e-5 3e-5 5e-5; do
  python scripts/train_neural.py --base-model Eraly-ml/KazBERT \
      --data data/train.conllu --dev-file data/dev.conllu \
      --folds 1 --epochs 12 --seed 42 --lr $lr --model-dir models/sweep/$lr
done

# Neural lemmatizer (prefix-length classifier)
python scripts/train_lemma.py --folds 5 --epochs 12
```

The UD corpus is small (754 train sentences). Silver annotations from
[Apertium-kaz](https://github.com/apertium/apertium-kaz) can be mixed in via
`--silver`; they are first decontaminated against the test split to prevent
label leakage.

The entire experimental chain of the paper — the LR sweep for all three
encoders, promotion of the best checkpoints, temperature calibration, seed
replicas (seeds 1/2/42 per encoder, with guards against model-less fallback),
full evaluation, per-class McNemar tests, the 5-repetition benchmark, and
figure generation — is reproducible with a single command:

```bash
bash scripts/repro_seeded.sh
```

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
