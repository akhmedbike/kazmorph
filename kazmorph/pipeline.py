"""
Full 4-stage morphological analysis pipeline.

Stage 1: RMA (Reverse Morphological Analyzer) — suffix decomposition
Stage 2: CRF — POS disambiguation with context window
Stage 3: Neural — transformer-based POS tagging
Stage 4: MFRN — morphophonological rule post-processing

Two operating modes, controlled by ``routing``:

  * ``routing=False`` (default) — a linear cascade RMA -> (CRF|skip) -> Neural
    -> Lemma -> MFRN (diagnostic). This is the production path and matches the
    historical behavior; the difficulty class is a *post-hoc* label assigned by
    :func:`kazmorph.taxonomy.classify`.

  * ``routing=True`` — the *error-taxonomy-driven routing* architecture
    described in the paper. All three POS components (RMA, CRF, Neural) are run
    for every token, the taxonomy scores are computed from the RMA result, and
    the final hypothesis is a confidence-weighted fusion whose weights depend
    on the assigned error class. This is the configuration used in the ablation
    study to measure the isolated contribution of routing-aware fusion.
"""

from __future__ import annotations

import re

from kazmorph.rma import ReverseMorphAnalyzer, RMAResult
from kazmorph.crf_tagger import CRFTagger
from kazmorph.neural import NeuralTagger
from kazmorph.neural_lemma import NeuralLemmatizer
from kazmorph.mfrn import MFRN
from kazmorph.taxonomy import classify, classify_scored, TaxonomyResult
from kazmorph.suggestor import suggest


# Known abbreviations whose trailing period is PART of the token (млрд., ж.,
# млн., долл., ...). Without this whitelist the tokenizer glued the
# sentence-final period onto ANY word ending in "." (келді., үлкен.), which
# broke POS/lemma on real user text. This is a closed class — real UD
# abbreviations with a period are all short and well-defined. Initials
# (А., М.) and multi-period forms (Б.з.б.) are handled structurally by the
# regex below and need not be listed here.
_ABBREVIATIONS: frozenset[str] = frozenset({
    "ж.", "млрд.", "млн.", "трлн.", "долл.", "драм.", "мыс.", "мөлш.",
    "реж.", "т.б.", "пр.", "проф.", "асс.", "доц.", "б.з.б.", "Б.з.б.",
    "қ.", "ғ.", "т.", "ш.", "б.",
})

# Первый проход токенизации: точка ВСЕГДА отделяется от слова. Паттерн вынесен
# на уровень модуля, чтобы корректор (kazmorph/corrector.py, api_v1.py) получал
# ровно те же границы токенов, что и Pipeline._tokenize, — расхождение копий
# регулярки приводило к потере хвоста текста и «уезжающим» точкам в /correct.
_TOKEN_FIRST_PASS = re.compile(
    r"\d+[.,]\d+%?"                                                    # decimal: 7,2 / 58,3%
    r"|\d+-[а-яәөүғқңһұі]+"                                            # numeral+suffix: 3-ші, 100-ге
    r"|[а-яәөүғқңһұіА-ЯӘӨҮҒҚҢҺҰІa-zA-Z]+-[а-яәөүғқңһұіА-ЯӘӨҮҒҚҢҺҰІa-zA-Z]+"  # hyphen: сондай-ақ
    r"|[а-яәөүғқңһұіА-ЯӘӨҮҒҚҢҺҰІa-zA-Z]+"                              # plain word (NO period)
    r"|\d+"                                                             # plain number
    r"|[^\s]",                                                          # single non-space char
)


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """Разбить текст на токены, вернув (токен, начало, конец) в поверхностном
    порядке. Единственный источник правил токенизации: Pipeline._tokenize
    возвращает токены этой функции, а корректоры (corrector.correct, API
    /correct3) используют её позиции для реконструкции строки — копий
    регулярки в других модулях быть не должно.
    """
    raw = [(m.group(0), m.start(), m.end()) for m in _TOKEN_FIRST_PASS.finditer(text)]
    spans: list[tuple[str, int, int]] = []
    i = 0
    while i < len(raw):
        tok, start, end = raw[i]
        # Второй проход: точка приклеивается назад только к известным
        # сокращениям (млрд., ж.) или одиночным заглавным инициалам (А.).
        if i + 1 < len(raw) and raw[i + 1][0] == ".":
            combo = tok + "."
            if (combo in _ABBREVIATIONS
                    or (len(tok) == 1 and tok.isupper() and tok.isalpha())):
                spans.append((combo, start, raw[i + 1][2]))
                i += 2
                continue
        spans.append((tok, start, end))
        i += 1
    return spans

# ── Routing / fusion parameters (paper, §"Threshold-Based Routing") ──────────
# Base fusion weights for the linear combination c* = w_FST·c_FST + w_CRF·c_CRF
# + w_NN·c_NN (eq. 5), chosen by grid search on the dev split.
DEFAULT_FUSION_WEIGHTS: tuple[float, float, float] = (0.30, 0.30, 0.40)
# Minimum confidence improvement required before a non-default component wins.
HYBRID_CONF_MARGIN: float = 0.05
# Lower confidence bound for the Neural component in the AFC class.
HYBRID_NEURAL_FLOOR: float = 0.40
# Active-learning confidence threshold (tokens below this are flagged for
# human annotation). Used by the /low_confidence endpoint and the AL simulation.
ACTIVE_LEARNING_THRESHOLD: float = 0.70

# ── Hard neural fallback parameters (paper, config F) ─────────────────────────
# The literature-endorsed alternative to confidence-weighted fusion/routing:
# the CRF is the default POS source, and the neural tagger contributes ONLY for
# tokens the CRF leaves unknown or low-confidence. This is the inverse of the
# routed mode — the transformer never outvotes a confident CRF, it only fills
# in where the CRF is unsure (see Discussion §5, "known anti-pattern").
FALLBACK_CONF_THRESHOLD: float = 0.50
FALLBACK_UNKNOWN_POS: tuple[str, ...] = ("UNK", "X", "")
# Per-class overrides of the base fusion weights (paper Table 4 routing column).
# Each entry replaces (w_FST, w_CRF, w_NN) for tokens of that error class.
CLASS_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "BOR": (0.20, 0.20, 0.60),   # borrowings/OOV -> trust the neural subword model
    "AFC": (0.15, 0.45, 0.40),   # long affix chains -> trust contextual CRF
    "SEG": (0.50, 0.30, 0.20),   # segmentation -> trust RMA boundary knowledge
    "AGR": (0.25, 0.25, 0.50),   # agreement -> neural reranking
    "NONE": DEFAULT_FUSION_WEIGHTS,
}


class Pipeline:
    """Full 4-stage Kazakh morphological analysis pipeline."""

    def __init__(
        self,
        crf_model_path: str | None = None,
        neural_model_path: str | None = None,
        lemma_model_path: str | None = None,
        *,
        routing: bool = False,
        fallback: bool = False,
        mfrn_scoring: bool = False,
        fusion_weights: tuple[float, float, float] = DEFAULT_FUSION_WEIGHTS,
        fallback_conf_threshold: float = FALLBACK_CONF_THRESHOLD,
    ):
        self.rma = ReverseMorphAnalyzer()
        self.crf = CRFTagger(model_path=crf_model_path)
        self.neural = NeuralTagger(model_path=neural_model_path)
        self.lemmatizer = NeuralLemmatizer(model_path=lemma_model_path)
        self.mfrn = MFRN(scoring=mfrn_scoring)
        # When True, run all three POS components and fuse their outputs with
        # error-class-conditioned weights (paper architecture). When False,
        # keep the legacy linear cascade.
        self.routing = routing
        # When True, use the hard neural-fallback mode (config F): CRF is the
        # default POS source; the neural tagger is taken only for tokens the
        # CRF leaves unknown or low-confidence. Mutually exclusive with
        # ``routing`` in practice (``fallback`` takes precedence if both set).
        self.fallback = fallback
        self.fallback_conf_threshold = fallback_conf_threshold
        self.fallback_unknown_pos = FALLBACK_UNKNOWN_POS
        self.fusion_weights = fusion_weights
        # Warn once if no learned models are loaded: a bare Pipeline() runs as
        # RMA-only (the rule-based analyzer), which has different — and lower —
        # POS accuracy than the trained CRF/Neural stages. This is easy to hit
        # by accident (e.g. demo.py, coverage_check.py) and silently produces
        # stub-quality output. Point users at the trained model paths.
        loaded = sum(1 for t in (self.crf, self.neural, self.lemmatizer)
                     if getattr(t, "model", None) is not None)
        if loaded == 0:
            import warnings
            warnings.warn(
                "Pipeline() loaded with NO trained models (CRF/Neural/Lemma all "
                "in stub mode). Output is RMA-only (rule-based), which gives "
                "lower POS accuracy than the trained pipeline. For full results, "
                "pass model paths: Pipeline(crf_model_path='models/crf_model.joblib', "
                "neural_model_path='models/neural_pos', lemma_model_path='models/neural_lemma').",
                stacklevel=2,
            )

    def process(self, text: str) -> list[dict]:
        """
        Process Kazakh text through the full pipeline.

        Returns list of token dicts with analysis results.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return []

        # Stage 1: RMA — morphological decomposition (always runs; produces the
        # surface features the taxonomy scores from).
        rma_results = [self.rma.analyze(t) for t in tokens]

        if self.fallback:
            final_results = self._process_fallback(rma_results)
        elif self.routing:
            final_results = self._process_routed(rma_results)
        else:
            final_results = self._process_cascade(rma_results)

        # Taxonomy classification + correction suggestions
        output = []
        for result in final_results:
            tax = classify(result)
            suggestions = suggest(result.token, result)
            output.append({
                "token": result.token,
                "lemma": result.lemma,
                "pos": result.pos,
                "suffixes": result.suffixes,
                "suffix_categories": result.suffix_categories,
                "confidence": result.confidence,
                "difficulty_class": tax.difficulty_class,
                "description": tax.description,
                "suggestions": suggestions,
            })

        return output

    # ------------------------------------------------------------------
    # Linear cascade (legacy / production path)
    # ------------------------------------------------------------------
    def _process_cascade(self, rma_results: list[RMAResult]) -> list[RMAResult]:
        # Stage 2: CRF — POS disambiguation (skip if Neural is active,
        # since the CRF trained on a small corpus hurts accuracy once a
        # transformer is available).
        if self.neural.model is not None:
            intermediate = rma_results
        else:
            intermediate = self.crf.tag(rma_results)

        # Stage 3: Neural — transformer POS tagging
        neural_results = self.neural.tag(intermediate)

        # Stage 3b: Neural lemmatizer — rewrite lemma via prefix-length
        # prediction. Runs after POS so it can stay POS-agnostic, and before
        # MFRN so MFRN sees corrected lemmas.
        lemma_results = self.lemmatizer.tag(neural_results)

        # Stage 4: MFRN — morphophonological rules (diagnostic or scoring)
        return self.mfrn.apply(lemma_results)

    # ------------------------------------------------------------------
    # Error-taxonomy-driven routing + fusion (paper architecture)
    # ------------------------------------------------------------------
    def _process_routed(self, rma_results: list[RMAResult]) -> list[RMAResult]:
        """
        Run all three POS components and fuse their per-token hypotheses.

        For each token the RMA result is taxonomy-scored; the resulting error
        class selects the fusion weights (CLASS_WEIGHTS). The fused POS is the
        argmax-vote across components weighted by (confidence, weight); the
        fused confidence is the weighted sum of the three component confidences
        (paper eq. 5). Lemmatization and MFRN then run on the fused result, as
        in the cascade path, so the only difference between the two modes is
        the POS-selection mechanism.
        """
        # Run the CRF and Neural components independently on the RMA features.
        # Each component returns a fresh RMAResult list with its own POS/confidence.
        crf_results = self.crf.tag(rma_results)
        neural_results = self.neural.tag(crf_results)

        fused: list[RMAResult] = []
        for rma_r, crf_r, neu_r in zip(rma_results, crf_results, neural_results):
            # Taxonomy signal is computed from the RMA result (surface features)
            # with the MFRN kappa where available. In routed mode MFRN runs
            # after fusion, so kappa defaults to a confidence proxy here.
            tax = classify_scored(rma_r)
            w_fst, w_crf, w_nn = CLASS_WEIGHTS.get(
                tax.difficulty_class, self.fusion_weights
            )

            components = [
                (rma_r.pos, rma_r.confidence, w_fst),
                (crf_r.pos, crf_r.confidence, w_crf),
                (neu_r.pos, neu_r.confidence, w_nn),
            ]
            pos, fused_conf = self._weighted_vote(components)

            # Preserve RMA's morphological decomposition and the neural lemma
            # candidate; only POS + confidence come from the fusion.
            neu_floor = HYBRID_NEURAL_FLOOR if tax.difficulty_class == "AFC" else 0.0
            fused_conf = max(fused_conf, neu_floor)
            fused_conf = min(fused_conf, 1.0)

            fused.append(RMAResult(
                token=rma_r.token,
                lemma=neu_r.lemma or rma_r.lemma,
                pos=pos,
                suffixes=rma_r.suffixes,
                suffix_categories=rma_r.suffix_categories,
                confidence=fused_conf,
            ))

        # Lemma + MFRN run on the fused stream, identical to the cascade path.
        lemma_results = self.lemmatizer.tag(fused)
        return self.mfrn.apply(lemma_results)

    # ------------------------------------------------------------------
    # Hard neural fallback (paper config F)
    # ------------------------------------------------------------------
    def _process_fallback(self, rma_results: list[RMAResult]) -> list[RMAResult]:
        """
        Hard neural fallback — the literature-endorsed alternative to fusion.

        The CRF is the default POS source (it is the strongest POS component
        on this corpus). The neural tagger runs once on the whole sequence,
        but its POS is taken ONLY for tokens the CRF leaves unknown
        (UNK / X) or labels with marginal confidence below
        ``fallback_conf_threshold``. On every other token the CRF decision
        stands. This is the inverse of :meth:`_process_routed`: instead of
        letting an over-confident transformer outvote a better-calibrated
        CRF, the transformer only ever contributes where the CRF is unsure.

        Lemma is taken from the RMA lexicon (the most accurate lemma source
        on this corpus); the neural lemmatizer is intentionally NOT invoked
        here, so that config F differs from config B (RMA+CRF) only in the
        POS source — isolating the effect of the neural fallback.
        """
        crf_results = self.crf.tag(rma_results)
        # Snapshot the CRF decisions BEFORE the neural pass: NeuralTagger.tag()
        # updates the RMAResult objects in place, so reading crf_r.pos after
        # self.neural.tag() would see the neural's overridden POS and silently
        # turn this fallback into a full neural override (i.e. config C).
        crf_decisions = [(r.pos, r.confidence) for r in crf_results]
        neural_results = self.neural.tag(crf_results)

        fused: list[RMAResult] = []
        neural_taken = 0
        for i, (rma_r, neu_r) in enumerate(zip(rma_results, neural_results)):
            crf_pos, crf_conf = crf_decisions[i]
            needs_neural = (
                crf_pos in self.fallback_unknown_pos
                or crf_conf < self.fallback_conf_threshold
            )
            if needs_neural and neu_r.pos not in self.fallback_unknown_pos:
                pos, conf = neu_r.pos, neu_r.confidence
                neural_taken += 1
            else:
                pos, conf = crf_pos, crf_conf

            fused.append(RMAResult(
                token=rma_r.token,
                lemma=rma_r.lemma,
                pos=pos,
                suffixes=rma_r.suffixes,
                suffix_categories=rma_r.suffix_categories,
                confidence=conf,
            ))

        # Instrumentation for the paper: how many tokens the neural tagger
        # actually decided (share of fallback-routed tokens). Counters
        # accumulate across process() calls; evaluate.py reads them after the
        # full corpus pass. Purely additive.
        prev = getattr(self, "last_fallback_stats",
                       {"total": 0, "neural_candidates": 0, "neural_taken": 0})
        self.last_fallback_stats = {
            "total": prev["total"] + len(rma_results),
            "neural_candidates": prev["neural_candidates"] + sum(
                1 for p, c in crf_decisions
                if p in self.fallback_unknown_pos
                or c < self.fallback_conf_threshold
            ),
            "neural_taken": prev["neural_taken"] + neural_taken,
        }

        return self.mfrn.apply(fused)

    @staticmethod
    def _weighted_vote(
        components: list[tuple[str, float, float]],
    ) -> tuple[str, float]:
        """
        Aggregate three (pos, confidence, weight) votes into one POS + score.

        POS is chosen by the highest total weighted confidence mass; the fused
        confidence is the weighted sum of confidences for the winning POS
        (paper eq. 5), normalized by the total weight so it stays in [0, 1].
        Ties are broken in insertion order (RMA, CRF, Neural), which favors the
        most interpretable component.
        """
        mass: dict[str, float] = {}
        total_w = 0.0
        for pos, conf, w in components:
            mass[pos] = mass.get(pos, 0.0) + conf * w
            total_w += w
        if total_w <= 0:
            # Degenerate fallback: take the first component.
            return components[0][0], components[0][1]
        # Argmax preserving insertion order for determinism on ties.
        best_pos = max(mass, key=lambda p: mass[p])
        fused_conf = mass[best_pos] / total_w
        return best_pos, fused_conf

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize Kazakh text into words and punctuation.

        Matches the UD Kazakh-KTB tokenization as closely as possible, so the
        tagger sees the same token shapes at inference time that it was trained
        on. Four patterns that a naive [word]+[punct] split gets wrong and that
        appear in the training data:

        - Decimal numbers with a comma: "7,2", "58,3%" (one token, not three).
        - Ordinals/numerals with a hyphenated suffix: "3-ші", "100-ге", "20-дан"
          (Kazakh attaches case/ordinal suffixes to numerals via hyphen).
        - Hyphenated compounds: "сондай-ақ", "әке-шешесін", "қарым-қатынас".
        - Abbreviations with a trailing period: "млрд.", "ж.", initials "А.".
          Without this, "млрд." becomes "млрд" + "." and the tagger misfires.

        Note on ordering in the alternation: longer/more-specific patterns come
        first so the regex engine commits to them before the generic word/punct
        fallback fires. MWEs like "болған емес" are intentionally NOT merged —
        they are a UD-specific convention and should be separate words in
        production text.
        """
        # Делегирует module-level token_spans — двухпроходная схема: точка
        # всегда отделена от слова, кроме известных сокращений (млрд., ж.)
        # и одиночных заглавных инициалов (А.). Единственный источник правил
        # токенизации — копий регулярки в других модулях быть не должно.
        return [tok for tok, _, _ in token_spans(text)]
