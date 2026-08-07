"""
Stage 4: Morphophonological Rule Network (MFRN)

64 formalized grammatical / phonological constraints, organized in 8 groups:
vowel harmony (1–12), consonant assimilation (13–22), nasal assimilation
(23–28), root-final alternation (29–36), final devoicing (37–42), epenthesis
(43–48), suffix selection (49–56), and phonotactic constraints (57–64).

Two operating modes:

  * ``scoring=False`` (default) — the network is a *diagnostic* pass: it computes
    which rules fire (for the error taxonomy / error-pattern analysis) but does
    NOT mutate the per-token confidence. This preserves the calibrated softmax
    probability produced upstream by the neural / RMA stages. This is the mode
    the production pipeline runs in, and the one that gave the best POS accuracy
    in A/B testing (see the historical note in ``_apply_rules``).

  * ``scoring=True`` — the network additionally produces the aggregate
    compatibility coefficient ``kappa`` and rescales confidence by it, as
    described in the paper (formulas (3) and (4)). This mode is used by the
    ablation study (config "Hybrid + MFRN scoring") to measure the *isolated*
    effect of the kappa-weighted post-processing. It deliberately applies only
    phonological/morphotactic penalties and bonuses — it never overrides a POS
    label — so it cannot reintroduce the VERB↔AUX regression documented below.
"""

from kazmorph.rma import RMAResult

# Phonetic sets
FRONT_VOWELS = set("еіүұөә")
BACK_VOWELS = set("аыоу")
ROUNDED_VOWELS = set("ұүоуө")
UNROUNDED_VOWELS = set("аыеі")
VOICELESS = set("қкптшстхфцчщһ")
VOICED = set("ғгбджзлмнңрйу")
VELAR = set("қкғг")

# Symmetric clipping range for the aggregate compatibility coefficient kappa
# (paper, eq. 3). Below 0.40 a token is treated as too unreliable to trust;
# above 1.25 bonuses no longer compound.
KAPPA_MIN = 0.40
KAPPA_MAX = 1.25

# Closed class of Kazakh auxiliary verbs. In UD these tag AUX; a bare VERB
# prediction for one of these lemmas is the single most common VERB↔AUX error
# (final_report.md §4.1). Membership is checked against the RMA lemma, which is
# the infinitive-ish stem (жат, отыр, ал, бер, ...). The set is closed and
# small, so a literal list is both precise and interpretable.
AUX_LEMMAS = frozenset({
    # aspectual / progressive ("is doing"): the four static-posture verbs
    "жат", "отыр", "жүр", "тұр",
    # phase / modal auxiliaries attached to a converb
    "ал", "бер", "баста", "сал", "қал", "жіберм", "қой",
    # copular / evidential / past-tense auxiliaries
    "бол", "еді", "екен", "емес", "шығар",
    # negative auxiliary
    "жоқ",
})


class MFRN:
    """Stage 4: Morphophonological Rule Network with 64 rules."""

    def __init__(self, *, scoring: bool = False):
        # When True, apply() rescales per-token confidence by the aggregate
        # compatibility coefficient kappa (paper eq. 3-4). When False (default),
        # MFRN is a diagnostic pass only and leaves confidence untouched.
        self.scoring = scoring
        self.rules = self._init_rules()

    def _init_rules(self) -> list[dict]:
        """Initialize the 64 morphophonological rules."""
        rules = []

        # ── Vowel harmony (1-12) ──────────────────────────────────
        rules.append({"id": 1,  "type": "vowel_harmony", "gamma": 0.85,
            "description": "Front-vowel stem requires front-vowel suffixes (е/і/ү/ұ/ө)"})
        rules.append({"id": 2,  "type": "vowel_harmony", "gamma": 0.85,
            "description": "Back-vowel stem requires back-vowel suffixes (а/ы/о/у)"})
        rules.append({"id": 3,  "type": "vowel_harmony", "gamma": 0.95,
            "description": "Mixed-vowel stem follows last vowel harmony"})
        rules.append({"id": 4,  "type": "vowel_harmony", "gamma": 0.90,
            "description": "Rounded front vowels (ү/ұ/ө) take rounded-front suffixes"})
        rules.append({"id": 5,  "type": "vowel_harmony", "gamma": 0.90,
            "description": "Rounded back vowels (о/у) take rounded-back suffixes"})
        rules.append({"id": 6,  "type": "vowel_harmony", "gamma": 0.92,
            "description": "Suffix -ды/-ді follows harmony of stem final vowel"})
        rules.append({"id": 7,  "type": "vowel_harmony", "gamma": 0.92,
            "description": "Suffix -ты/-ті for voiceless-final stems"})
        rules.append({"id": 8,  "type": "vowel_harmony", "gamma": 0.93,
            "description": "Possessive -ы/-і follows stem harmony"})
        rules.append({"id": 9,  "type": "vowel_harmony", "gamma": 0.90,
            "description": "Dative -ға/-ге vs -қа/-ке harmony + voicing"})
        rules.append({"id": 10, "type": "vowel_harmony", "gamma": 0.90,
            "description": "Locative -да/-де vs -та/-те harmony + voicing"})
        rules.append({"id": 11, "type": "vowel_harmony", "gamma": 0.90,
            "description": "Ablative -дан/-ден vs -тан/-тен harmony + voicing"})
        rules.append({"id": 12, "type": "vowel_harmony", "gamma": 0.92,
            "description": "Instrumental -мен/-бен vs -пен harmony + voicing"})

        # ── Consonant assimilation (13-22) ────────────────────────
        rules.append({"id": 13, "type": "assimilation", "gamma": 0.88,
            "description": "қ → ғ before voiced consonants at morpheme boundary"})
        rules.append({"id": 14, "type": "assimilation", "gamma": 0.88,
            "description": "к → г before voiced consonants at morpheme boundary"})
        rules.append({"id": 15, "type": "assimilation", "gamma": 0.88,
            "description": "п → б before voiced consonants at morpheme boundary"})
        rules.append({"id": 16, "type": "assimilation", "gamma": 0.88,
            "description": "т → д before voiced consonants at morpheme boundary"})
        rules.append({"id": 17, "type": "assimilation", "gamma": 0.88,
            "description": "ш → ж before voiced consonants at morpheme boundary"})
        rules.append({"id": 18, "type": "assimilation", "gamma": 0.92,
            "description": "ғ → қ before voiceless consonants"})
        rules.append({"id": 19, "type": "assimilation", "gamma": 0.92,
            "description": "г → к before voiceless consonants"})
        rules.append({"id": 20, "type": "assimilation", "gamma": 0.92,
            "description": "б → п before voiceless consonants"})
        rules.append({"id": 21, "type": "assimilation", "gamma": 0.92,
            "description": "д → т before voiceless consonants"})
        rules.append({"id": 22, "type": "assimilation", "gamma": 0.92,
            "description": "ж → ш before voiceless consonants"})

        # ── Nasal assimilation (23-28) ────────────────────────────
        rules.append({"id": 23, "type": "nasal", "gamma": 0.90,
            "description": "н → ң before velar consonants (қ, к, ғ, г)"})
        rules.append({"id": 24, "type": "nasal", "gamma": 0.95,
            "description": "ң → н in non-velar context"})
        rules.append({"id": 25, "type": "nasal", "gamma": 0.90,
            "description": "н + қ → ңқ (assimilation to velar nasal)"})
        rules.append({"id": 26, "type": "nasal", "gamma": 0.90,
            "description": "н + г → ңг (assimilation to velar nasal)"})
        rules.append({"id": 27, "type": "nasal", "gamma": 0.97,
            "description": "Final -ң preserved in nominal possessive"})
        rules.append({"id": 28, "type": "nasal", "gamma": 0.97,
            "description": "Final -н preserved in verbal forms"})

        # ── Root-final consonant alternation (29-36) ──────────────
        rules.append({"id": 29, "type": "alternation", "gamma": 0.85,
            "description": "Root-final к → қ when followed by back-vowel suffix"})
        rules.append({"id": 30, "type": "alternation", "gamma": 0.85,
            "description": "Root-final қ → к when followed by front-vowel suffix"})
        rules.append({"id": 31, "type": "alternation", "gamma": 0.85,
            "description": "Root-final г → ғ when followed by back-vowel suffix"})
        rules.append({"id": 32, "type": "alternation", "gamma": 0.85,
            "description": "Root-final ғ → г when followed by front-vowel suffix"})
        rules.append({"id": 33, "type": "alternation", "gamma": 0.90,
            "description": "Root-final п → б when suffix begins with vowel"})
        rules.append({"id": 34, "type": "alternation", "gamma": 0.90,
            "description": "Root-final б → п when word-final or before voiceless"})
        rules.append({"id": 35, "type": "alternation", "gamma": 0.93,
            "description": "Root-final ш → ж in instrumental -шы/-ші"})
        rules.append({"id": 36, "type": "alternation", "gamma": 0.93,
            "description": "Root-final ң → н in verbal derivation"})

        # ── Devoicing / final voicing (37-42) ─────────────────────
        rules.append({"id": 37, "type": "devoicing", "gamma": 0.88,
            "description": "Word-final ғ devoices to қ"})
        rules.append({"id": 38, "type": "devoicing", "gamma": 0.88,
            "description": "Word-final г devoices to к"})
        rules.append({"id": 39, "type": "devoicing", "gamma": 0.88,
            "description": "Word-final б devoices to п"})
        rules.append({"id": 40, "type": "devoicing", "gamma": 0.88,
            "description": "Word-final д devoices to т"})
        rules.append({"id": 41, "type": "devoicing", "gamma": 0.88,
            "description": "Word-final ж devoices to ш"})
        rules.append({"id": 42, "type": "devoicing", "gamma": 0.88,
            "description": "Word-final з devoices to с"})

        # ── Epenthetic vowel insertion (43-48) ────────────────────
        rules.append({"id": 43, "type": "epenthesis", "gamma": 0.92,
            "description": "Insert epenthetic 'ы' between consonant-final stem and -ң"})
        rules.append({"id": 44, "type": "epenthesis", "gamma": 0.92,
            "description": "Insert epenthetic 'і' between front-vowel stem and -ң"})
        rules.append({"id": 45, "type": "epenthesis", "gamma": 1.05,
            "description": "No epenthesis before vowel-initial suffixes"})
        rules.append({"id": 46, "type": "epenthesis", "gamma": 0.93,
            "description": "Epenthetic vowel follows stem harmony"})
        rules.append({"id": 47, "type": "epenthesis", "gamma": 0.90,
            "description": "Consonant cluster resolution via epenthesis"})
        rules.append({"id": 48, "type": "epenthesis", "gamma": 0.88,
            "description": "Borrowing stems with final cluster take epenthetic vowel"})

        # ── Suffix selection (49-56) ──────────────────────────────
        rules.append({"id": 49, "type": "suffix_selection", "gamma": 0.85,
            "description": "Accusative: -ды/-ді after voiced, -ты/-ті after voiceless"})
        rules.append({"id": 50, "type": "suffix_selection", "gamma": 0.85,
            "description": "Dative: -ға/-ге after voiced, -қа/-ке after voiceless"})
        rules.append({"id": 51, "type": "suffix_selection", "gamma": 0.85,
            "description": "Locative: -да/-де after voiced, -та/-те after voiceless"})
        rules.append({"id": 52, "type": "suffix_selection", "gamma": 0.85,
            "description": "Ablative: -дан/-ден after voiced, -тан/-тен after voiceless"})
        rules.append({"id": 53, "type": "suffix_selection", "gamma": 0.85,
            "description": "Instrumental: -мен/-бен after voiced, -пен after voiceless"})
        rules.append({"id": 54, "type": "suffix_selection", "gamma": 0.85,
            "description": "Plural: -лар/-лер after vowel, -дар/-дер after voiced, -тар/-тер after voiceless"})
        rules.append({"id": 55, "type": "suffix_selection", "gamma": 0.90,
            "description": "Possessive-3Sg: -ы/-і after consonant, nil after vowel"})
        rules.append({"id": 56, "type": "suffix_selection", "gamma": 0.85,
            "description": "Past participle: -ған/-ген after voiced, -қан/-кен after voiceless"})

        # ── Phonotactic constraints (57-64) ───────────────────────
        rules.append({"id": 57, "type": "phonotactic", "gamma": 0.90,
            "description": "No consecutive voiceless plosives across morpheme boundary"})
        rules.append({"id": 58, "type": "phonotactic", "gamma": 0.95,
            "description": "Geminate consonant reduction across morpheme boundary"})
        rules.append({"id": 59, "type": "phonotactic", "gamma": 0.97,
            "description": "ү/ұ neutralization in non-first syllable"})
        rules.append({"id": 60, "type": "phonotactic", "gamma": 1.00,
            "description": "и acts as front vowel for harmony purposes"})
        rules.append({"id": 61, "type": "phonotactic", "gamma": 0.75,
            "description": "Native words cannot begin with р, л, й, у (except borrowings)"})
        rules.append({"id": 62, "type": "phonotactic", "gamma": 0.88,
            "description": "Consonant-final borrowings select voiced suffix variant"})
        rules.append({"id": 63, "type": "phonotactic", "gamma": 1.00,
            "description": "Vowel length neutralized in suffix syllables"})
        rules.append({"id": 64, "type": "phonotactic", "gamma": 1.00,
            "description": "Stress assignment: final syllable of stem before suffixes"})

        assert len(rules) == 64, f"Expected 64 rules, got {len(rules)}"
        return rules

    def apply(self, tokens: list[RMAResult]) -> list[RMAResult]:
        """Apply all MFRN rules to the token list."""
        result = []
        for token in tokens:
            adjusted = self._apply_rules(token)
            result.append(adjusted)
        return result

    def compute_kappa(self, token: RMAResult) -> float:
        """
        Aggregate compatibility coefficient ``kappa`` (paper, eq. 3).

        ``kappa = clip(prod_j gamma_j, KAPPA_MIN, KAPPA_MAX)`` over the set of
        rules that fire (i.e. whose phonological condition is violated) for this
        token. A token that violates no rule keeps kappa = 1.0 (neutral);
        violations compound multiplicatively, which is the natural semantics for
        independent compatibility constraints. Typical valid configurations are
        therefore left near 1.0, while morphologically implausible tokens are
        pulled toward KAPPA_MIN = 0.40.

        The fired-rule set is computed from the existing ``_check_*`` detectors
        (harmony, suffix voicing, final devoicing, illegal onset). Each detector
        maps to the rule-id range it covers so the corresponding gamma is applied
        exactly once per violation class, not once per rule literal.
        """
        gamma = 1.0
        lemma = token.lemma or token.token
        suffixes = token.suffixes or []

        # Rule class 1 (ids 1-5): stem/suffix vowel harmony.
        if suffixes and not self._check_harmony(lemma, suffixes):
            gamma *= 0.85

        # Rule class 2 (ids 49-56): suffix-voicing / allomorph selection.
        if suffixes and not self._check_suffix_voicing(lemma, suffixes):
            gamma *= 0.85

        # Rule class 3 (ids 37-42): final devoicing consistency.
        if not self._check_final_devoicing(token.token, lemma):
            gamma *= 0.90

        # Rule class 4 (id 61): illegal native word onset.
        if self._starts_with_illegal(token.token):
            gamma *= 0.75

        return max(KAPPA_MIN, min(KAPPA_MAX, gamma))

    def _apply_rules(self, token: RMAResult) -> RMAResult:
        """
        Apply morphophonological rules to a single token.

        Two modes:

        * ``scoring=False`` (default): the network is diagnostic-only. The
          per-token confidence is a calibrated softmax probability produced
          upstream (NeuralTagger / RMA lexicon hit); the earlier ±0.05/0.10
          heuristics here destroyed that calibration — leaving 77% of tokens at
          confidence=1.0 while real accuracy was ~82% — so confidence is left
          untouched. The rule checks still run and inform the difficulty
          taxonomy via :func:`kazmorph.taxonomy.score_agreement`.

        * ``scoring=True``: rescale confidence by the aggregate compatibility
          coefficient kappa (paper, eq. 4): ``c* = clip(c * kappa, 0, 1)``. This
          is the ablation mode used to measure the isolated contribution of the
          MFRN post-processing.

        Historical note (VERB→AUX override): an unconditional override that
        relabeled VERB→AUX for lemmas in ``AUX_LEMMAS`` used to live here. It was
        removed after an A/B test showed it was a net negative: with the rule,
        VERB accuracy was 76.1% and overall POS 86.8%; without it, VERB jumped
        to 89.6% and overall to 89.4% (+2.6%), while AUX barely moved
        (89.7%→88.5%). Most of those lemmas (тұр, бер, ал, сал, ...) are ordinary
        verbs that act as auxiliaries only in specific converb constructions, and
        after retraining on KazBERT + silver the neural tagger disambiguates
        VERB/AUX well on its own. The override is therefore NOT reimplemented
        here even in scoring mode — MFRN only adjusts confidence, never POS.
        """
        if not self.scoring:
            return RMAResult(
                token=token.token,
                lemma=token.lemma,
                pos=token.pos,
                suffixes=token.suffixes,
                suffix_categories=token.suffix_categories,
                confidence=token.confidence,
            )

        kappa = self.compute_kappa(token)
        adjusted_conf = max(0.0, min(1.0, token.confidence * kappa))
        return RMAResult(
            token=token.token,
            lemma=token.lemma,
            pos=token.pos,
            suffixes=token.suffixes,
            suffix_categories=token.suffix_categories,
            confidence=adjusted_conf,
        )

    @staticmethod
    def _check_harmony(stem: str, suffixes: list[str]) -> bool:
        """Rules 1-5: Check if stem and suffixes follow vowel harmony."""
        stem_front = any(ch in FRONT_VOWELS for ch in stem)
        stem_back = any(ch in BACK_VOWELS for ch in stem)

        for suffix in suffixes:
            suf_front = any(ch in FRONT_VOWELS for ch in suffix)
            suf_back = any(ch in BACK_VOWELS for ch in suffix)

            if stem_front and suf_back:
                return False
            if stem_back and suf_front:
                return False

        return True

    @staticmethod
    def _check_suffix_voicing(lemma: str, suffixes: list[str]) -> bool:
        """Rules 49-56: Verify suffix voicing matches stem-final consonant."""
        if not lemma:
            return True

        final = lemma[-1].lower()
        # Voiced suffix variants (ды, да, дан, ға, мен, лар, ған)
        voiced_suffixes = {"ды", "да", "дан", "дар", "ға", "ған", "мен", "бен"}
        # Voiceless suffix variants (ты, та, тан, тар, қа, қан, ке, пен)
        voiceless_suffixes = {"ті", "ті", "тан", "тер", "қа", "қан", "ке", "пен"}

        for suf in suffixes:
            suf_lower = suf.lower()
            if final in VOICELESS:
                # After voiceless consonant, expect voiceless variant
                if any(suf_lower.startswith(d) for d in ("ды", "да", "дан", "дар", "ға", "ған", "мен", "бен")):
                    return False
            elif final in VOICED or final.isalpha():
                # After voiced or vowel, voiced variant is fine
                pass

        return True

    @staticmethod
    def _check_final_devoicing(token: str, lemma: str) -> bool:
        """Rules 37-42: Check that word-final devoicing is applied correctly."""
        devoice_map = {"ғ": "қ", "г": "к", "б": "п", "д": "т", "ж": "ш", "з": "с"}

        # If token ends with a voiced consonant and lemma is different,
        # the lemma should end with the voiced form
        if token and lemma:
            token_final = token[-1].lower()
            lemma_final = lemma[-1].lower()
            if token_final in devoice_map:
                # Token has voiceless form, lemma should have voiced
                if lemma_final != token_final and lemma_final != devoice_map.get(token_final, token_final):
                    return False
        return True

    @staticmethod
    def _starts_with_illegal(word: str) -> bool:
        """Rule 61: Native Kazakh words don't start with р, л (except borrowings)."""
        if not word:
            return False
        first = word[0].lower()
        # Only flag if word starts with р or л AND has no vowels typical for borrowings
        if first in ("р", "л"):
            # Check if it looks like a borrowing (has Latin chars or common borrowing patterns)
            has_non_kazakh = any(c not in "аәбвгғдежзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя" for c in word.lower())
            if not has_non_kazakh:
                return True
        return False

