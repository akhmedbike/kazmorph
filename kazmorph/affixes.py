"""
Kazakh affix definitions for the RMA (Reverse Morphological Analyzer).
~120 affixes covering major morphological categories.

Kazakh suffix order (canonical): Stem + PL + POSS + CASE
  Example: мектеп + тер(PL) + іміз(POSS.1PL) + ден(ABL)
"""

# Each affix has: surface form, grammatical category, and POS constraint
AFFIXES: list[dict] = [
    # ── Plural suffixes (6) ──────────────────────────────────────
    {"surface": "дар", "category": "PL", "pos": "NOUN"},
    {"surface": "дер", "category": "PL", "pos": "NOUN"},
    {"surface": "тар", "category": "PL", "pos": "NOUN"},
    {"surface": "тер", "category": "PL", "pos": "NOUN"},
    {"surface": "лар", "category": "PL", "pos": "NOUN"},
    {"surface": "лер", "category": "PL", "pos": "NOUN"},

    # ── Possessive suffixes — 1st person (4) ─────────────────────
    {"surface": "ым", "category": "POSS.1SG", "pos": "NOUN"},
    {"surface": "ім", "category": "POSS.1SG", "pos": "NOUN"},
    {"surface": "ымыз", "category": "POSS.1PL", "pos": "NOUN"},
    {"surface": "іміз", "category": "POSS.1PL", "pos": "NOUN"},

    # ── Possessive suffixes — 2nd person (8) ─────────────────────
    {"surface": "ң", "category": "POSS.2SG.INF", "pos": "NOUN"},
    {"surface": "ың", "category": "POSS.2SG", "pos": "NOUN"},
    {"surface": "ің", "category": "POSS.2SG", "pos": "NOUN"},
    {"surface": "ыңыз", "category": "POSS.2PL", "pos": "NOUN"},
    {"surface": "іңіз", "category": "POSS.2PL", "pos": "NOUN"},
    {"surface": "ңыз", "category": "POSS.2PL", "pos": "NOUN"},
    {"surface": "ңіз", "category": "POSS.2PL", "pos": "NOUN"},
    {"surface": "ыңдар", "category": "POSS.2SG.PL", "pos": "NOUN"},
    {"surface": "іңдер", "category": "POSS.2SG.PL", "pos": "NOUN"},

    # ── Possessive suffixes — 3rd person (4) ─────────────────────
    {"surface": "сы", "category": "POSS.3", "pos": "NOUN"},
    {"surface": "сі", "category": "POSS.3", "pos": "NOUN"},
    {"surface": "ы", "category": "POSS.3", "pos": "NOUN"},
    {"surface": "і", "category": "POSS.3", "pos": "NOUN"},

    # ── Case suffixes — Genitive (6) ─────────────────────────────
    {"surface": "дың", "category": "GEN", "pos": "NOUN"},
    {"surface": "дің", "category": "GEN", "pos": "NOUN"},
    {"surface": "тың", "category": "GEN", "pos": "NOUN"},
    {"surface": "тің", "category": "GEN", "pos": "NOUN"},
    {"surface": "ның", "category": "GEN", "pos": "NOUN"},
    {"surface": "нің", "category": "GEN", "pos": "NOUN"},

    # ── Case suffixes — Dative (6) ───────────────────────────────
    {"surface": "ға", "category": "DAT", "pos": "NOUN"},
    {"surface": "ге", "category": "DAT", "pos": "NOUN"},
    {"surface": "қа", "category": "DAT", "pos": "NOUN"},
    {"surface": "ке", "category": "DAT", "pos": "NOUN"},
    {"surface": "на", "category": "DAT", "pos": "NOUN"},
    {"surface": "не", "category": "DAT", "pos": "NOUN"},

    # ── Case suffixes — Accusative (6) ───────────────────────────
    {"surface": "ды", "category": "ACC", "pos": "NOUN"},
    {"surface": "ді", "category": "ACC", "pos": "NOUN"},
    {"surface": "ты", "category": "ACC", "pos": "NOUN"},
    {"surface": "ті", "category": "ACC", "pos": "NOUN"},
    {"surface": "ны", "category": "ACC", "pos": "NOUN"},
    {"surface": "ні", "category": "ACC", "pos": "NOUN"},

    # ── Case suffixes — Ablative (6) ─────────────────────────────
    {"surface": "дан", "category": "ABL", "pos": "NOUN"},
    {"surface": "ден", "category": "ABL", "pos": "NOUN"},
    {"surface": "тан", "category": "ABL", "pos": "NOUN"},
    {"surface": "тен", "category": "ABL", "pos": "NOUN"},
    {"surface": "нан", "category": "ABL", "pos": "NOUN"},
    {"surface": "нен", "category": "ABL", "pos": "NOUN"},

    # ── Case suffixes — Locative (8) ─────────────────────────────
    {"surface": "да", "category": "LOC", "pos": "NOUN"},
    {"surface": "де", "category": "LOC", "pos": "NOUN"},
    {"surface": "та", "category": "LOC", "pos": "NOUN"},
    {"surface": "те", "category": "LOC", "pos": "NOUN"},
    {"surface": "нда", "category": "LOC", "pos": "NOUN"},
    {"surface": "нде", "category": "LOC", "pos": "NOUN"},
    {"surface": "нша", "category": "LOC", "pos": "NOUN"},
    {"surface": "нше", "category": "LOC", "pos": "NOUN"},

    # ── Case suffixes — Instrumental (4) ─────────────────────────
    {"surface": "мен", "category": "INST", "pos": "NOUN"},
    {"surface": "бен", "category": "INST", "pos": "NOUN"},
    {"surface": "пен", "category": "INST", "pos": "NOUN"},
    {"surface": "ндей", "category": "INST", "pos": "NOUN"},

    # ── Adjective-forming suffixes (8) ───────────────────────────
    {"surface": "лы", "category": "ADJ", "pos": "NOUN"},
    {"surface": "лі", "category": "ADJ", "pos": "NOUN"},
    {"surface": "ды", "category": "ADJ", "pos": "NOUN"},
    {"surface": "ді", "category": "ADJ", "pos": "NOUN"},
    {"surface": "ты", "category": "ADJ", "pos": "NOUN"},
    {"surface": "ті", "category": "ADJ", "pos": "NOUN"},
    {"surface": "ғы", "category": "ADJ", "pos": "NOUN"},
    {"surface": "гі", "category": "ADJ", "pos": "NOUN"},
    {"surface": "қы", "category": "ADJ", "pos": "NOUN"},
    {"surface": "кі", "category": "ADJ", "pos": "NOUN"},
    {"surface": "лық", "category": "ADJ.ABST", "pos": "NOUN"},
    {"surface": "лік", "category": "ADJ.ABST", "pos": "NOUN"},
    {"surface": "дық", "category": "ADJ.ABST", "pos": "NOUN"},
    {"surface": "дік", "category": "ADJ.ABST", "pos": "NOUN"},
    {"surface": "тық", "category": "ADJ.ABST", "pos": "NOUN"},
    {"surface": "тік", "category": "ADJ.ABST", "pos": "NOUN"},

    # ── Verb tense/person suffixes (4) ───────────────────────────
    {"surface": "ды", "category": "PAST.3", "pos": "VERB"},
    {"surface": "ді", "category": "PAST.3", "pos": "VERB"},
    {"surface": "ты", "category": "PAST.3", "pos": "VERB"},
    {"surface": "ті", "category": "PAST.3", "pos": "VERB"},

    # ── Participle suffixes (4) ──────────────────────────────────
    {"surface": "ған", "category": "PST.PTCP", "pos": "VERB"},
    {"surface": "ген", "category": "PST.PTCP", "pos": "VERB"},
    {"surface": "қан", "category": "PST.PTCP", "pos": "VERB"},
    {"surface": "кен", "category": "PST.PTCP", "pos": "VERB"},

    # ── Present participle (4) ───────────────────────────────────
    {"surface": "атын", "category": "PRS.PTCP", "pos": "VERB"},
    {"surface": "етін", "category": "PRS.PTCP", "pos": "VERB"},
    {"surface": "йтын", "category": "PRS.PTCP", "pos": "VERB"},
    {"surface": "йтін", "category": "PRS.PTCP", "pos": "VERB"},

    # ── Future participle (4) ────────────────────────────────────
    {"surface": "ар", "category": "FUT.PTCP", "pos": "VERB"},
    {"surface": "ер", "category": "FUT.PTCP", "pos": "VERB"},
    {"surface": "р", "category": "FUT.PTCP", "pos": "VERB"},

    # ── Converb / gerund (7) ─────────────────────────────────────
    # Perfective converb -п (after vowel, e.g. таста+п → тастап) and its
    # consonant-protecting variants -ып/-іп/-п. Imperfective converb -а/-е/-й.
    # The bare -п was previously missing, which caused under-stripping
    # (т астап → lemma "тастап" instead of "таста") — a top lemmatization error.
    {"surface": "ып", "category": "CVB", "pos": "VERB"},
    {"surface": "іп", "category": "CVB", "pos": "VERB"},
    {"surface": "п", "category": "CVB", "pos": "VERB"},
    {"surface": "й", "category": "CVB.IPFV", "pos": "VERB"},
    {"surface": "а", "category": "CVB.IPFV", "pos": "VERB"},
    {"surface": "е", "category": "CVB.IPFV", "pos": "VERB"},
    {"surface": "ғалы", "category": "CVB.PURP", "pos": "VERB"},
    {"surface": "гелі", "category": "CVB.PURP", "pos": "VERB"},
    {"surface": "қалы", "category": "CVB.PURP", "pos": "VERB"},
    {"surface": "келі", "category": "CVB.PURP", "pos": "VERB"},

    # ── Verbal noun / agent noun (4) ─────────────────────────────
    {"surface": "у", "category": "VN", "pos": "VERB"},
    {"surface": "ғыш", "category": "AGT.N", "pos": "VERB"},
    {"surface": "гіш", "category": "AGT.N", "pos": "VERB"},
    {"surface": "шы", "category": "AGT.N", "pos": "NOUN"},
    {"surface": "ші", "category": "AGT.N", "pos": "NOUN"},

    # ── Adverbial suffixes (4) ───────────────────────────────────
    {"surface": "ша", "category": "ADV", "pos": "ADJ"},
    {"surface": "ше", "category": "ADV", "pos": "ADJ"},
    {"surface": "дай", "category": "EQU", "pos": "NOUN"},
    {"surface": "дей", "category": "EQU", "pos": "NOUN"},
    {"surface": "тай", "category": "EQU", "pos": "NOUN"},
    {"surface": "тей", "category": "EQU", "pos": "NOUN"},

    # ── Diminutive (2) ───────────────────────────────────────────
    {"surface": "ша", "category": "DIM", "pos": "NOUN"},
    {"surface": "ше", "category": "DIM", "pos": "NOUN"},

    # ── Voice / derivation suffixes (VERB) ─────────────────────
    # Causative (каузатив)
    {"surface": "тұр", "category": "CAUS", "pos": "VERB"},
    {"surface": "тір", "category": "CAUS", "pos": "VERB"},
    {"surface": "тыр", "category": "CAUS", "pos": "VERB"},
    {"surface": "тір", "category": "CAUS", "pos": "VERB"},
    {"surface": "дыр", "category": "CAUS", "pos": "VERB"},
    {"surface": "дір", "category": "CAUS", "pos": "VERB"},
    {"surface": "қыз", "category": "CAUS", "pos": "VERB"},
    {"surface": "кіз", "category": "CAUS", "pos": "VERB"},
    {"surface": "ғыз", "category": "CAUS", "pos": "VERB"},
    {"surface": "гіз", "category": "CAUS", "pos": "VERB"},
    {"surface": "тандыр", "category": "CAUS", "pos": "VERB"},
    {"surface": "тендір", "category": "CAUS", "pos": "VERB"},

    # Passive (пассив)
    {"surface": "ыл", "category": "PASS", "pos": "VERB"},
    {"surface": "іл", "category": "PASS", "pos": "VERB"},
    {"surface": "л", "category": "PASS", "pos": "VERB"},

    # Reflexive (рефлексив)
    {"surface": "ын", "category": "REFL", "pos": "VERB"},
    {"surface": "ін", "category": "REFL", "pos": "VERB"},
    {"surface": "н", "category": "REFL", "pos": "VERB"},

    # Reciprocal (взаимный)
    {"surface": "ыс", "category": "RECIP", "pos": "VERB"},
    {"surface": "іс", "category": "RECIP", "pos": "VERB"},
    {"surface": "с", "category": "RECIP", "pos": "VERB"},

    # Negative (болыссыз)
    {"surface": "ма", "category": "NEG", "pos": "VERB"},
    {"surface": "ме", "category": "NEG", "pos": "VERB"},
    {"surface": "ба", "category": "NEG", "pos": "VERB"},
    {"surface": "бе", "category": "NEG", "pos": "VERB"},
    {"surface": "па", "category": "NEG", "pos": "VERB"},
    {"surface": "пе", "category": "NEG", "pos": "VERB"},

    # Potential (мүмкіндік)
    {"surface": "а", "category": "POT", "pos": "VERB"},
    {"surface": "е", "category": "POT", "pos": "VERB"},
    {"surface": "й", "category": "POT", "pos": "VERB"},

    # Volitional / imperative (қалау рай)
    {"surface": "ғы", "category": "VOL", "pos": "VERB"},
    {"surface": "гі", "category": "VOL", "pos": "VERB"},
    {"surface": "қы", "category": "VOL", "pos": "VERB"},
    {"surface": "кі", "category": "VOL", "pos": "VERB"},

    # ── Verbalizer (noun → verb) suffixes ──────────────────────
    # Note: дан/ден/тан/тен omitted — conflict with ABL case suffixes
    {"surface": "лан", "category": "VBLZ", "pos": "VERB"},
    {"surface": "лен", "category": "VBLZ", "pos": "VERB"},
    {"surface": "ла", "category": "VBLZ", "pos": "VERB"},
    {"surface": "ле", "category": "VBLZ", "pos": "VERB"},
]
