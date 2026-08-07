#!/usr/bin/env python3
"""
Import lemmas from Apertium Kazakh lexc dictionary into kazmorph lexicons.

Usage:
    python3 scripts/import_apertium.py [path/to/apertium-kaz.kaz.lexc]

Extracts lemmas by POS category and writes to kazmorph/lexicons.py
"""

import re
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_lexc(filepath: str) -> dict[str, set[str]]:
    """Parse Apertium lexc file and extract lemmas by POS category."""
    nouns: set[str] = set()
    verbs: set[str] = set()
    adjectives: set[str] = set()
    proper_nouns: set[str] = set()
    adverbs: set[str] = set()

    # Patterns for lexc entries: lemma:lemma CONTINUATION_CLASS ;
    # Noun patterns: N1, N1-PL, N1-VN, N1-NAT, N1-ABBR, N2, etc.
    noun_re = re.compile(
        r'^([а-яәөүғқңһұі]+(?:[а-яәөүғқңһұі\-%\s]*[а-яәөүғқңһұі])?):\S+\s+N\d'
    )
    # Verb patterns: V-TV, V-IV, V-TV-CAUS, V-IV-REFL, etc.
    verb_re = re.compile(
        r'^([а-яәөүғқңһұі]+(?:[а-яәөүғқңһұі\-%\s]*[а-яәөүғқңһұі])?):\S+\s+V-'
    )
    # Adjective patterns: A1, A2, A3, A4, A6
    adj_re = re.compile(
        r'^([а-яәөүғқңһұі]+(?:[а-яәөүғқңһұі\-%\s]*[а-яәөүғқңһұі])?):\S+\s+A\d'
    )
    # Proper nouns
    prop_re = re.compile(
        r'^([А-ЯӘӨҮҒҚҢҺҰІ][а-яәөүғқңһұі]*):'
    )
    # Adverbs
    adv_re = re.compile(
        r'^([а-яәөүғқңһұі]+):\S+\s+ADV'
    )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("!"):
                continue

            # Try each pattern
            m = noun_re.match(line)
            if m:
                lemma = _clean_lemma(m.group(1))
                if lemma and len(lemma) >= 2:
                    nouns.add(lemma)
                continue

            m = verb_re.match(line)
            if m:
                lemma = _clean_lemma(m.group(1))
                if lemma and len(lemma) >= 2:
                    verbs.add(lemma)
                continue

            m = adj_re.match(line)
            if m:
                lemma = _clean_lemma(m.group(1))
                if lemma and len(lemma) >= 2:
                    adjectives.add(lemma)
                continue

            m = prop_re.match(line)
            if m:
                lemma = _clean_lemma(m.group(1))
                if lemma and len(lemma) >= 2:
                    proper_nouns.add(lemma.lower())
                continue

            m = adv_re.match(line)
            if m:
                lemma = _clean_lemma(m.group(1))
                if lemma and len(lemma) >= 2:
                    adverbs.add(lemma)

    return {
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "proper_nouns": proper_nouns,
        "adverbs": adverbs,
    }


def _clean_lemma(raw: str) -> str:
    """Clean a lemma from lexc format artifacts."""
    # Remove Apertium multichar symbols like %{I%}, %> etc.
    lemma = re.sub(r'%\{[^\}]*%\}', '', raw)
    lemma = re.sub(r'%.', '', lemma)
    lemma = re.sub(r'%', '', lemma)
    # Remove spaces and hyphens from multi-word entries
    lemma = lemma.strip()
    return lemma.lower()


def detect_borrowings(nouns: set[str]) -> set[str]:
    """Heuristic: detect likely Russian/international borrowings."""
    borrowing_indicators = [
        # Common Russian borrowing endings
        'ция', 'тор', 'зор', 'ник', 'чик', 'тель', 'ист', 'изм',
        'ика', 'логия', 'графия', 'метрия', 'лог', 'фон', 'метр',
        'тия', 'пия', 'рия', 'сия', 'дия', 'гия',
        # International
        'тор', 'сор', 'мент', 'ация', 'ция', 'ика',
        'про', 'ком', 'тер', 'нер', 'кер', 'пер', 'сер',
        # Additional patterns
        'атор', 'итор', 'отор', 'етор',
        'атор', 'ент', 'ант', 'ист', 'изм', 'истік',
        'ка', 'ко', 'ке', 'кі',
        'мма', 'нда', 'нта', 'нти',
        'рия', 'лия', 'мия', 'фия',
        'зав', 'вод', 'хоз', 'тех', 'пром',
        'ске', 'цке', 'ное', 'ное', 'тне',
        'град', 'бург', 'поль', 'дар',
        'кон', 'тон', 'фон', 'рон',
        'дио', 'иле', 'ото', 'ино',
        'мат', 'маш', 'стан', 'хана',
        'без', 'бас', 'инж', 'дир', 'бух',
        'пресс', 'прок', 'транс', 'суб', 'интер',
        ' супер', 'гипер', 'микро', 'макро', 'мини',
        'аудио', 'видео', 'медиа', 'радио',
        'спорт', 'культ', 'масс',
        'электр', 'энерг', 'авто',
        'био', 'гео', 'гидро', 'аэро', 'космо',
        'техника', 'система', 'программа', 'компания',
        'университет', 'институт', 'факультет',
        'министр', 'президент', 'директор', 'ректор',
        'объект', 'субъект', 'проект', 'продукт',
        'процесс', 'прогресс', 'ресс',
        'центр', 'театр', 'метр',
        'пункт', 'факт', 'акт',
        'формат', 'комплекс', 'синдикат',
        'ике', 'ка', 'га', 'го', 'ги',
        'ика', 'ата', 'ада', 'ари', 'ери',
        'ния', 'ния', 'вие', 'тие', 'ние',
        'изм', 'ист', 'ика', 'ация', 'ция',
        'ция', 'сия', 'зия', 'фия', 'пия',
        'тор', 'дор', 'зор', 'вор',
        'граф', 'скоп', 'метр', 'литр',
        'тип', 'класс', 'масс', 'баз',
        'план', 'блан', 'клан',
        'этап', 'шаг', 'модель',
    ]
    borrowed = set()
    for noun in nouns:
        for ending in borrowing_indicators:
            if noun.endswith(ending) and len(noun) > len(ending) + 1:
                borrowed.add(noun)
                break

    # Also include nouns containing non-Kazakh consonant clusters
    # (бв, гд, etc.) as likely borrowings
    non_native_clusters = ['бв', 'вг', 'гд', 'дж', 'дз', 'жб', 'жг', 'зб',
                           'зг', 'зд', 'кл', 'кр', 'лк', 'мп', 'мф', 'нк',
                           'нт', 'нф', 'пр', 'рб', 'рг', 'рд', 'рж', 'рк',
                           'рм', 'рн', 'рп', 'рс', 'рт', 'рф', 'рх', 'ск',
                           'сп', 'ст', 'стр', 'тр', 'фл', 'фр', 'хр', 'шт']
    for noun in nouns:
        for cluster in non_native_clusters:
            if cluster in noun:
                borrowed.add(noun)
                break

    # Known international words that should always be in borrowings
    known_international = {
        "компьютер", "интернет", "программа", "система", "фактор",
        "проблема", "республика", "конституция", "экономика", "технология",
        "политика", "демократия", "культура", "симфония", "опера",
        "балет", "театр", "музей", "галерея", "фестиваль",
        "университет", "институт", "факультет", "академия", "лицей",
        "компания", "корпорация", "организация", "ассоциация", "федерация",
        "банк", "бизнес", "маркетинг", "менеджмент", "офис",
        "телефон", "телевизор", "радио", "камера", "фото",
        "компьютер", "процессор", "память", "диск", "файл",
        "автомобиль", "автобус", "метро", "трамвай", "троллейбус",
        "самолет", "вертолет", "ракета", "спутник", "космос",
        "президент", "премьер", "министр", "депутат", "сенатор",
        "доктор", "профессор", "ректор", "директор", "инспектор",
        "инженер", "архитектор", "дизайнер", "программист", "оператор",
        "спорт", "футбол", "баскетбол", "волейбол", "хоккей",
        "теннис", "бокс", "шахмат", "гимнастика", "атлетика",
        "химия", "физика", "биология", "математика", "география",
        "история", "философия", "психология", "социология", "логика",
        "медицина", "хирургия", "терапия", "диагностика", "клиника",
        "рецепт", "процедура", "анализ", "симптом", "диагноз",
        "программа", "проект", "бюджет", "план", "график",
        "документ", "протокол", "регламент", "договор", "контракт",
        "электрон", "протон", "нейтрон", "атом", "молекула",
        "генератор", "мотор", "турбина", "трансформатор", "реактор",
        "центр", "сектор", "регион", "район", "квартал",
    }
    borrowed.update(known_international)

    return borrowed


def write_lexicons(lexicons_path: str, data: dict[str, set[str]], borrowings: set[str]):
    """Write lexicons.py with extracted data."""
    nouns = sorted(data["nouns"])
    verbs = sorted(data["verbs"])
    adjs = sorted(data["adjectives"])
    borrows = sorted(borrowings)

    content = f'''"""
Lexicon sets for Kazakh morphological analysis.
Auto-generated from Apertium-kaz lexc dictionary by import_apertium.py.

Source: https://github.com/apertium/apertium-kaz

NOTE: this file is regenerated by scripts/import_apertium.py. Manual
corrections live in kazmorph/lexicons_extra.py, which is merged in below
and survives regeneration. Do not edit the sets above by hand.
"""

from kazmorph.lexicons_extra import (
    EXTRA_NOUNS,
    EXTRA_VERBS,
    EXTRA_ADJ,
    NATIVE_LEMMAS,
)

# Nouns: {len(nouns)} entries (target: >= 5000)
KNOWN_NOUNS: set[str] = {{
{", ".join(f"{_quote(n)}" for n in nouns)}
}}

# Verbs: {len(verbs)} entries (target: >= 2000)
KNOWN_VERBS: set[str] = {{
{", ".join(f"{_quote(v)}" for v in verbs)}
}}

# Adjectives: {len(adjs)} entries (target: >= 1000)
KNOWN_ADJ: set[str] = {{
{", ".join(f"{_quote(a)}" for a in adjs)}
}}

# Borrowings (heuristic detection): {len(borrows)} entries (target: >= 3000)
BORROWINGS: set[str] = {{
{", ".join(f"{_quote(b)}" for b in borrows)}
}}


def is_known_noun(lemma: str) -> bool:
    return lemma in KNOWN_NOUNS or lemma in EXTRA_NOUNS


def is_known_verb(lemma: str) -> bool:
    return lemma in KNOWN_VERBS or lemma in EXTRA_VERBS


def is_known_adj(lemma: str) -> bool:
    return lemma in KNOWN_ADJ or lemma in EXTRA_ADJ


def is_borrowing(token: str) -> bool:
    t = token.lower()
    # Native whitelist overrides the heuristic BORROWINGS list, so that
    # common Turkic words accidentally matched by suffix patterns (e.g.
    # жеке via the "-ке" ending) are not mislabelled as borrowings.
    # See REVIEW_RECOMMENDATIONS.md problem #4.
    if t in NATIVE_LEMMAS:
        return False
    return t in BORROWINGS or token in BORROWINGS
'''

    with open(lexicons_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Written to {lexicons_path}")
    print(f"  NOUNS:      {len(nouns)} (target >= 5000)")
    print(f"  VERBS:      {len(verbs)} (target >= 2000)")
    print(f"  ADJ:        {len(adjs)} (target >= 1000)")
    print(f"  BORROWINGS: {len(borrows)} (target >= 3000)")


def _quote(s: str) -> str:
    """Quote a string for Python set literal."""
    return f'"{s}"'


def main():
    # Default path
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "apertium-kaz",
        "apertium-kaz.kaz.lexc",
    )
    lexc_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(lexc_path):
        print(f"Error: {lexc_path} not found")
        print("Usage: python3 scripts/import_apertium.py [path/to/apertium-kaz.kaz.lexc]")
        sys.exit(1)

    print(f"Parsing {lexc_path}...")
    data = parse_lexc(lexc_path)

    print(f"Raw counts:")
    for k, v in data.items():
        print(f"  {k}: {len(v)}")

    # Detect borrowings from nouns
    borrowings = detect_borrowings(data["nouns"])
    print(f"\nBorrowings detected: {len(borrowings)}")

    # Write lexicons
    lexicons_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kazmorph",
        "lexicons.py",
    )
    write_lexicons(lexicons_path, data, borrowings)


if __name__ == "__main__":
    main()
