"""Keyboard-neighbour and mixed-script maps for the corruption operators.

The controlled-error generator (:mod:`kazmorph.correction.corrupt`) needs two
phonetically-grounded sources of plausible typos:

  1. Keyboard neighbours — which keys sit next to a given key on the standard
     Kazakh Cyrillic layout. Real insertion/substitution errors are dominated
     by adjacent-key slips, not uniformly random characters.

  2. A Kazakh↔Russian visually-similar grapheme map. Mixed-script substitution
     replaces a Kazakh-specific grapheme with a visually similar Russian
     Cyrillic one (or vice versa). This is a frequent real-world error in
     Kazakhstan where both layouts and both orthographies are in active use.

The Kazakh alphabet has 9 graphemes absent from Russian (ә ғ қ ң ө ұ ү һ і),
and Russian has 9 absent from Kazakh (ф в ц щ э ъ ь и й — the last two are
shared letters but behave differently; here we only treat the unambiguous
non-shared set). The `_BORROWING_PHONEMES` set in :mod:`kazmorph.taxonomy`
confirms the Russian-only side: frozenset("фвцщэъь").

Adapted verbatim from the IHCONCS_2026_paper project (MIT).
"""

from __future__ import annotations

# ── Kazakh keyboard layout (standard macOS / Windows Cyrillic-Kazakh) ────────
# Rows laid out left-to-right. The neighbour map is derived from physical
# adjacency on a QWERTY-shaped keyboard, so "ф"'s neighbours are the keys
# immediately to its left, right, above-left, above, above-right, etc.
#
# This is the Kazakh extended layout: Russian base + 9 Kazakh letters on the
# digit row and shifted positions.
_KZ_ROWS = [
    "фывапролджэ",     # middle row (home row), left→right
    "йцукенгшщзхъ",    # top row, left→right
    "ячсмитьбю",       # bottom row, left→right
    "әғқңөұүһі",       # Kazakh-specific (digit-row positions on the real layout)
]


def _build_neighbours(rows: list[str]) -> dict[str, list[str]]:
    """Build a {char: [neighbour chars]} map from keyboard rows.

    A neighbour is any key at Chebyshev distance 1 (the 8 surrounding cells)
    on the row grid. We deliberately use the *physical* layout (rows of
    different lengths, staggered), not a rectilinear projection — that's what
    makes 'а' adjacent to 'п' and 'р' but not to 'ф'.
    """
    coords: dict[str, tuple[int, int]] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            # Last-write-wins is fine: keys are unique within a row, and a
            # letter appearing in two rows (shouldn't happen here) would just
            # pick the last position.
            coords[ch] = (r, c)
    neighbours: dict[str, list[str]] = {}
    for ch, (r, c) in coords.items():
        ns: list[str] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                # Find the key whose column is nearest to c+dc on row r+dr.
                # Rows have different lengths and a half-key stagger, so we
                # match by proximity: pick the column on the target row that
                # minimises |target_col - (c + dc)|.
                target_row = r + dr
                if target_row < 0 or target_row >= len(rows):
                    continue
                row_chars = rows[target_row]
                if not row_chars:
                    continue
                target_col = c + dc
                # Clamp to the row's extent, then pick the nearest actual key.
                best_j = min(
                    range(len(row_chars)),
                    key=lambda j: abs(j - target_col),
                )
                neighbour = row_chars[best_j]
                if neighbour != ch and neighbour not in ns:
                    ns.append(neighbour)
        neighbours[ch] = ns
    return neighbours


# Physical-key neighbour map. Lowercase keys; callers must `.lower()` input
# before lookup and decide their own capitalisation policy.
KZ_KEYBOARD_NEIGHBORS: dict[str, list[str]] = _build_neighbours(_KZ_ROWS)


def keyboard_neighbours(ch: str) -> list[str]:
    """Return keyboard-adjacent chars for `ch`, lowercased.

    Returns an empty list for chars not on the Kazakh layout (digits,
    punctuation, Latin). The corruption operators fall back to a random
    Kazakh vowel when this is empty, so an empty result is always safe.
    """
    return KZ_KEYBOARD_NEIGHBORS.get(ch.lower(), [])


# ── Kazakh vowels (for the random-vowel fallback in insert/substitute) ───────
# All nine vowels of the Kazakh alphabet. Used when a keyboard neighbour is
# unavailable (e.g. the char is a digit) or when we want a phonotactically
# legal insertion. Kept lowercase.
KAZAKH_VOWELS = frozenset("аәеіүұөоыу")


# ── Mixed-script (Kazakh ↔ Russian) visually-similar grapheme pairs ──────────
# Each entry maps a Kazakh-specific grapheme to the Russian Cyrillic grapheme
# it is most often confused with visually. These pairs drive the "mixed-script
# substitution" corruption operator. The mapping is deliberately asymmetric in
# intent: we generate Kazakh→Russian corruptions (a Kazakh letter is replaced
# by its Russian visual twin), because that is the documented failure mode in
# mixed-script Kazakh text. The reverse direction is generated by inverting
# this dict at call time.
#
# Sources of the visual similarity:
#   ә ↔ э : both are "e with a diacritic"; ә is Kazakh front /æ/, э is Russian /e/.
#   ғ ↔ г : both are "г with a stroke/hook"; ғ is Kazakh voiced uvular /ɣ/.
#   қ ↔ к : both are "к with a descender"; қ is Kazakh voiceless uvular /q/.
#   ң ↔ н : both are "н with a descender"; ң is Kazakh velar nasal /ŋ/.
#   ө ↔ о : both are "o with a stroke"; ө is Kazakh front /ø/.
#   ұ ↔ ү : both are "u with a stroke/horn"; ұ is back /ʊ/, ү is front /ʉ/.
#          We map ұ→у (closest Russian u) as the more common confusion.
#   ү ↔ у : see above; the front/back pair collapses to Russian у.
#   һ ↔ х : һ is a "h with a descender"; closest Russian is х (/x/).
#   і ↔ и : і is a dotted i (/i/); Russian и is /i/. (Russian also has й.)
KAZAKH_TO_RUSSIAN: dict[str, str] = {
    "ә": "э",
    "ғ": "г",
    "қ": "к",
    "ң": "н",
    "ө": "о",
    "ұ": "у",
    "ү": "у",
    "һ": "х",
    "і": "и",
}

# Russian-only graphemes (from kazmorph.taxonomy._BORROWING_PHONEMES) that have
# no native Kazakh counterpart. These are the natural "Russian injection"
# targets when we want a corruption that introduces a non-Kazakh phoneme.
RUSSIAN_ONLY = frozenset("фвцщэъь")


def mixed_script_substitute(ch: str) -> str | None:
    """Return the visually-similar Russian grapheme for a Kazakh `ch`, or None.

    Case-preserving: if `ch` is uppercase, the returned Russian letter is also
    uppercase. Returns None when `ch` has no defined visual twin, signalling
    the caller to skip this corruption site.
    """
    lower = ch.lower()
    rus = KAZAKH_TO_RUSSIAN.get(lower)
    if rus is None:
        return None
    return rus.upper() if ch.isupper() else rus
