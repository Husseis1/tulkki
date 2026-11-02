"""Finnish numeral parsing and generation.

Finnish numerals are hard for voice assistants because they

* inflect in every part of a compound: ``kahdeltatoista`` (at twelve),
  ``kahdenkymmenenviiden`` (of twenty-five),
* have colloquial (puhekieli) forms: ``kakskytviis``, ``kasilta``, ``yheltätoist``,
* get split by speech recognisers: ``kaks kymmentä viis``.

This module segments a token into numeral morphemes (stem + optional case
ending) with a small backtracking search, then evaluates the morpheme sequence
with a place-value grammar. Spans of several tokens are merged as long as the
combined sequence stays grammatical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

__all__ = [
    "ABLATIVE_SUFFIXES",
    "Kind",
    "NumberSpan",
    "Segment",
    "find_numbers",
    "parse_number",
    "segment_token",
    "to_words",
]


class Kind(Enum):
    UNIT = "unit"
    TEN = "ten"
    TEEN = "teen"
    HUNDRED = "hundred"
    THOUSAND = "thousand"


@dataclass(frozen=True)
class Segment:
    stem: str
    suffix: str
    value: int
    kind: Kind
    bound: bool  # cannot start a number on its own (kymmentä, sataa, toista)


@dataclass(frozen=True)
class NumberSpan:
    start: int
    end: int  # exclusive token index
    value: int


_STEMS: dict[str, tuple[int, Kind]] = {}


def _add(value: int, kind: Kind, *stems: str) -> None:
    for stem in stems:
        _STEMS[stem] = (value, kind)


# Standard stems, inflection stems (yhde-, kahte-) and colloquial stems (yhe-, kahe-).
_add(1, Kind.UNIT, "yksi", "yks", "yhde", "yhte", "yht", "yhe")
_add(2, Kind.UNIT, "kaksi", "kaks", "kahde", "kahte", "kaht", "kahe")
_add(3, Kind.UNIT, "kolme", "kol")
_add(4, Kind.UNIT, "neljä", "nel")
_add(5, Kind.UNIT, "viisi", "viis", "viide", "viite", "viit", "viie")
_add(6, Kind.UNIT, "kuusi", "kuus", "kuude", "kuute", "kuut", "kuue")
_add(7, Kind.UNIT, "seitsemän", "seittemän", "seitsemä", "seittemä", "seiska", "seis")
_add(8, Kind.UNIT, "kahdeksan", "kaheksan", "kahdeksa", "kaheksa", "kasi")
_add(9, Kind.UNIT, "yhdeksän", "yheksän", "yhdeksä", "yheksä", "ysi")
_add(10, Kind.TEN, "kymmenen", "kymmene", "kymmen", "kymment", "kymppi", "kympi", "kyt")
_add(10, Kind.TEEN, "toista", "toist")
_add(100, Kind.HUNDRED, "sata", "sada", "sato")
_add(1000, Kind.THOUSAND, "tuhat", "tuhanne")

_STEMS_SORTED = sorted(_STEMS, key=len, reverse=True)
_BOUND_STEMS = frozenset({"kyt", "kymment"})
# Colloquial prefixes that only exist inside compounds (kolkyt, nelkyt, seiskyt).
_PREFIX_ONLY = frozenset({"kol", "nel", "seis"})
_PARTITIVE = frozenset({"a", "ä", "ta", "tä"})
_SUFFIXES = sorted(
    {
        "", "n", "a", "ä", "ta", "tä", "na", "nä", "ksi", "lla", "llä", "lta", "ltä",
        "lle", "ssa", "ssä", "sta", "stä", "en", "een", "an", "ään", "lt", "ll", "st", "ks",
    },
    key=len,
    reverse=True,
)
ABLATIVE_SUFFIXES = frozenset({"lta", "ltä", "lt"})
# Ordinary words that happen to segment as numerals.
_BLACKLIST = frozenset({"yksin", "yhdessä", "yhteen", "yhtä", "yhteensä", "kuuta", "viite"})


@lru_cache(maxsize=8192)
def segment_token(token: str) -> tuple[Segment, ...] | None:
    """Split one token into numeral morphemes, or return None if it is not a numeral."""
    token = token.lower()
    if token in _BLACKLIST or not token.isalpha():
        return None
    n = len(token)
    memo: dict[int, tuple[Segment, ...] | None] = {}

    def go(pos: int) -> tuple[Segment, ...] | None:
        if pos == n:
            return ()
        if pos in memo:
            return memo[pos]
        result = None
        for stem in _STEMS_SORTED:
            if not token.startswith(stem, pos):
                continue
            end = pos + len(stem)
            for suffix in _SUFFIXES:
                if not token.startswith(suffix, end):
                    continue
                rest = go(end + len(suffix))
                if rest is None:
                    continue
                # "kol", "nel" only exist before -kyt; allow them token-finally
                # because recognisers split "nelkyt" into "nel kyt".
                if stem in _PREFIX_ONLY and (suffix or (rest and rest[0].kind not in (
                        Kind.TEN, Kind.TEEN, Kind.HUNDRED))):
                    continue
                value, kind = _STEMS[stem]
                bound = stem in _BOUND_STEMS or (
                    kind in (Kind.TEN, Kind.HUNDRED, Kind.THOUSAND) and suffix in _PARTITIVE
                )
                result = (Segment(stem, suffix, value, kind, bound),) + rest
                break
            if result is not None:
                break
        memo[pos] = result
        return result

    return go(0)


def _evaluate(segs: Sequence[Segment], allow_bound_first: bool = False) -> int | None:
    """Evaluate a morpheme sequence; None if it is not a well-formed numeral."""
    total = cur = pending = 0
    last: str | None = None
    for idx, seg in enumerate(segs):
        bound = seg.bound and not (allow_bound_first and idx == 0)
        if seg.kind is Kind.UNIT:
            if last in ("unit", "teen", "ten10"):
                return None
            cur += seg.value
            pending = seg.value
            last = "unit"
        elif seg.kind is Kind.TEEN:
            if last != "unit" or cur % 100 >= 10:
                return None
            cur += 10
            last = "teen"
        elif seg.kind is Kind.TEN:
            if last == "unit" and cur % 100 < 10:
                cur += pending * 9  # unit already counted once
                last = "tens"
            elif bound or last not in (None, "hundred", "thousand"):
                return None
            else:
                cur += 10
                last = "ten10"
        elif seg.kind is Kind.HUNDRED:
            if last == "unit" and cur == pending:
                cur = pending * 100
            elif bound or last not in (None, "thousand") or cur:
                return None
            else:
                cur = 100
            last = "hundred"
        else:  # THOUSAND
            if last == "thousand" or (total and cur):
                return None
            if cur:
                total += cur * 1000
                cur = 0
            elif bound or total:
                return None
            else:
                total = 1000
            last = "thousand"
    if last is None:
        return None
    return total + cur


def find_numbers(tokens: Sequence[str], allow_bound: bool = False) -> list[NumberSpan]:
    """Find numeral spans in a token list, merging ASR-split compounds greedily.

    ``allow_bound`` lets a span start with a bound form such as ``kymmentä``
    (needed for clock expressions like *kymmentä yli viisi*).
    """
    spans: list[NumberSpan] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i].lower()
        if tok.isdigit():
            spans.append(NumberSpan(i, i + 1, int(tok)))
            i += 1
            continue
        segs: list[Segment] = []
        best: NumberSpan | None = None
        j = i
        while j < n:
            seg = segment_token(tokens[j].lower())
            if seg is None:
                break
            candidate = segs + list(seg)
            value = _evaluate(candidate, allow_bound_first=allow_bound)
            if value is None:
                break
            segs = candidate
            best = NumberSpan(i, j + 1, value)
            j += 1
        if best is not None:
            spans.append(best)
            i = best.end
        else:
            i += 1
    return spans


def parse_number(text: str) -> int | None:
    """Parse a string that is exactly one numeral (possibly several tokens)."""
    tokens = text.split()
    spans = find_numbers(tokens)
    if len(spans) == 1 and spans[0].start == 0 and spans[0].end == len(tokens):
        return spans[0].value
    return None


# --- generation -------------------------------------------------------------

_UNITS = {
    "nom": ["nolla", "yksi", "kaksi", "kolme", "neljä", "viisi", "kuusi", "seitsemän", "kahdeksan", "yhdeksän"],
    "gen": ["nollan", "yhden", "kahden", "kolmen", "neljän", "viiden", "kuuden", "seitsemän", "kahdeksan", "yhdeksän"],
    "abl": ["nollalta", "yhdeltä", "kahdelta", "kolmelta", "neljältä", "viideltä", "kuudelta", "seitsemältä", "kahdeksalta", "yhdeksältä"],
    "part": ["nollaa", "yhtä", "kahta", "kolmea", "neljää", "viittä", "kuutta", "seitsemää", "kahdeksaa", "yhdeksää"],
    "tra": ["nollaksi", "yhdeksi", "kahdeksi", "kolmeksi", "neljäksi", "viideksi", "kuudeksi", "seitsemäksi", "kahdeksaksi", "yhdeksäksi"],
    "all": ["nollalle", "yhdelle", "kahdelle", "kolmelle", "neljälle", "viidelle", "kuudelle", "seitsemälle", "kahdeksalle", "yhdeksälle"],
}
_UNITS_COL = {
    "nom": ["nolla", "yks", "kaks", "kolme", "neljä", "viis", "kuus", "seittemän", "kaheksan", "yheksän"],
    "gen": ["nollan", "yhen", "kahen", "kolmen", "neljän", "viiden", "kuuden", "seittemän", "kaheksan", "yheksän"],
    "abl": ["nollalt", "yhelt", "kahelt", "kolmelt", "neljält", "viidelt", "kuudelt", "seittemält", "kaheksalt", "yheksält"],
}
_TEN_ALONE = {"nom": "kymmenen", "gen": "kymmenen", "abl": "kymmeneltä", "part": "kymmentä", "tra": "kymmeneksi", "all": "kymmenelle"}
_TEN_MULT = {"nom": "kymmentä", "gen": "kymmenen", "abl": "kymmeneltä", "part": "kymmentä", "tra": "kymmeneksi", "all": "kymmenelle"}
_COL_TENS = {2: "kakskyt", 3: "kolkyt", 4: "nelkyt", 5: "viiskyt", 6: "kuuskyt", 7: "seiskyt", 8: "kasikyt", 9: "ysikyt"}


def to_words(n: int, form: str = "nom", colloquial: bool = False) -> str:
    """Render 1–99 as Finnish words in a given case.

    ``form`` is one of nom, gen, abl, part, tra, all. Colloquial variants exist
    for nom, gen and abl; other cases fall back to standard Finnish.
    """
    if not 1 <= n <= 99:
        raise ValueError("to_words supports 1..99")
    if form not in _UNITS:
        raise ValueError(f"unknown form {form!r}")
    col = colloquial and form in _UNITS_COL
    units = _UNITS_COL[form] if col else _UNITS[form]
    tens, unit = divmod(n, 10)
    if n < 10:
        return units[n]
    if n == 10:
        if col and form == "abl":
            return "kymmenelt"
        return _TEN_ALONE[form]
    if n < 20:
        if col and form == "abl":
            vowel = "a" if any(c in "aou" for c in units[unit]) else "ä"
            return units[unit] + vowel + "toist"  # kaheltatoist, yheltätoist
        return units[unit] + ("toist" if col else "toista")
    if col and form == "nom":
        return _COL_TENS[tens] + (units[unit] if unit else "")
    if col and form == "abl" and unit:
        return _COL_TENS[tens] + units[unit]  # kakskytviidelt
    if col and form == "abl":
        return _UNITS["abl"][tens] + "kymmeneltä"
    return units[tens] + _TEN_MULT[form] + (units[unit] if unit else "")
