"""Inflection and lemmatisation of Finnish proper names.

A voice assistant hears ``soita Mikolle`` and must call *Mikko*; it hears
``millainen sää on Lahdessa`` and must look up *Lahti*. Proper names are an
open class (every user's contact list is different), so a fixed lookup table of
inflected forms is not enough: we need to *generate* the forms of any new name.

The approach is generate-and-index. Each name has an inflection class
(roughly following the Kotus classes used in Finnish dictionaries) and an
optional consonant gradation flag. :func:`inflect` produces the case forms
(standard and colloquial); :class:`Lexicon` indexes all of them so that
lemmatisation is a dictionary lookup at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Entry", "InflClass", "Lexicon", "gradate", "guess_class", "harmony", "inflect"]

_VOWELS = "aeiouyäö"


class InflClass(Enum):
    VOWEL = "vowel"    # Liisa, Mikko, Turku, Helsinki (new i-stem)
    OLD_I = "old_i"    # Lahti -> Lahden, Rovaniemi -> Rovaniemellä
    E = "e"            # Tampere -> Tampereella
    NEN = "nen"        # Virtanen -> Virtaselle
    LONG = "long"      # Espoo, Vantaa, Joensuu (long vowel)
    KSE = "kse"        # Itäkeskus -> Itäkeskuksessa, Tuomas -> Tuomakselle
    CONS = "cons"      # consonant-final loan names: Daniel -> Danielille


# Strong -> weak grade, checked longest first at the end of the stem body.
_GRADATION = [
    ("kk", "k"), ("pp", "p"), ("tt", "t"),
    ("nk", "ng"), ("mp", "mm"), ("lt", "ll"), ("nt", "nn"), ("rt", "rr"),
    ("ht", "hd"), ("rk", "r"), ("lk", "l"),
    ("t", "d"), ("p", "v"), ("k", ""),
]


def harmony(word: str) -> str:
    """Return 'a' for back-vowel words and 'ä' for front-vowel words.

    The last back/front vowel decides, which handles compounds such as
    *Jyväskylä* (front) and *Lappeenranta* (back).
    """
    for ch in reversed(word.lower()):
        if ch in "aou":
            return "a"
        if ch in "äöy":
            return "ä"
    return "ä"


def gradate(stem: str) -> str:
    """Apply strong-to-weak consonant gradation to the last syllable of ``stem``.

    ``stem`` must end in a vowel: ``Mikko`` -> ``Miko``, ``Helsinki`` ->
    ``Helsingi``, ``Lahte`` -> ``Lahde``, ``Turku`` -> ``Turu``.
    """
    if not stem or stem[-1].lower() not in _VOWELS:
        return stem
    body, final = stem[:-1], stem[-1]
    for strong, weak in _GRADATION:
        if body.lower().endswith(strong):
            pre = body[: -len(strong)]
            # Single-consonant patterns need a vowel (or h for k) before them.
            if len(strong) == 1 and (not pre or pre[-1].lower() not in _VOWELS):
                continue
            return pre + weak + final
    return stem


def guess_class(name: str) -> InflClass:
    """Heuristic inflection class for a name not in the lexicon."""
    low = name.lower()
    if low.endswith("nen"):
        return InflClass.NEN
    if low.endswith(("as", "es", "is", "os", "us", "ys", "äs")):
        return InflClass.KSE
    if len(low) >= 2 and low[-1] == low[-2] and low[-1] in _VOWELS:
        return InflClass.LONG
    if low[-1] not in _VOWELS:
        return InflClass.CONS
    if low.endswith("e"):
        return InflClass.E
    return InflClass.VOWEL


def _stems(name: str, cls: InflClass, gradation: bool, a: str) -> tuple[str, str, str]:
    """Return (strong vowel stem, weak vowel stem, partitive form)."""
    if cls is InflClass.VOWEL:
        strong = name
        weak = gradate(name) if gradation else name
        # Two final vowels take -ta: Kuopiota, Kalliota; otherwise -a: Turkua.
        two_vowels = len(name) > 1 and name[-2].lower() in _VOWELS
        return strong, weak, strong + ("t" + a if two_vowels else a)
    if cls is InflClass.OLD_I:
        strong = name[:-1] + "e"
        weak = gradate(strong) if gradation else strong
        return strong, weak, strong + a
    if cls is InflClass.E:
        stem = name + "e"
        return stem, stem, name + "tt" + a
    if cls is InflClass.NEN:
        stem = name[:-3] + "se"
        return stem, stem, name[:-3] + "st" + a
    if cls is InflClass.LONG:
        return name, name, name + "t" + a
    if cls is InflClass.KSE:
        stem = name[:-1] + "kse"
        return stem, stem, name + "t" + a
    # consonant-final loan names: Daniel -> Danielille
    stem = name + "i"
    return stem, stem, stem + a


def inflect(name: str, cls: InflClass | None = None, gradation: bool = False,
            colloquial: bool = True, vowel: str | None = None) -> dict[str, str]:
    """Generate case forms of a proper name.

    Returns a mapping from case label to form. With ``colloquial=True`` extra
    spoken forms are included (``Helsingis``, ``Mikolt``, ``Tampereel``).
    ``vowel`` overrides vowel harmony for compounds whose last part has only
    neutral vowels (*Rovaniemi* -> *Rovaniemellä*).
    """
    cls = cls or guess_class(name)
    a = vowel or harmony(name)
    strong, weak, part = _stems(name, cls, gradation, a)
    forms = {
        "nom": name,
        "gen": weak + "n",
        "part": part,
        "ine": weak + "ss" + a,
        "ela": weak + "st" + a,
        "ade": weak + "ll" + a,
        "abl": weak + "lt" + a,
        "all": weak + "lle",
        "tra": weak + "ksi",
        "ess": strong + "n" + a,
    }
    if cls is InflClass.LONG:
        last = name[-1]
        forms["ill"] = name + ("h" + last + "n" if last in "uyi" else "seen")
    elif cls is InflClass.E:
        forms["ill"] = strong + "seen"   # Tampereeseen
    elif cls in (InflClass.NEN, InflClass.OLD_I, InflClass.KSE):
        forms["ill"] = strong + "en"     # Virtaseen, Lahteen
    else:
        forms["ill"] = strong + strong[-1] + "n"  # Mikkoon, Helsinkiin
    if colloquial:
        # Spoken Finnish drops the final vowel: Helsingis, Tampereel, Mikolt, Oulust.
        for case in ("ine", "ela", "ade", "abl"):
            forms[case + "_col"] = forms[case][:-1]
        forms["ine_col2"] = weak + "s"  # Helsingis, Oulus
        forms["tra_col"] = weak + "ks"
        forms["all_col"] = weak + "l"
        # Illative loses its -n: Kamppii, Otaniemee, Tampereesee.
        forms["ill_col"] = forms["ill"][:-1] if forms["ill"].endswith("n") else forms["ill"]
    return forms


@dataclass(frozen=True)
class Entry:
    lemma: str
    kind: str  # "place" | "person"
    cls: InflClass | None = None
    gradation: bool = False
    canonical: str | None = None  # for nicknames/aliases: the name they refer to
    vowel: str | None = None      # harmony override for compounds

    @property
    def target(self) -> str:
        return self.canonical or self.lemma


@dataclass
class Lexicon:
    """An index from every generated surface form to its lemma."""

    entries: list[Entry] = field(default_factory=list)
    colloquial: bool = True
    _index: dict[str, list[Entry]] = field(default_factory=dict, init=False, repr=False)
    _max_tokens: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        for entry in list(self.entries):
            self._index_entry(entry)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)
        self._index_entry(entry)

    def _index_entry(self, entry: Entry) -> None:
        # Only the last word of a multi-word name inflects: "Henna Virtaselle".
        words = entry.lemma.split()
        head, last = words[:-1], words[-1]
        for form in inflect(last, entry.cls, entry.gradation, self.colloquial, entry.vowel).values():
            key = " ".join([*(w.lower() for w in head), form.lower()])
            bucket = self._index.setdefault(key, [])
            if entry not in bucket:
                bucket.append(entry)
        self._max_tokens = max(self._max_tokens, len(words))

    def lookup(self, form: str, kind: str | None = None) -> Entry | None:
        for entry in self._index.get(form.lower(), []):
            if kind is None or entry.kind == kind:
                return entry
        return None

    def find(self, tokens: list[str], kind: str | None = None) -> Iterable[tuple[int, int, Entry]]:
        """Yield (start, end, entry) for every longest match in ``tokens``."""
        i = 0
        while i < len(tokens):
            for size in range(min(self._max_tokens, len(tokens) - i), 0, -1):
                entry = self.lookup(" ".join(tokens[i:i + size]), kind)
                if entry is not None:
                    yield i, i + size, entry
                    i += size
                    break
            else:
                i += 1


# --- open-vocabulary analysis ------------------------------------------------------

_ENDINGS = sorted({"n", "a", "ä", "ta", "tä", "ssa", "ssä", "sta", "stä", "lla", "llä", "lta", "ltä",
                   "lle", "ksi", "na", "nä", "ss", "st", "ll", "lt", "l", "ks", "s", "seen", "en",
                   "an", "än", "on", "un", "yn", "in", "ön"}, key=len, reverse=True)
_STRENGTHEN = [("ng", "nk"), ("mm", "mp"), ("ll", "lt"), ("nn", "nt"), ("rr", "rt"), ("hd", "ht"),
               ("k", "kk"), ("p", "pp"), ("t", "tt"), ("d", "t"), ("v", "p")]


# Gradation reversals that are common inside Finnish given names (Riikalle -> Riikka,
# Sadulle -> Satu). The others (mm -> mp, nn -> nt, ...) are real but rare in names,
# where geminates are usually underlying: Emmalle -> Emma, not *Empa.
_NAME_GRADATIONS = {"k", "p", "t", "d"}


def guess_lemmas(form: str, kind: str | None = None) -> list[tuple[str, InflClass, bool]]:
    """Propose (lemma, class, gradation) analyses of an unknown inflected name.

    Analysis by synthesis: strip a candidate ending, undo gradation, guess the
    class, and keep only candidates whose generated paradigm actually contains
    ``form``. With ``kind="person"`` candidates are re-ranked with priors for
    given names: -e names do not gradate (Petelle -> Pete), k/p/t gradation is
    common (Riikalle -> Riikka), other reversals are rare (Emmalle -> Emma),
    and old i-stems are place-name territory (Lahti), not people.
    """
    form = form.lower()
    seen: set[str] = set()
    scored: list[tuple[int, int, tuple[str, InflClass, bool]]] = []
    for ending in _ENDINGS:
        if not form.endswith(ending) or len(form) - len(ending) < 2:
            continue
        stem = form[: -len(ending)]
        # (lemma, class, gradation, which weak grade was reversed)
        candidates: list[tuple[str, InflClass | None, bool, str]] = []
        if stem.endswith("kse"):
            candidates.append((stem[:-3] + "s", InflClass.KSE, False, ""))       # Tuomakse-
        elif stem.endswith("se"):
            candidates.append((stem[:-2] + "nen", InflClass.NEN, False, ""))     # Virtase-
        if stem.endswith("ee"):
            candidates.append((stem[:-1], InflClass.E, False, ""))               # Tamperee-
        elif stem.endswith("e"):
            ebody = stem[:-1]
            for weak, strong in _STRENGTHEN:                                      # Lahde- -> Lahti
                if ebody.endswith(weak):
                    candidates.append((ebody[: -len(weak)] + strong + "i", InflClass.OLD_I, True, weak))
                    break
            candidates.append((ebody + "i", InflClass.OLD_I, False, ""))
        body, last = stem[:-1], stem[-1:]
        if last in _VOWELS:
            for weak, strong in _STRENGTHEN:
                if body.endswith(weak):
                    candidates.append((body[: -len(weak)] + strong + last, InflClass.VOWEL, True, weak))
                    break
            candidates.append((stem, InflClass.VOWEL, False, ""))
        for lemma, cls, grad, weak in candidates:
            if lemma in seen:
                continue
            paradigm = {f.lower() for v in ("a", "ä") for f in inflect(lemma, cls, grad, vowel=v).values()}
            if form not in paradigm:
                continue
            seen.add(lemma)
            score = 0
            if kind == "person":
                if cls is InflClass.OLD_I:
                    score -= 2
                if cls is InflClass.VOWEL and not grad and lemma.endswith("e"):
                    score += 2
                if grad:
                    score += 1 if weak in _NAME_GRADATIONS else -1
            elif grad:
                score += 1
            scored.append((-score, len(scored), (lemma, cls or guess_class(lemma), grad)))
    return [item for _, _, item in sorted(scored)]
