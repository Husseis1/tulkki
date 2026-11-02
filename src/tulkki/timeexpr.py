"""Finnish clock times, days and durations.

Handles the ways Finns actually say times to an assistant::

    kahdeltatoista            12:00   (ablative numeral = "at")
    puoli kahdeksalta         07:30   ("half to eight")
    varttia vaille viis       04:45
    kymmentä yli kuus         06:10
    seittemän kolkyt          07:30
    kasilta illalla           20:00
    huomenaamulla kuudelta    06:00 tomorrow
    viideks minuutiks         300 s
    puolentoista tunnin       5400 s

Every parser returns the token indices it consumed so that the NLU layer can
exclude them from free-text slots such as a message body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .numbers import find_numbers

__all__ = ["TimeOfDay", "Duration", "parse_time", "parse_duration", "parse_day"]

_WEEKDAYS = {
    "maanantai": "monday", "tiistai": "tuesday", "keskiviikko": "wednesday",
    "torstai": "thursday", "perjantai": "friday", "lauantai": "saturday", "sunnuntai": "sunday",
}
_EVENING = ("illalla", "illal", "illasta", "iltana", "iltapäivällä", "iltapäiväl",
            "iltapäivästä", "ip")
_MORNING = ("aamulla", "aamul", "aamusta", "aamuna", "aamupäivällä", "aamupäiväl", "ap")
_NIGHT = ("yöllä", "yöl", "yöstä")
_PAST = ("yli",)
_TO = ("vaille", "vailla", "vaile")


@dataclass
class TimeOfDay:
    hour: int
    minute: int
    used: set[int] = field(default_factory=set)

    @property
    def hhmm(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"


@dataclass
class Duration:
    seconds: int
    used: set[int] = field(default_factory=set)


def _is_quarter(tok: str) -> bool:
    return tok.startswith("vart")  # vartti, varttia, vartin, varttii, vartiks


def _is_half(tok: str) -> bool:
    return tok in ("puoli", "puol", "puolen", "puoleks", "puoleksi", "puolelta")


def _unit_seconds(tok: str) -> int | None:
    if tok.startswith(("sekun", "sek")):
        return 1
    if tok.startswith(("minuut", "minsa")) or tok == "min":  # not "minua", "minut"
        return 60
    if tok.startswith(("tunti", "tunni", "tuntee", "tunne", "tunt", "tunn")):
        return 3600
    return None


def parse_day(tokens: list[str]) -> tuple[str | None, str | None, set[int]]:
    """Return (day, part_of_day, used indices).

    ``day`` is today/tomorrow/day_after_tomorrow or an English weekday name;
    ``part_of_day`` is morning/evening/night when a compound like
    *huomisaamuna* carries it.
    """
    day = part = None
    used: set[int] = set()
    for i, tok in enumerate(tokens):
        if tok.startswith("ylihuom"):
            day = "day_after_tomorrow"
        elif tok.startswith(("huomen", "huomis")):
            day = "tomorrow"
        elif tok in ("tänään", "tänää", "tänä", "tänäiltana", "tänäaamuna"):
            day = "today"
        else:
            for fi, en in _WEEKDAYS.items():
                if tok.startswith(fi):
                    day = en
                    break
            else:
                continue
        used.add(i)
        if "aamu" in tok:
            part = "morning"
        elif "ilta" in tok or "illa" in tok:
            part = "evening"
        if i > 0 and tokens[i - 1] in ("ensi", "tulevana", "seuraavana"):
            used.add(i - 1)
    return day, part, used


def _part_of_day(tokens: list[str]) -> tuple[str | None, set[int]]:
    for i, tok in enumerate(tokens):
        if tok in _EVENING or tok.startswith("illa"):
            return "evening", {i}
        if tok in _MORNING:
            return "morning", {i}
        if tok in _NIGHT:
            return "night", {i}
    return None, set()


def _apply_period(hour: int, period: str | None) -> int:
    if period == "evening" and 1 <= hour < 12:
        return hour + 12
    if period == "night" and hour == 12:
        return 0
    if period == "morning" and hour == 12:
        return 0
    return hour % 24


def parse_time(tokens: list[str]) -> TimeOfDay | None:
    """Find a clock time in ``tokens``."""
    period, period_used = _part_of_day(tokens)
    _, day_part, _ = parse_day(tokens)
    period = period or day_part

    # 1. Explicit digits: 7:30, klo 7, kello 18
    for i, tok in enumerate(tokens):
        if ":" in tok and tok.replace(":", "").isdigit():
            h, m = map(int, tok.split(":"))
            return TimeOfDay(_apply_period(h, period), m, {i} | period_used)

    spans = find_numbers(tokens, allow_bound=True)
    # Spans directly followed by a duration unit are durations, not clock times.
    spans = [s for s in spans if not (s.end < len(tokens) and _unit_seconds(tokens[s.end]))]
    by_start = {s.start: s for s in spans}

    def hour_after(idx: int) -> tuple[int, int] | None:
        span = by_start.get(idx)
        if span and 1 <= span.value <= 24:
            return span.value, span.end
        return None

    for i, tok in enumerate(tokens):
        # 2. puoli N -> (N-1):30
        if _is_half(tok) and (h := hour_after(i + 1)):
            hour, end = h
            base = hour - 1 if hour > 1 else 12  # puoli yksi = 12:30
            return TimeOfDay(_apply_period(base, period), 30, set(range(i, end)) | period_used)
        # 3. varttia yli / vaille N
        if _is_quarter(tok) and i + 1 < len(tokens) and (h := hour_after(i + 2)):
            hour, end = h
            if tokens[i + 1] in _PAST:
                return TimeOfDay(_apply_period(hour, period), 15, set(range(i, end)) | period_used)
            if tokens[i + 1] in _TO:
                return TimeOfDay(_apply_period(hour - 1 if hour > 1 else 12, period), 45, set(range(i, end)) | period_used)

    # 4. M (minuuttia) yli / vaille N
    for span in spans:
        j = span.end
        if j < len(tokens) and tokens[j].startswith("minuut"):
            j += 1
        if j < len(tokens) and (tokens[j] in _PAST or tokens[j] in _TO) and (h := hour_after(j + 1)):
            hour, end = h
            if not 1 <= span.value <= 59:
                continue
            used = set(range(span.start, end)) | period_used
            if tokens[j] in _PAST:
                return TimeOfDay(_apply_period(hour, period), span.value, used)
            return TimeOfDay(_apply_period(hour - 1 if hour > 1 else 12, period), 60 - span.value, used)

    # 5. (kello) H [M]. "viisi kymmenen" after kello is 05:10, not 50.
    resplit: list = []
    for span in spans:
        follows_hour = bool(resplit) and resplit[-1].end == span.start
        if span.value > 24 and span.end - span.start > 1 and not follows_hour:
            resplit += find_numbers(tokens[:span.start + 1], allow_bound=True)[-1:]
            resplit += [type(span)(s.start + span.start + 1, s.end + span.start + 1, s.value)
                        for s in find_numbers(tokens[span.start + 1:span.end], allow_bound=True)]
        else:
            resplit.append(span)
    spans = resplit
    for idx, span in enumerate(spans):
        if not 0 <= span.value <= 24:
            continue
        prev = tokens[span.start - 1] if span.start > 0 else ""
        has_kello = prev in ("kello", "klo", "kellon", "kelloon")
        next_span = spans[idx + 1] if idx + 1 < len(spans) else None
        minute = 0
        end = span.end
        if next_span and next_span.start == span.end and 0 <= next_span.value <= 59:
            minute = next_span.value
            end = next_span.end
        elif (span.end + 1 < len(tokens) and tokens[span.end] == "nolla" and next_span
              and next_span.start == span.end + 1 and next_span.value < 10):
            minute = next_span.value  # "kuusi nolla viisi" -> 06:05
            end = next_span.end
        used = set(range(span.start - (1 if has_kello else 0), end)) | period_used
        return TimeOfDay(_apply_period(span.value, period), minute, used)
    return None


def parse_duration(tokens: list[str]) -> Duration | None:
    """Sum every duration component: ``tunti ja kakskyt minuuttia`` -> 4800 s."""
    total = 0
    used: set[int] = set()
    spans = {s.start: s for s in find_numbers(tokens, allow_bound=True)}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # puolitoista / puolentoista / puoltoist + unit
        if tok.startswith("puol") and "toist" in tok and i + 1 < len(tokens) and (u := _unit_seconds(tokens[i + 1])):
            total += int(1.5 * u)
            used |= {i, i + 1}
            i += 2
            continue
        if _is_half(tok) and i + 1 < len(tokens) and (u := _unit_seconds(tokens[i + 1])):
            total += u // 2
            used |= {i, i + 1}
            i += 2
            continue
        if i in spans:
            span = spans[i]
            if span.end < len(tokens) and (u := _unit_seconds(tokens[span.end])):
                total += span.value * u
                used |= set(range(span.start, span.end + 1))
                i = span.end + 1
                continue
        if _is_quarter(tok) and not (i + 1 < len(tokens) and tokens[i + 1] in _PAST + _TO):
            total += 900
            used.add(i)
        elif (u := _unit_seconds(tok)) and (i == 0 or i - 1 not in used):
            total += u  # "tunniks", "minuutiks" with no number = one unit
            used.add(i)
        i += 1
    return Duration(total, used) if total else None
