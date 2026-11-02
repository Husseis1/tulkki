"""Procedural generator of labelled Finnish assistant requests.

Hand-written test sets are small; this generator covers the combinatorics
(every contact x every case x every register, every clock time x every way of
saying it) and adds an ASR-noise channel that mimics what a recogniser emits:
lower case, no punctuation, compound numerals split into words, occasional
digits.

Usage::

    python -m tulkki.datagen --n 3000 --seed 0 --out data/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterator
from typing import Any

from .gazetteer import CONTACTS, PLACES
from .morphology import Entry, inflect
from .numbers import to_words

# Place names that take external locative cases (Tampereella, Vantaalle).
_EXTERNAL = {"Tampere", "Rovaniemi", "Seinäjoki", "Vantaa", "lentoasema", "lentokenttä", "Mansesteri"}
_DAYS = {
    None: ("", ""),
    "tomorrow": ("huomenna", "huomen"),
    "day_after_tomorrow": ("ylihuomenna", "ylihuomenna"),
    "today": ("tänään", "tänää"),
    "monday": ("maanantaina", "maanantaina"),
    "wednesday": ("keskiviikkona", "keskiviikkona"),
    "friday": ("perjantaina", "perjantaina"),
    "sunday": ("sunnuntaina", "sunnuntaina"),
}
_MESSAGES = [
    ("olen myöhässä", "oon myöhässä"),
    ("tulen kotiin kuudelta", "tuun kotiin kuudelta"),
    ("muista ostaa maitoa", "muista ostaa maitoo"),
    ("kiitos eilisestä", "kiitti eilisestä"),
    ("nähdään asemalla", "nähää asemal"),
    ("soitan sinulle myöhemmin", "soitan sulle myöhemmin"),
    ("onko sinulla avaimet", "onks sul avaimet"),
    ("kokous alkaa vähän myöhemmin", "palaveri alkaa vähä myöhemmin"),
]


def _forms(entry: Entry) -> dict[str, str]:
    last = entry.lemma.split()[-1]
    head = " ".join(entry.lemma.split()[:-1])
    forms = inflect(last, entry.cls, entry.gradation, True, entry.vowel)
    return {k: (head + " " + v).strip() for k, v in forms.items()}


def _cap(entry: Entry, form: str) -> str:
    return form if entry.lemma[0].islower() else form[0].upper() + form[1:]


# --- clock times ----------------------------------------------------------------

def _clock(rng: random.Random, col: bool) -> tuple[str, str]:
    """Return (spoken phrase, HH:MM)."""
    style = rng.choice(["kello", "abl", "half", "q_past", "q_to", "past", "to", "hm", "evening"])
    h = rng.randint(1, 11)
    w = lambda n, f: to_words(n, f, col)  # noqa: E731
    if style == "kello":
        return f"kello {w(h, 'nom')}", f"{h:02d}:00"
    if style == "abl":
        return w(h, "abl"), f"{h:02d}:00"
    if style == "half":
        return f"{'puol' if col else 'puoli'} {w(h + 1, 'abl')}", f"{h:02d}:30"
    if style == "q_past":
        return f"{'vartin' if col else 'varttia'} yli {w(h, 'nom')}", f"{h:02d}:15"
    if style == "q_to":
        return f"{'vartin' if col else 'varttia'} vaille {w(h + 1, 'nom')}", f"{h:02d}:45"
    if style in ("past", "to"):
        m = rng.choice([5, 10, 20, 25])
        mw = w(m, "nom") if m in (5, 25) or col else to_words(m, "part")
        if style == "past":
            return f"{mw} yli {w(h, 'nom')}", f"{h:02d}:{m:02d}"
        return f"{mw} vaille {w(h + 1, 'nom')}", f"{h:02d}:{60 - m:02d}"
    if style == "hm":
        m = rng.choice([10, 15, 20, 30, 35, 40, 45, 50])
        return f"kello {w(h, 'nom')} {w(m, 'nom')}", f"{h:02d}:{m:02d}"
    return f"{w(h, 'abl')} {'illal' if col else 'illalla'}", f"{h + 12:02d}:00"


def _alarm(rng: random.Random, col: bool) -> tuple[str, dict[str, Any]]:
    phrase, hhmm = _clock(rng, col)
    day = rng.choice([None, None, "tomorrow", "day_after_tomorrow", "monday", "friday"])
    day_word = _DAYS[day][1 if col else 0]
    opener = rng.choice(["herätä mut", "laita herätys", "herätys"] if col
                        else ["herätä minut", "aseta herätys", "laita herätys"])
    parts = [opener, day_word, phrase] if rng.random() < 0.5 else [opener, phrase, day_word]
    args: dict[str, Any] = {"time": hhmm}
    if day:
        args["day"] = day
    return " ".join(p for p in parts if p), {"name": "set_alarm", "args": args}


def _timer(rng: random.Random, col: bool) -> tuple[str, dict[str, Any]]:
    unit, secs = rng.choice([("minuutti", 60), ("minuutti", 60), ("tunti", 3600), ("sekunti", 1)])
    n = rng.choice([2, 3, 5, 7, 8, 10, 12, 15, 20, 25, 30, 45]) if unit != "tunti" else rng.choice([1, 2, 3])
    tra = {"minuutti": "minuutiks", "tunti": "tunniks", "sekunti": "sekunniks"}[unit] + ("" if col else "i")
    part = {"minuutti": "minuuttia", "tunti": "tuntia", "sekunti": "sekuntia"}[unit]
    gen = {"minuutti": "minuutin", "tunti": "tunnin", "sekunti": "sekunnin"}[unit]
    if col:
        part = part[:-1] + "i"  # minuuttii
    style = rng.choice(["tra", "part", "gen"])
    if style == "tra":
        num = to_words(n, "tra")
        num = num[:-1] if col else num  # viideks
        if n == 1:
            num = ""  # "tunniksi", not "yhdeksi tunniksi"
        text = f"{'laita' if col else 'aseta'} ajastin {num} {tra}"
    elif style == "part":
        text = f"ajastin {to_words(n, 'nom', col)} {part}"
    else:
        text = f"{'laita' if col else 'käynnistä'} {to_words(n, 'gen', col)} {gen} ajastin"
    return text, {"name": "set_timer", "args": {"seconds": n * secs}}


def _contact_form(rng: random.Random, col: bool) -> tuple[Entry, str]:
    entry = rng.choice(CONTACTS)
    forms = _forms(entry)
    form = forms["all_col"] if col and rng.random() < 0.4 else forms["all"]
    return entry, _cap(entry, form)


def _call(rng: random.Random, col: bool) -> tuple[str, dict[str, Any]]:
    entry, form = _contact_form(rng, col)
    verb = rng.choice(["soita", "pirauta", "soitatsä"] if col else ["soita", "voisitko soittaa"])
    return f"{verb} {form}", {"name": "call", "args": {"contact": entry.target}}


def _message(rng: random.Random, col: bool) -> tuple[str, dict[str, Any]]:
    entry, form = _contact_form(rng, col)
    body = rng.choice(_MESSAGES)[1 if col else 0]
    if col:
        text = rng.choice([f"tekstaa {form} et {body}", f"laita {form} viesti et {body}",
                           f"sano {form} et {body}"])
    else:
        text = rng.choice([f"lähetä {form} viesti että {body}", f"lähetä viesti {form} että {body}",
                           f"kerro {form} että {body}"])
    return text, {"name": "send_message", "args": {"contact": entry.target, "text": body}}


def _place(rng: random.Random) -> tuple[Entry, dict[str, str]]:
    entry = rng.choice(PLACES)
    return entry, _forms(entry)


def _weather(rng: random.Random, col: bool) -> tuple[str, dict[str, Any]]:
    entry, forms = _place(rng)
    ext = entry.lemma in _EXTERNAL
    loc = forms[("ade" if ext else "ine") + ("_col" if col else "")]
    if col and not ext and rng.random() < 0.5:
        loc = forms["ine_col2"]
    loc = _cap(entry, loc)
    day = rng.choice([None, "today", "tomorrow", "day_after_tomorrow", "wednesday", "sunday"])
    dw = _DAYS[day][1 if col else 0]
    if col:
        text = rng.choice([f"millane keli on {loc} {dw}", f"sataaks {loc} {dw}", f"onks {loc} kylmä {dw}"])
    else:
        text = rng.choice([f"millainen sää on {loc} {dw}", f"sataako {loc} {dw}",
                           f"mikä on sääennuste {loc} {dw}"])
    args: dict[str, Any] = {"location": entry.target}
    if day:
        args["day"] = day
    return text.strip(), {"name": "get_weather", "args": args}


def _navigate(rng: random.Random, col: bool) -> tuple[str, dict[str, Any]]:
    entry, forms = _place(rng)
    dest = forms["all"] if entry.lemma in _EXTERNAL else forms["ill" + ("_col" if col else "")]
    dest = _cap(entry, dest)
    verb = rng.choice(["vie mut", "mennään", "reitti", "miten pääsee"] if col
                      else ["navigoi", "näytä reitti", "opasta minut", "miten pääsen"])
    return f"{verb} {dest}", {"name": "navigate", "args": {"destination": entry.target}}


_GENERATORS = [_alarm, _timer, _call, _message, _weather, _navigate]

# --- ASR noise -------------------------------------------------------------------

_SPLIT = re.compile(r"(kymmentä|kymmenen|toista|sataa|kyt)(?=\w)")


def asr_noise(text: str, rng: random.Random) -> str:
    """Make text look like recogniser output."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = []
    for w in text.split():
        if rng.random() < 0.6:
            # kaksikymmentäviisi -> kaksi kymmentä viisi
            w = re.sub(r"(?<=\w)(kymmentä|kymmenen|toista|kyt)", r" \1", w)
            w = _SPLIT.sub(r"\1 ", w)
        words.append(w)
    return " ".join(words)


def generate(n: int, seed: int = 0) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for _ in range(n):
        gen = rng.choice(_GENERATORS)
        col = rng.random() < 0.5
        text, expected = gen(rng, col)
        text = re.sub(r"\s+", " ", text).strip()
        slice_ = "colloquial" if col else "standard"
        if rng.random() < 0.3:
            text, slice_ = asr_noise(text, rng), "asr"
        else:
            text = text[0].upper() + text[1:]
        yield {"text": text, "slice": slice_, "expected": expected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write UTF-8 JSONL here instead of stdout (use this on Windows)")
    args = ap.parse_args()
    lines = [json.dumps(ex, ensure_ascii=False) for ex in generate(args.n, args.seed)]
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
