"""A deliberately naive baseline: what you get by porting an English pipeline.

Same intent keywords as Tulkki, but slots are filled the English way:
digits-only times and durations, and an exact-match gazetteer of nominative
names. The gap between this and :class:`tulkki.nlu.Parser` is the value of
handling Finnish morphology.
"""

from __future__ import annotations

import re
from typing import Any

from .gazetteer import CONTACTS, PLACES
from .nlu import Parser
from .text import tokenize

_DAY_WORDS = {"huomenna": "tomorrow", "tänään": "today", "ylihuomenna": "day_after_tomorrow"}
_UNITS = {"sekuntia": 1, "minuuttia": 60, "tuntia": 3600}


def naive_parse(text: str) -> dict[str, Any] | None:
    tokens = tokenize(text)
    intent = Parser.intent(tokens)
    if intent is None:
        return None
    args: dict[str, Any] = {}
    joined = " ".join(tokens)
    if intent == "set_alarm":
        if m := re.search(r"\b(\d{1,2})(?::(\d{2}))?\b", joined):
            args["time"] = f"{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}"
    elif intent == "set_timer":
        if m := re.search(r"\b(\d+) (sekuntia|minuuttia|tuntia)\b", joined):
            args["seconds"] = int(m.group(1)) * _UNITS[m.group(2)]
    elif intent in ("call", "send_message"):
        for e in CONTACTS:
            if e.lemma.lower() in tokens:
                args["contact"] = e.target
                break
        if intent == "send_message":
            for marker in ("että", "et"):
                if marker in tokens:
                    args["text"] = " ".join(tokens[tokens.index(marker) + 1:])
                    break
    else:
        for e in PLACES:
            if e.lemma.lower() in tokens:
                args["location" if intent == "get_weather" else "destination"] = e.target
                break
    for tok in tokens:
        if tok in _DAY_WORDS and intent in ("set_alarm", "get_weather"):
            args["day"] = _DAY_WORDS[tok]
    return {"name": intent, "args": args}
