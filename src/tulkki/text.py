"""Tokenisation shared by all modules.

Input may come straight from a speech recogniser (lower case, no punctuation)
or from typed text. We lower-case, keep clock tokens such as ``7.30`` and
``07:30`` intact, and strip other punctuation.
"""

from __future__ import annotations

import re

_CLOCK = re.compile(r"^\d{1,2}[:.]\d{2}$")
_PUNCT = re.compile(r"[^\w\s:.\-]")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _PUNCT.sub(" ", text.lower()).split():
        raw = raw.rstrip(".:")
        if _CLOCK.match(raw):
            tokens.append(raw.replace(".", ":"))
            continue
        raw = raw.strip(".:-")
        if raw:
            tokens.append(raw)
    return tokens
