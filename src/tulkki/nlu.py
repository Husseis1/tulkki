"""Rule-based Finnish NLU that turns an utterance into a tool call.

The output schema mirrors what an LLM agent would emit via function calling,
so the same benchmark can score this parser, an LLM, or a hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gazetteer import default_contacts, default_places
from .morphology import InflClass, Lexicon, guess_lemmas
from .text import tokenize
from .timeexpr import parse_day, parse_duration, parse_time

__all__ = ["Config", "Parser", "ToolCall", "TOOLS"]

TOOLS: list[dict[str, Any]] = [
    {"name": "set_alarm", "description": "Set an alarm at a clock time.",
     "parameters": {"type": "object", "properties": {
         "time": {"type": "string", "description": "24h HH:MM"},
         "day": {"type": "string", "enum": ["today", "tomorrow", "day_after_tomorrow", "monday", "tuesday",
                                            "wednesday", "thursday", "friday", "saturday", "sunday"]}},
         "required": ["time"]}},
    {"name": "set_timer", "description": "Start a countdown timer.",
     "parameters": {"type": "object", "properties": {"seconds": {"type": "integer"}}, "required": ["seconds"]}},
    {"name": "call", "description": "Phone a contact.",
     "parameters": {"type": "object", "properties": {"contact": {"type": "string"}}, "required": ["contact"]}},
    {"name": "send_message", "description": "Send a text message to a contact.",
     "parameters": {"type": "object", "properties": {"contact": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["contact", "text"]}},
    {"name": "get_weather", "description": "Weather forecast for a place (default: current location).",
     "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "day": {"type": "string"}}}},
    {"name": "navigate", "description": "Start navigation to a destination.",
     "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}, "required": ["destination"]}},
]

_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("set_timer", ("ajastin", "ajastim", "ajasta", "munakello", "timer", "ajastus")),
    ("set_alarm", ("herät", "heräty", "herätä", "heratä", "herätt", "alarm")),
    ("send_message", ("viest", "tekstaa", "tekstari", "tekstiviest", "whatsapp")),
    ("call", ("soita", "soitto", "soitta", "pirauta", "pirautus", "soitel", "puhelu", "puhelun")),
    ("get_weather", ("sää", "keli", "sataa", "sataak", "lämpöt", "lämmin", "lämmintä", "pakkas",
                     "kylmä", "sateenvarj", "tuulee", "lunta", "aurinkoinen", "paistaak")),
    ("navigate", ("navigoi", "navigaat", "reitti", "reitin", "ajo-ohjeet", "opasta", "vie", "pääsen",
                  "pääsee", "suunnista", "mennään", "mennää", "ajetaan", "ajetaa")),
]
_MESSAGE_FILLER = {"viesti", "viestin", "viestiä", "viestii", "tekstari", "tekstarin", "tekstaa",
                   "lähetä", "laita", "lähettäisitkö", "kirjoita", "kirjota", "whatsapp", "whatsappiin",
                   "tekstiviesti", "tekstiviestin", "tekstiviestillä", "jolle", "sano", "kerro"}


_CONTACT_VERBS = ("soita", "soitta", "pirauta", "tekstaa", "lähetä", "kerro", "sano", "ilmoita",
                  "kirjoita", "kirjota", "puhelu", "viesti")
_REPAIR = {"eiku", "siis", "tarkoitan", "korjaan", "sori"}


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args}


@dataclass
class Config:
    """Ablation switches used by the benchmark."""

    morphology: bool = True   # match inflected forms (Mikolle) or only nominative (Mikko)
    colloquial: bool = True   # include spoken forms (Helsingis, Mikolt)
    open_vocab: bool = True   # guess lemmas of names missing from the lexicon (Riikalle -> Riikka)
    repair: bool = True       # honour self-corrections ("kahdeksalta, eiku seittemältä")


class Parser:
    def __init__(self, config: Config | None = None, places: Lexicon | None = None,
                 contacts: Lexicon | None = None) -> None:
        self.config = config or Config()
        self.places = places or default_places(self.config.colloquial)
        self.contacts = contacts or default_contacts(self.config.colloquial)

    # -- helpers -------------------------------------------------------------
    def _find(self, lex: Lexicon, tokens: list[str], kind: str) -> tuple[int, int, str] | None:
        for start, end, entry in lex.find(tokens, kind):
            if not self.config.morphology and " ".join(tokens[start:end]) != entry.lemma.lower():
                continue
            return start, end, entry.target
        return None

    def _contact(self, tokens: list[str]) -> tuple[int, int, str] | None:
        hit = self._find(self.contacts, tokens, "person")
        if not self.config.open_vocab:
            return hit
        if hit:
            start, end, name = hit
            # "Mikko Virtaselle": a known first name in the nominative followed by an
            # inflected surname that is not in the lexicon.
            if end < len(tokens) and tokens[start:end] == [w.lower() for w in name.split()][:end - start]:
                surname = [g for g in guess_lemmas(tokens[end], kind="person") if g[1] is InflClass.NEN]
                if surname and " " not in name:
                    return start, end + 1, f"{name} {surname[0][0].capitalize()}"
            return hit
        # Unknown name: analyse the token(s) right after the verb.
        for i, tok in enumerate(tokens[:-1]):
            if not tok.startswith(_CONTACT_VERBS):
                continue
            first = tokens[i + 1]
            nxt = tokens[i + 2] if i + 2 < len(tokens) else ""
            # "Ville Nieminen" / "Ville Niemiselle": first name in the nominative + surname.
            if nxt.endswith("nen"):
                return i + 1, i + 3, f"{first.capitalize()} {nxt.capitalize()}"
            surname = [g for g in guess_lemmas(nxt) if g[1] is InflClass.NEN] if nxt else []
            if surname:
                return i + 1, i + 3, f"{first.capitalize()} {surname[0][0].capitalize()}"
            if first.endswith(("lle", "l")) and (guesses := guess_lemmas(first, kind="person")):
                return i + 1, i + 2, guesses[0][0].capitalize()
        return None

    def _repair(self, tokens: list[str]) -> list[str]:
        """Keep only what follows the last self-correction marker."""
        if not self.config.repair:
            return tokens
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in _REPAIR or (tokens[i] == "kun" and i and tokens[i - 1] == "ei"):
                return tokens[:1] + tokens[i + 1:]  # keep the verb for context
        return tokens

    @staticmethod
    def intent(tokens: list[str]) -> str | None:
        # "kerro Sadulle et soitan illalla" is a message even though it contains "soitan".
        if tokens and tokens[0] in ("kerro", "sano", "ilmoita", "viesti", "kirjoita", "kirjota") \
                and {"että", "et", "ett"} & set(tokens):
            return "send_message"
        for name, prefixes in _KEYWORDS:
            if any(tok.startswith(prefixes) for tok in tokens):
                return name
        return None

    # -- main entry ------------------------------------------------------------
    def parse(self, text: str) -> ToolCall | None:
        tokens = tokenize(text)
        intent = self.intent(tokens)
        if intent is None:
            return None
        repaired = self._repair(tokens)

        def slot(fn):  # type: ignore[no-untyped-def]
            """Prefer the self-corrected tail of the utterance, fall back to all of it."""
            return (fn(repaired) if repaired is not tokens else None) or fn(tokens)

        if intent == "set_timer":
            dur = slot(parse_duration)
            return ToolCall("set_timer", {"seconds": dur.seconds} if dur else {})

        if intent == "set_alarm":
            args: dict[str, Any] = {}
            if t := slot(parse_time):
                args["time"] = t.hhmm
            if day := parse_day(tokens)[0]:
                args["day"] = day
            return ToolCall("set_alarm", args)

        if intent == "call":
            hit = slot(self._contact)
            return ToolCall("call", {"contact": hit[2]} if hit else {})

        if intent == "send_message":
            hit = slot(self._contact)
            args = {"contact": hit[2]} if hit else {}
            # The body is dictated content: never apply repair or slot logic inside it.
            body_hit = self._contact(tokens)
            args["text"] = self._message_text(tokens, body_hit)
            return ToolCall("send_message", args)

        place = slot(lambda t: self._find(self.places, t, "place"))
        if intent == "get_weather":
            args = {"location": place[2]} if place else {}
            if day := parse_day(tokens)[0]:
                args["day"] = day
            return ToolCall("get_weather", args)
        return ToolCall("navigate", {"destination": place[2]} if place else {})

    @staticmethod
    def _message_text(tokens: list[str], hit: tuple[int, int, str] | None) -> str:
        for marker in ("että", "et", "ett"):
            if marker in tokens:
                return " ".join(tokens[tokens.index(marker) + 1:])
        start = hit[1] if hit else 0
        rest = [t for t in tokens[start:] if t not in _MESSAGE_FILLER]
        return " ".join(rest)
