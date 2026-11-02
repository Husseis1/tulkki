"""Benchmark an LLM (any OpenAI-compatible endpoint) on the same tool-calling task.

Two modes:

* ``llm`` – the model sees the utterance and the tool schemas, nothing else.
* ``hybrid`` – the prompt also carries Tulkki's deterministic analysis
  (normalised times, lemmatised names) as hints. This is the pattern for
  shipping LLMs in a morphologically rich market: let the model handle
  open-ended language, and ground the parts that must be exact.

Configuration comes from the environment: ``OPENAI_BASE_URL`` (default
https://api.openai.com/v1), ``OPENAI_API_KEY`` and the ``--model`` flag. Works
with local servers such as vLLM, llama.cpp or Ollama that expose the same API.
Standard library only.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .nlu import TOOLS, Parser

SYSTEM_PROMPT = """You are the language-understanding component of a Finnish voice assistant.
Convert the user's utterance into exactly one tool call.
Rules:
- Times are 24h HH:MM. Without "illalla"/"iltapäivällä" keep the hour as spoken (kahdeksalta -> 08:00).
- "puoli kahdeksan" means 07:30.
- day is one of today, tomorrow, day_after_tomorrow or an English weekday; omit if not said.
- contact and location are the base (nominative) form of the name: "Mikolle" -> "Mikko",
  "Helsingissä" -> "Helsinki". Nicknames: mutsi/äiskä -> äiti, faija/iskä -> isä, Stadi -> Helsinki.
- message text is the dictated content only, verbatim, lower case."""


def _chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "tool_choice": "required", "temperature": 0}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (URL is user-configured)
        return json.load(resp)


class LLMSystem:
    def __init__(self, model: str, hybrid: bool = False) -> None:
        self.model = model
        self.hybrid = hybrid
        self.parser = Parser() if hybrid else None
        self.tools = [{"type": "function", "function": t} for t in TOOLS]

    def __call__(self, text: str) -> dict[str, Any] | None:
        user = text
        if self.parser is not None:
            hint = self.parser.parse(text)
            user += f"\n\n[deterministic parser analysis: {json.dumps(hint.to_dict() if hint else None, ensure_ascii=False)}]"
        resp = _chat(self.model, [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user}], self.tools)
        calls = resp["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            return None
        fn = calls[0]["function"]
        return {"name": fn["name"], "args": json.loads(fn.get("arguments") or "{}")}
