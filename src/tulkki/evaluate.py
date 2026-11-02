"""Benchmark harness: score any utterance -> tool-call system.

Metrics
-------
* **intent accuracy** – right tool chosen
* **slot F1** – micro-averaged over (slot, value) pairs
* **exact match** – the whole call is correct; this is what the user feels

Results are broken down by slice (standard / colloquial / asr / ...), because
an aggregate number hides exactly the failures a market owner cares about.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .text import tokenize

System = Callable[[str], dict[str, Any] | None]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _norm_value(key: str, value: Any) -> Any:
    if key == "text" and isinstance(value, str):
        return " ".join(tokenize(value))
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _pairs(call: dict[str, Any] | None) -> set[tuple[str, Any]]:
    if not call:
        return set()
    return {(k, _norm_value(k, v)) for k, v in (call.get("args") or {}).items() if v not in (None, "")}


@dataclass
class Counts:
    n: int = 0
    intent: int = 0
    exact: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        for k in ("n", "intent", "exact", "tp", "fp", "fn"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def summary(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"n": self.n, "intent_acc": self.intent / max(self.n, 1),
                "slot_f1": f1, "exact_match": self.exact / max(self.n, 1)}


@dataclass
class Report:
    overall: Counts = field(default_factory=Counts)
    by_slice: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    by_intent: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    errors: list[dict[str, Any]] = field(default_factory=list)
    ms_per_utt: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.summary(),
            "by_slice": {k: v.summary() for k, v in sorted(self.by_slice.items())},
            "by_intent": {k: v.summary() for k, v in sorted(self.by_intent.items())},
            "ms_per_utt": self.ms_per_utt,
            "errors": self.errors,
        }


def score_one(expected: dict[str, Any], predicted: dict[str, Any] | None) -> Counts:
    c = Counts(n=1)
    if predicted and predicted.get("name") == expected["name"]:
        c.intent = 1
        gold, pred = _pairs(expected), _pairs(predicted)
    else:
        gold, pred = _pairs(expected), set()
        if predicted:
            pred = {("__wrong_tool__", predicted.get("name"))}
    c.tp = len(gold & pred)
    c.fp = len(pred - gold)
    c.fn = len(gold - pred)
    c.exact = int(c.intent == 1 and gold == pred)
    return c


def evaluate(system: System, examples: Iterable[dict[str, Any]], keep_errors: int = 1000) -> Report:
    report = Report()
    examples = list(examples)
    start = time.perf_counter()
    for ex in examples:
        try:
            pred = system(ex["text"])
        except Exception as exc:  # a crashing system scores zero, it does not kill the run
            pred = {"name": "__error__", "args": {"error": repr(exc)}}
        c = score_one(ex["expected"], pred)
        report.overall.add(c)
        report.by_slice[ex.get("slice", "all")].add(c)
        report.by_intent[ex["expected"]["name"]].add(c)
        if not c.exact and len(report.errors) < keep_errors:
            report.errors.append({"text": ex["text"], "slice": ex.get("slice"),
                                  "expected": ex["expected"], "predicted": pred})
    report.ms_per_utt = 1000 * (time.perf_counter() - start) / max(len(examples), 1)
    return report


def format_table(rows: dict[str, Report], slices: list[str] | None = None) -> str:
    """Markdown table of exact match per slice for several systems."""
    slices = slices or sorted({s for r in rows.values() for s in r.by_slice})
    head = "| system | " + " | ".join(slices) + " | overall EM | intent acc | slot F1 |"
    sep = "|" + "---|" * (len(slices) + 4)
    lines = [head, sep]
    for name, rep in rows.items():
        cells = [f"{rep.by_slice[s].summary()['exact_match']:.1%}" if s in rep.by_slice else "–" for s in slices]
        o = rep.overall.summary()
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | **{o['exact_match']:.1%}** | {o['intent_acc']:.1%} | {o['slot_f1']:.1%} |")
    return "\n".join(lines)
