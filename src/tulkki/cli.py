"""Command line interface.

    tulkki parse "herätä mut huomenna puol kasilta"
    tulkki bench                       # all rule systems on all datasets -> results/
    tulkki bench --llm gpt-4o-mini     # add an LLM and an LLM+Tulkki hybrid
    tulkki inflect Lahti --cls old_i --gradation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baseline import naive_parse
from .evaluate import evaluate, format_table, load_jsonl
from .morphology import InflClass, inflect
from .nlu import Config, Parser

DATASETS = {
    "dev": "data/dev.jsonl",
    "heldout": "data/heldout.jsonl",
    "test2": "data/test2.jsonl",
    "synthetic": "data/synthetic.jsonl",
}


def _wrap(parser: Parser):  # type: ignore[no-untyped-def]
    def run(text: str) -> dict[str, Any] | None:
        call = parser.parse(text)
        return call.to_dict() if call else None
    return run


def systems(llm: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "naive baseline (English-style slots)": naive_parse,
        "Tulkki − name inflection (nominative only)": _wrap(Parser(Config(morphology=False))),
        "Tulkki − colloquial forms": _wrap(Parser(Config(colloquial=False))),
        "Tulkki − open-vocab & repair": _wrap(Parser(Config(open_vocab=False, repair=False))),
        "**Tulkki (full)**": _wrap(Parser()),
    }
    if llm:
        from .llm import LLMSystem
        out[f"LLM ({llm})"] = LLMSystem(llm)
        out[f"LLM ({llm}) + Tulkki hints"] = LLMSystem(llm, hybrid=True)
    return out


def bench(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    md = ["# Benchmark results", "", "Exact match (the whole tool call is correct) per slice.", ""]
    for ds, rel in DATASETS.items():
        path = root / rel
        if not path.exists():
            continue
        examples = load_jsonl(path)
        reports = {name: evaluate(fn, examples) for name, fn in systems(args.llm).items()}
        md += [f"## {ds} (n={len(examples)})", "", format_table(reports, ["standard", "colloquial", "asr"]), ""]
        full = reports["**Tulkki (full)**"]
        (out_dir / f"{ds}_full.json").write_text(
            json.dumps(full.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        md.append(f"Tulkki latency: {full.ms_per_utt:.3f} ms/utterance (pure Python, single core)\n")
    (out_dir / "benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(prog="tulkki", description="Finnish NLU for voice assistants")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="parse one utterance into a tool call")
    p.add_argument("text")
    b = sub.add_parser("bench", help="run the benchmark")
    b.add_argument("--llm", help="model name for an OpenAI-compatible endpoint")
    b.add_argument("--root", default=".")
    i = sub.add_parser("inflect", help="show the paradigm of a name")
    i.add_argument("name")
    i.add_argument("--cls", choices=[c.value for c in InflClass])
    i.add_argument("--gradation", action="store_true")
    args = ap.parse_args()

    if args.cmd == "parse":
        call = Parser().parse(args.text)
        print(json.dumps(call.to_dict() if call else None, ensure_ascii=False, indent=2))
    elif args.cmd == "bench":
        bench(args)
    else:
        cls = InflClass(args.cls) if args.cls else None
        for case, form in inflect(args.name, cls, args.gradation).items():
            print(f"{case:9s} {form}")


if __name__ == "__main__":
    main()
