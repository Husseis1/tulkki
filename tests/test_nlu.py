from pathlib import Path

import pytest

from tulkki.evaluate import evaluate, load_jsonl
from tulkki.nlu import Parser
from tulkki.timeexpr import parse_duration, parse_time
from tulkki.text import tokenize

DATA = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize("text,hhmm", [
    ("kahdeltatoista", "12:00"), ("puoli kahdeksalta", "07:30"), ("varttia vaille viis", "04:45"),
    ("kymmentä yli kuus", "06:10"), ("seittemän kolkyt", "07:30"), ("kasilta illalla", "20:00"),
    ("klo 7.30", "07:30"), ("kello viisi kymmenen", "05:10"), ("kello kuusi nolla viisi", "06:05"),
    ("kaks kymmentä vaille kaheksan", "07:40"),
])
def test_time(text, hhmm):
    assert parse_time(tokenize(text)).hhmm == hhmm


@pytest.mark.parametrize("text,seconds", [
    ("viideks minuutiks", 300), ("puolentoista tunnin", 5400), ("tunti ja kakskyt minuuttia", 4800),
    ("vartiks", 900), ("kymmenen minuutin päästä muistuta minua", 600),
])
def test_duration(text, seconds):
    assert parse_duration(tokenize(text)).seconds == seconds


def test_parse_examples():
    p = Parser()
    assert p.parse("Herätä mut huomenna kasilta").to_dict() == {
        "name": "set_alarm", "args": {"time": "08:00", "day": "tomorrow"}}
    assert p.parse("tekstaa Mikolle et tuun kohta").to_dict() == {
        "name": "send_message", "args": {"contact": "Mikko", "text": "tuun kohta"}}
    assert p.parse("Navigoi Turkuun, ei kun Saloon").args == {"destination": "Salo"}
    assert p.parse("Soita Riikalle").args == {"contact": "Riikka"}


@pytest.mark.parametrize("name,minimum", [("dev.jsonl", 1.0), ("synthetic.jsonl", 1.0), ("heldout.jsonl", 1.0), ("test2.jsonl", 1.0)])
def test_no_regression(name, minimum):
    p = Parser()
    report = evaluate(lambda t: (c.to_dict() if (c := p.parse(t)) else None), load_jsonl(DATA / name))
    assert report.overall.summary()["exact_match"] >= minimum


@pytest.mark.parametrize("word,day", [
    ("keskiviikoksi", "wednesday"), ("keskiviikolla", "wednesday"), ("keskiviikkona", "wednesday"),
    ("maanantaiks", "monday"), ("lauantaiaamuna", "saturday"),
])
def test_inflected_weekdays(word, day):
    from tulkki.timeexpr import parse_day
    assert parse_day([word])[0] == day


@pytest.mark.parametrize("text,contact", [
    ("Soita Ville Nieminen", "Ville Nieminen"), ("Soita Ville Niemiselle", "Ville Nieminen"),
    ("Soita Mikko Virtaselle", "Mikko Virtanen"), ("Soita Petelle", "Pete"),
])
def test_unknown_contacts(text, contact):
    assert Parser().parse(text).args["contact"] == contact
