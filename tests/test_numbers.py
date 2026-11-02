import pytest

from tulkki.numbers import find_numbers, parse_number, to_words


@pytest.mark.parametrize("text,value", [
    ("kahdeltatoista", 12), ("kakskytviis", 25), ("kasilta", 8), ("yheltätoist", 11),
    ("kaks kymmentä viis", 25), ("kahdenkymmenenviiden", 25), ("kolmesataaneljäkymmentäkaksi", 342),
    ("tuhat yhdeksänsataa", 1900), ("nel kyt viis", 45), ("seittemält", 7), ("ysi", 9),
    ("viideks", 5), ("15", 15),
])
def test_parse(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize("word", ["yksin", "kuuta", "puoli", "kello", "yhteensä"])
def test_not_numbers(word):
    assert parse_number(word) is None


def test_bound_forms_need_context():
    assert find_numbers(["kymmentä"]) == []
    assert find_numbers(["kymmentä"], allow_bound=True)[0].value == 10


def test_adjacent_numbers_are_not_merged():
    spans = find_numbers("seittemän kolkyt".split())
    assert [s.value for s in spans] == [7, 30]


@pytest.mark.parametrize("form", ["nom", "gen", "abl", "tra", "all"])
@pytest.mark.parametrize("colloquial", [False, True])
def test_roundtrip(form, colloquial):
    for n in range(1, 100):
        assert parse_number(to_words(n, form, colloquial)) == n, (n, form, to_words(n, form, colloquial))
