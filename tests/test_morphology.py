import pytest

from tulkki.gazetteer import default_contacts, default_places
from tulkki.morphology import InflClass, gradate, guess_lemmas, harmony, inflect


@pytest.mark.parametrize("strong,weak", [
    ("Mikko", "Miko"), ("Helsinki", "Helsingi"), ("Lahte", "Lahde"), ("Turku", "Turu"),
    ("Lappeenranta", "Lappeenranna"), ("Satu", "Sadu"), ("Kamppi", "Kampi"), ("Liisa", "Liisa"),
])
def test_gradation(strong, weak):
    assert gradate(strong) == weak


def test_harmony():
    assert harmony("Jyväskylä") == "ä"
    assert harmony("Lappeenranta") == "a"
    assert harmony("Helsinki") == "ä"


@pytest.mark.parametrize("name,cls,grad,case,form", [
    ("Helsinki", InflClass.VOWEL, True, "ine", "Helsingissä"),
    ("Helsinki", InflClass.VOWEL, True, "ill", "Helsinkiin"),
    ("Lahti", InflClass.OLD_I, True, "ine", "Lahdessa"),
    ("Lahti", InflClass.OLD_I, True, "ill", "Lahteen"),
    ("Tampere", InflClass.E, False, "ade", "Tampereella"),
    ("Tampere", InflClass.E, False, "ill", "Tampereeseen"),
    ("Espoo", InflClass.LONG, False, "ill", "Espooseen"),
    ("Joensuu", InflClass.LONG, False, "ill", "Joensuuhun"),
    ("Virtanen", InflClass.NEN, False, "all", "Virtaselle"),
    ("Tuomas", InflClass.KSE, False, "all", "Tuomakselle"),
    ("Kuopio", InflClass.VOWEL, False, "part", "Kuopiota"),
    ("Mikko", InflClass.VOWEL, True, "abl_col", "Mikolt"),
])
def test_inflect(name, cls, grad, case, form):
    assert inflect(name, cls, grad)[case] == form


def test_vowel_override_for_compounds():
    assert inflect("Rovaniemi", InflClass.OLD_I, vowel="ä")["ade"] == "Rovaniemellä"


@pytest.mark.parametrize("form,lemma", [
    ("helsingis", "Helsinki"), ("stadissa", "Helsinki"), ("itikseen", "Itäkeskus"),
    ("seinäjoella", "Seinäjoki"), ("lentokentälle", "Helsinki-Vantaan lentoasema"),
])
def test_place_lookup(form, lemma):
    assert default_places().lookup(form).target == lemma


@pytest.mark.parametrize("form,lemma", [
    ("mikolle", "Mikko"), ("sadulle", "Satu"), ("mutsille", "äiti"), ("henna virtaselle", "Henna Virtanen"),
])
def test_contact_lookup(form, lemma):
    assert default_contacts().lookup(form).target == lemma


@pytest.mark.parametrize("form,lemma", [
    ("riikalle", "riikka"), ("virtaselle", "virtanen"), ("tuomakselle", "tuomas"),
    ("lahdessa", "lahti"), ("tampereella", "tampere"), ("rovaniemellä", "rovaniemi"),
])
def test_guess_lemmas(form, lemma):
    assert guess_lemmas(form)[0][0] == lemma


@pytest.mark.parametrize("form,lemma", [
    ("petelle", "pete"), ("emmalle", "emma"), ("hannalle", "hanna"), ("kallelle", "kalle"),
    ("riikalle", "riikka"), ("sadulle", "satu"),
])
def test_guess_person_names(form, lemma):
    assert guess_lemmas(form, kind="person")[0][0] == lemma
