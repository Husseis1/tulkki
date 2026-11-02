"""Built-in places and a demo contact list.

Each entry declares its inflection class and whether it undergoes consonant
gradation — the same information a Finnish dictionary (Kotus) records. Users'
own contacts can be added at runtime with :func:`tulkki.morphology.guess_class`,
so the lexicon is open-ended. Nicknames (``Stadi``, ``mutsi``) are entries with
a ``canonical`` target.
"""

from __future__ import annotations

from .morphology import Entry, InflClass, Lexicon

V, OI, E, NEN, L, C, KSE = (InflClass.VOWEL, InflClass.OLD_I, InflClass.E, InflClass.NEN,
                            InflClass.LONG, InflClass.CONS, InflClass.KSE)


def _p(lemma: str, cls: InflClass, grad: bool = False, canonical: str | None = None,
       vowel: str | None = None) -> Entry:
    return Entry(lemma, "place", cls, grad, canonical, vowel)


def _c(lemma: str, cls: InflClass, grad: bool = False, canonical: str | None = None) -> Entry:
    return Entry(lemma, "person", cls, grad, canonical)


PLACES: list[Entry] = [
    _p("Helsinki", V, True), _p("Stadi", V, canonical="Helsinki"),
    _p("Espoo", L), _p("Vantaa", L),
    _p("Tampere", E), _p("Mansesteri", V, canonical="Tampere"),
    _p("Turku", V, True), _p("Oulu", V), _p("Jyväskylä", V), _p("Lahti", OI, True),
    _p("Kuopio", V), _p("Pori", V), _p("Joensuu", L), _p("Lappeenranta", V, True),
    _p("Hämeenlinna", V), _p("Vaasa", V), _p("Seinäjoki", OI, True), _p("Rovaniemi", OI, vowel="ä"),
    _p("Mikkeli", V), _p("Kotka", V), _p("Salo", V), _p("Porvoo", L), _p("Kokkola", V),
    _p("Hanko", V, True), _p("Kajaani", V), _p("Kemi", V), _p("Savonlinna", V),
    _p("Kerava", V), _p("Inari", V),
    # Helsinki-region districts and landmarks used in navigation
    _p("Otaniemi", OI, vowel="ä"), _p("Kamppi", V, True), _p("Kallio", V), _p("Pasila", V),
    _p("Tikkurila", V), _p("Leppävaara", V), _p("Itäkeskus", KSE),
    _p("Itis", KSE, canonical="Itäkeskus"),
    _p("koti", V, True),  # resolved to the user's home address downstream
    _p("lentoasema", V, canonical="Helsinki-Vantaan lentoasema"),
    _p("lentokenttä", V, True, canonical="Helsinki-Vantaan lentoasema"),
]

CONTACTS: list[Entry] = [
    _c("Mikko", V, True), _c("Liisa", V), _c("Pekka", V, True), _c("Antti", V, True),
    _c("Matti", V, True), _c("Jukka", V, True), _c("Heikki", V, True), _c("Satu", V, True),
    _c("Outi", V, True), _c("Aino", V), _c("Leena", V), _c("Sanna", V), _c("Venla", V),
    _c("Eetu", V), _c("Kaisa", V), _c("Tuomas", KSE), _c("Daniel", C),
    _c("Henna Virtanen", NEN), _c("Henna", V, canonical="Henna Virtanen"),
    _c("Jari Korhonen", NEN), _c("Jari", V, canonical="Jari Korhonen"),
    _c("äiti", V, True), _c("mutsi", V, canonical="äiti"), _c("äiskä", V, canonical="äiti"),
    _c("isä", V), _c("faija", V, canonical="isä"), _c("iskä", V, canonical="isä"),
    _c("mummo", V), _c("mummi", V, canonical="mummo"),
]


def default_places(colloquial: bool = True) -> Lexicon:
    return Lexicon(list(PLACES), colloquial=colloquial)


def default_contacts(colloquial: bool = True) -> Lexicon:
    return Lexicon(list(CONTACTS), colloquial=colloquial)
