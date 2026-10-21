"""The graphical app's languages, headless (issue #124): every word in every language.

The table in gui/i18n.py is checked here, so a new key that one language lacks, or a placeholder
that one translation dropped, fails in CI on every platform.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from questxtract.gui import i18n, state
from questxtract.gui.options import OPTIONS

TRANSCRIPT = Path(__file__).parent / "data" / "ttk4225_2026_09_14.transcript.json"
PLACEHOLDER = re.compile(r"\{(\w+)\}")


@pytest.fixture(autouse=True)
def english() -> Iterator[None]:
    """Each test starts in English and leaves the app in English."""
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def test_every_language_has_every_message():
    """AC: no message is missing in one of the three languages."""
    assert set(i18n.LANGUAGES) == {"en", "nb", "it"}
    for key, translations in i18n._MESSAGES.items():
        assert set(translations) == set(i18n.LANGUAGES), key


def test_every_translation_keeps_the_placeholders_of_the_english():
    """AC: a translation fills the same {placeholders} as the English, so nothing is left blank."""
    for key, translations in i18n._MESSAGES.items():
        expected = set(PLACEHOLDER.findall(translations["en"]))
        for code, text in translations.items():
            assert set(PLACEHOLDER.findall(text)) == expected, (key, code)


def test_every_option_has_a_label_in_every_language():
    """AC: each control the form shows has its label translated (the CLI's flags are not)."""
    for option in OPTIONS:
        if option.kind == "implicit":
            continue
        assert i18n.has("opt." + option.dest), option.dest


def test_an_unknown_language_is_refused():
    """A switch to a language the app does not have raises, rather than showing nothing."""
    with pytest.raises(ValueError, match="no such language"):
        i18n.set_language("sv")
    assert i18n.language() == "en"


def test_tr_fills_placeholders_in_the_chosen_language():
    """The current language picks the words and the placeholders are filled in."""
    i18n.set_language("nb")
    assert i18n.tr("status.written", path="x.json") == "Skrevet: x.json"
    i18n.set_language("it")
    assert i18n.tr("status.written", path="x.json") == "Scritto: x.json"


def test_the_headless_messages_follow_the_language():
    """The messages the form computes (why Run is disabled, the hints) switch with the language."""
    problems = {}
    for code in i18n.LANGUAGES:
        i18n.set_language(code)
        problems[code] = state.run_problem(None, "questions", {}, None)
    assert problems["en"] == "Drop a recording or transcript here, or click Browse."
    assert problems["nb"].startswith("Slipp")
    assert problems["it"].startswith("Trascina")


def test_the_cli_refusals_stay_english_in_every_language():
    """The CLI owns its one-line errors; the app shows them unchanged in each language."""
    i18n.set_language("it")
    problem = state.run_problem(TRANSCRIPT, "transcript", {}, None)

    assert problem == f"{TRANSCRIPT}: expected an .mp4 recording or an .mp3 recording"


def test_stage_names_are_translated_and_unknown_ones_are_readable():
    """Progress names are shown in the language; a name the table lacks is still readable."""
    i18n.set_language("nb")
    assert state.stage_label("transcribe") == "Transkriberer"
    assert state.stage_label("some_new_stage") == "Some new stage"
