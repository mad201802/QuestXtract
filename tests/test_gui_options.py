"""The graphical app's option table stays in step with the CLI (issue #124, AC parity).

Headless: no tkinter here, so this runs in CI on every platform. A new CLI option fails the
test until the app shows it (OPTIONS) or leaves it out with a reason (EXCLUDED).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import pytest

from questxtract.cli import build_parser
from questxtract.config import Config
from questxtract.gui.options import EXCLUDED, FLAG_FIELDS, OPTIONS


def _commands() -> dict[str, dict[str, argparse.Action]]:
    """Map each subcommand to its options by destination, help excluded."""
    parser = build_parser()
    (subs,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return {
        name: {
            action.dest: action
            for action in subs.choices[name]._actions
            if not isinstance(action, argparse._HelpAction)
        }
        for name in ("process", "transcribe")
    }


def _cli_dests() -> set[str]:
    return {dest for options in _commands().values() for dest in options}


def test_every_cli_option_is_shown_or_excluded_with_a_reason():
    """AC: a new CLI option fails here until the app shows it or lists it as excluded."""
    shown = {option.dest for option in OPTIONS}
    assert not shown & set(EXCLUDED), "an option is both shown and excluded"
    assert all(reason.strip() for reason in EXCLUDED.values()), "every exclusion needs a reason"
    missing = _cli_dests() - shown - set(EXCLUDED)
    assert not missing, (
        f"CLI options neither shown nor excluded in gui/options.py: {sorted(missing)}"
    )


def test_every_shown_option_is_a_real_cli_option():
    """AC: the table names no option the CLI does not have (a rename or removal shows here)."""
    unknown = {option.dest for option in OPTIONS} - _cli_dests()
    assert not unknown, f"gui/options.py names options the CLI lacks: {sorted(unknown)}"


def test_each_option_matches_its_cli_flag_and_choices():
    """AC: the same flag strings and the same choices as the CLI, so the form cannot drift."""
    commands = _commands()
    for option in OPTIONS:
        actions = [cmd[option.dest] for cmd in commands.values() if option.dest in cmd]
        assert actions, option.dest
        for action in actions:
            table = {flag.strip() for flag in option.flag.split(",")}
            assert table == set(action.option_strings), (
                f"{option.dest}: table says {option.flag}, CLI has {action.option_strings}"
            )
            if action.choices is not None:
                assert tuple(action.choices) == option.choices, option.dest
            elif option.choices:
                raise AssertionError(f"{option.dest}: the table has choices the CLI does not")


def test_each_setting_is_a_config_field_and_keeps_its_default():
    """AC: defaults come from Config; the form has no numbers of its own to drift from it."""
    fields = {field.name for field in dataclasses.fields(Config)}
    for option in OPTIONS:
        if option.field is None:
            continue
        assert option.field in fields, f"{option.dest}: {option.field} is not a Config field"
    mapped = {flag: field for flag, field in FLAG_FIELDS}
    for option in OPTIONS:
        if option.field is not None and option.dest in mapped:
            assert mapped[option.dest] == option.field, option.dest


def test_the_cli_and_the_app_share_one_flag_mapping():
    """AC: the mapping the CLI applies is the one the app's table uses (no second copy)."""
    from questxtract import cli
    from questxtract.gui import options

    assert cli.FLAG_FIELDS is options.FLAG_FIELDS


def test_a_new_cli_option_fails_the_parity_check(monkeypatch):
    """The guard itself: an option added to the CLI with nothing in the app fails the test."""
    real = build_parser

    def with_a_new_option() -> argparse.ArgumentParser:
        parser = real()
        (subs,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        subs.choices["process"].add_argument("--brand-new-flag", dest="brand_new_flag")
        return parser

    monkeypatch.setattr(sys.modules[__name__], "build_parser", with_a_new_option)
    with pytest.raises(AssertionError, match="brand_new_flag"):
        test_every_cli_option_is_shown_or_excluded_with_a_reason()
