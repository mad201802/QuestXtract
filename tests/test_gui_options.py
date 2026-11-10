from __future__ import annotations

import argparse
import dataclasses
import sys

import pytest

from questxtract.cli import FLAG_FIELDS, build_parser
from questxtract.config import Config
from questxtract.gui.options import EXCLUDED, OPTIONS


def _commands() -> dict[str, dict[str, argparse.Action]]:
    parser = build_parser()
    (subs,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    # The exports' quiz.json is the drop box's file, like the inputs of process and transcribe.
    return {
        name: {
            action.dest: action
            for action in subs.choices[name]._actions
            if not isinstance(action, argparse._HelpAction)
            and (name in ("process", "transcribe") or action.option_strings)
        }
        for name in ("process", "transcribe", "qti", "tex")
    }


def _cli_dests() -> set[str]:
    return {dest for options in _commands().values() for dest in options}


def test_every_cli_option_is_shown_or_excluded_with_a_reason():
    shown = {option.dest for option in OPTIONS}
    assert not shown & set(EXCLUDED), "an option is both shown and excluded"
    assert all(reason.strip() for reason in EXCLUDED.values()), "every exclusion needs a reason"
    missing = _cli_dests() - shown - set(EXCLUDED)
    assert not missing, (
        f"CLI options neither shown nor excluded in gui/options.py: {sorted(missing)}"
    )


def test_every_shown_option_is_a_real_cli_option():
    unknown = {option.dest for option in OPTIONS} - _cli_dests()
    assert not unknown, f"gui/options.py names options the CLI lacks: {sorted(unknown)}"


def test_each_option_matches_its_cli_flag_and_choices():
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
    fields = {field.name for field in dataclasses.fields(Config)}
    for option in OPTIONS:
        if option.field is None:
            continue
        assert option.field in fields, f"{option.dest}: {option.field} is not a Config field"
    mapped = {flag: field for flag, field in FLAG_FIELDS}
    for option in OPTIONS:
        if option.field is not None and option.dest in mapped:
            assert mapped[option.dest] == option.field, option.dest


def test_the_flag_mapping_lives_in_the_cli_only():
    from questxtract.gui import options

    assert not hasattr(options, "FLAG_FIELDS")


def test_a_new_cli_option_fails_the_parity_check(monkeypatch):
    real = build_parser

    def with_a_new_option() -> argparse.ArgumentParser:
        parser = real()
        (subs,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        subs.choices["process"].add_argument("--brand-new-flag", dest="brand_new_flag")
        return parser

    monkeypatch.setattr(sys.modules[__name__], "build_parser", with_a_new_option)
    with pytest.raises(AssertionError, match="brand_new_flag"):
        test_every_cli_option_is_shown_or_excluded_with_a_reason()
