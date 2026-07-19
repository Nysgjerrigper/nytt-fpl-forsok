"""
Tests for API-availability scaling in fpl.run_week (TODO 3.1, audit A4b):
status / chance_of_playing_next_round from bootstrap-static -> per-player factor
-> scaled predictions. Mocked bootstrap dicts, no network.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.run_week import apply_availability, availability_multipliers


def _bootstrap(elements):
    return {"elements": elements}


def _el(first, second, status="a", chance=None):
    return {"first_name": first, "second_name": second,
            "status": status, "chance_of_playing_next_round": chance}


NAME_MAP = {"alpha one": 1, "bravo two": 2, "charlie three": 3}


def test_status_semantics():
    factors = availability_multipliers(_bootstrap([
        _el("Alpha", "One", status="a"),
        _el("Bravo", "Two", status="d", chance=50),
        _el("Charlie", "Three", status="i"),
    ]), NAME_MAP)
    assert factors == {1: 1.0, 2: 0.5, 3: 0.0}


def test_doubtful_without_quantified_chance_defaults_to_75():
    factors = availability_multipliers(_bootstrap([_el("Alpha", "One", status="d")]), NAME_MAP)
    assert factors[1] == 0.75


def test_injured_with_partial_chance_keeps_the_chance():
    # FPL sometimes marks a returning player 'i' with chance 25 - scale, don't zero.
    factors = availability_multipliers(_bootstrap([_el("Alpha", "One", status="i", chance=25)]), NAME_MAP)
    assert factors[1] == 0.25


def test_unmatched_api_players_are_skipped():
    factors = availability_multipliers(_bootstrap([_el("Nobody", "Known", status="i")]), NAME_MAP)
    assert factors == {}


def test_apply_scales_all_horizon_rows_and_defaults_to_one():
    preds = pd.DataFrame({
        "player_id": [1, 1, 3, 99],           # player 1 across two horizon GWs; 99 unknown
        "GW": [10, 11, 10, 10],
        "predicted_total_points": [4.0, 6.0, 5.0, 3.0],
    })
    out = apply_availability(preds, {1: 0.5, 3: 0.0})
    assert out["predicted_total_points"].tolist() == [2.0, 3.0, 0.0, 3.0]
    # input frame untouched
    assert preds["predicted_total_points"].tolist() == [4.0, 6.0, 5.0, 3.0]
