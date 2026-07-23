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


def _el(code, status="a", chance=None):
    # Matching is by FPL `code` (== dataset player_id) since TODO 4.8; names are
    # carried by the real API but unused for identity.
    return {"code": code, "first_name": "First", "second_name": f"Player{code}",
            "status": status, "chance_of_playing_next_round": chance}


def test_status_semantics():
    factors = availability_multipliers(_bootstrap([
        _el(1, status="a"),
        _el(2, status="d", chance=50),
        _el(3, status="i"),
    ]))
    assert factors == {1: 1.0, 2: 0.5, 3: 0.0}


def test_doubtful_without_quantified_chance_defaults_to_75():
    factors = availability_multipliers(_bootstrap([_el(1, status="d")]))
    assert factors[1] == 0.75


def test_injured_with_partial_chance_keeps_the_chance():
    # FPL sometimes marks a returning player 'i' with chance 25 - scale, don't zero.
    factors = availability_multipliers(_bootstrap([_el(1, status="i", chance=25)]))
    assert factors[1] == 0.25


def test_element_without_code_is_skipped():
    # Defensive: an API element missing `code` (malformed row) must not crash or
    # produce a None key.
    factors = availability_multipliers(_bootstrap([{"status": "i"}]))
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


# --- Live prices (TODO 3.2) share the same bootstrap plumbing ---

from fpl.run_week import apply_live_prices, live_prices


def test_live_prices_matches_and_converts():
    boot = _bootstrap([
        dict(_el(1), now_cost=55),
        {"status": "a", "now_cost": 100},        # no code -> skipped
        dict(_el(2), now_cost=None),             # no price -> skipped
    ])
    assert live_prices(boot) == {1: 55.0}


def test_apply_live_prices_overrides_only_known():
    preds = pd.DataFrame({
        "player_id": [1, 1, 99],
        "GW": [10, 11, 10],
        "value": [50.0, 50.0, 45.0],
        "predicted_total_points": [4.0, 6.0, 3.0],
    })
    out = apply_live_prices(preds, {1: 55.0})
    assert out["value"].tolist() == [55.0, 55.0, 45.0]  # both horizon rows re-priced; unknown kept
    assert preds["value"].tolist() == [50.0, 50.0, 45.0]  # input untouched


# --- Registered-player filter (live-readiness rehearsal finding, 2026-07-23) ---

from fpl.run_week import filter_to_registered


def test_filter_to_registered_drops_unregistered_players():
    preds = pd.DataFrame({
        "player_id": [1, 2, 3],
        "GW": [1, 1, 1],
        "predicted_total_points": [7.0, 5.0, 3.0],
    })
    boot = _bootstrap([dict(_el(1)), dict(_el(3)), {"status": "a"}])  # 2 absent; codeless row ignored
    out = filter_to_registered(preds, boot)
    assert out["player_id"].tolist() == [1, 3]
    assert preds["player_id"].tolist() == [1, 2, 3]  # input untouched
