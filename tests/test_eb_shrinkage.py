"""
Sanity checks for the empirical-Bayes hierarchical shrinkage baseline
(fpl.model.baselines.add_eb_shrinkage_column). These guard the two properties
that make it trustworthy: the shrinkage math itself, and leakage-freeness
(a row's forecast must never see its own gameweek's outcomes - not even
other players').
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model.baselines import add_eb_shrinkage_column


def _frame(rows):
    return pd.DataFrame(rows, columns=["player_id", "GW_global", "position", "total_points"])


def test_debutant_gets_position_prior_not_own_score():
    # GW1: players 1 and 2 (MID) score 2 and 4 -> position prior entering GW2 is 3.0.
    # Player 3 debuts in GW2 with a huge haul; the forecast must be the prior (3.0),
    # completely unaffected by their own 20-point outcome.
    df = _frame([
        (1, 1, "MID", 2.0),
        (2, 1, "MID", 4.0),
        (3, 2, "MID", 20.0),
    ])
    out = add_eb_shrinkage_column(df)
    debut = out[(out["player_id"] == 3) & (out["GW_global"] == 2)]
    assert debut["eb_shrinkage_pred"].iloc[0] == pytest.approx(3.0)


def test_shrinkage_math_matches_conjugate_posterior_mean():
    # Player 1 at GW2: n=1 own gameweek (mean 2.0), position prior 3.0, k=10
    # -> (1*2 + 10*3) / (1+10) = 32/11.
    df = _frame([
        (1, 1, "MID", 2.0),
        (2, 1, "MID", 4.0),
        (1, 2, "MID", 6.0),
        (2, 2, "MID", 0.0),
    ])
    out = add_eb_shrinkage_column(df, prior_strength=10.0)
    row = out[(out["player_id"] == 1) & (out["GW_global"] == 2)]
    assert row["eb_shrinkage_pred"].iloc[0] == pytest.approx(32.0 / 11.0)


def test_first_gameweek_has_no_prior_and_forecasts_zero():
    # In the very first gameweek there is no earlier data at all - no player history,
    # no position prior - so the forecast must fall back to 0, like the other baselines.
    df = _frame([
        (1, 1, "MID", 5.0),
        (2, 1, "MID", 7.0),
    ])
    out = add_eb_shrinkage_column(df)
    assert (out["eb_shrinkage_pred"] == 0.0).all()


def test_position_prior_excludes_own_gameweek():
    # Player 3's first row is in GW2. The MID prior entering GW2 must be built from
    # GW1 only (mean 3.0) - if GW2's other scores leaked in, the value would shift.
    df = _frame([
        (1, 1, "MID", 2.0),
        (2, 1, "MID", 4.0),
        (1, 2, "MID", 10.0),
        (2, 2, "MID", 10.0),
        (3, 2, "MID", 0.0),
    ])
    out = add_eb_shrinkage_column(df)
    debut = out[(out["player_id"] == 3) & (out["GW_global"] == 2)]
    assert debut["eb_shrinkage_pred"].iloc[0] == pytest.approx(3.0)


def test_veteran_dominates_prior_as_history_grows():
    # A player with lots of history should sit close to their own mean, far from the prior.
    rows = [(99, gw, "FWD", 8.0) for gw in range(1, 42)]  # 41 GWs of scoring exactly 8
    rows += [(1, gw, "FWD", 0.0) for gw in range(1, 42)]   # prior-dragging teammate
    df = _frame(rows)
    out = add_eb_shrinkage_column(df, prior_strength=10.0)
    last = out[(out["player_id"] == 99) & (out["GW_global"] == 41)]
    # n=40, own mean 8.0, prior = 4.0 (pooled mean of both players) -> (40*8 + 10*4)/50 = 7.2
    assert last["eb_shrinkage_pred"].iloc[0] == pytest.approx(7.2)
