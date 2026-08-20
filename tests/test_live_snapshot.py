"""
Guards the live-mode fix in fpl.run_week.build_live_snapshot: the synthetic
next-gameweek row must carry form features that INCLUDE the player's most
recent played match (the bug this replaced reused a played row's shifted
features, silently dropping every player's freshest game), and long-inactive
players must be excluded from the live pool.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl import run_week
from fpl.model.mid_gate import MidGateConfig, RegimeSelection, TertileThresholds


def _always_alt_mid_gate():
    """Small frozen gate used to exercise the live routing integration."""
    selection = RegimeSelection("lightgbm", 1, 1, 1.0, 1.0, 0.0)
    return MidGateConfig(
        champion="catboost", candidates=("catboost", "lightgbm"),
        thresholds=TertileThresholds(0.3, 0.7),
        selections={"low": selection, "medium": selection, "high": selection},
        mase_scale=1.0, mase_scale_training_max_gw=3,
    )


def _raw_frame(rows):
    return pd.DataFrame(rows, columns=["player_id", "GW_global", "position", "team",
                                       "name", "was_home", "total_points", "minutes", "value"])


def test_snapshot_form_includes_most_recent_match():
    # Player 1 plays GW1-3 with minutes [90, 0, 90]. As-of-now mins60 rate over those
    # three games is 2/3. The old approach reused GW3's shifted feature, which only
    # saw GW1-2 (rate 1/2) - the fix must produce 2/3.
    df = _raw_frame([
        (1, 1, "MID", "Arsenal", "A", 1, 5.0, 90, 60),
        (1, 2, "MID", "Arsenal", "A", 0, 0.0, 0, 60),
        (1, 3, "MID", "Arsenal", "A", 1, 6.0, 90, 60),
    ])
    snapshot = run_week.build_live_snapshot(df)
    assert len(snapshot) == 1
    assert snapshot["mins60_rate_roll5"].iloc[0] == pytest.approx(2.0 / 3.0)
    # Rolling points form must likewise include GW3's 6 points: mean(5, 0, 6).
    assert snapshot["total_points_roll3"].iloc[0] == pytest.approx(11.0 / 3.0)


def test_snapshot_excludes_long_inactive_players():
    # Player 2's last appearance is GW3; player 1 is current through GW60. With a
    # 38-GW activity window, only player 1 belongs in the live pool.
    rows = [(1, gw, "MID", "Arsenal", "A", 1, 2.0, 90, 60) for gw in range(1, 61)]
    rows += [(2, gw, "FWD", "Chelsea", "B", 1, 2.0, 90, 60) for gw in range(1, 4)]
    snapshot = run_week.build_live_snapshot(_raw_frame(rows))
    assert snapshot["player_id"].tolist() == [1]


def test_snapshot_rows_are_synthetic_future_rows():
    df = _raw_frame([
        (1, 1, "MID", "Arsenal", "A", 1, 5.0, 90, 60),
        (1, 2, "MID", "Arsenal", "A", 0, 3.0, 90, 60),
    ])
    snapshot = run_week.build_live_snapshot(df)
    # The snapshot row sits one GW past the last played one and keeps identity columns.
    assert snapshot["GW_global"].iloc[0] == 3
    assert snapshot["team"].iloc[0] == "Arsenal"
    assert snapshot["position"].iloc[0] == "MID"


def test_live_identity_uses_current_api_club_and_position():
    snapshot = pd.DataFrame([
        {"player_id": 101, "team": "Nottingham Forest", "position": "MID"},
        {"player_id": 202, "team": "Fulham", "position": "MID"},
    ])
    bootstrap = {
        "teams": [
            {"id": 1, "name": "Man City"},
            {"id": 2, "name": "Nott'm Forest"},
        ],
        "elements": [
            {"code": 101, "team": 1, "element_type": 3},
            {"code": 202, "team": 2, "element_type": 2},
        ],
    }

    updated = run_week.apply_live_identity(snapshot, bootstrap)

    assert updated.loc[updated["player_id"] == 101, "team"].iloc[0] == "Man City"
    assert updated.loc[updated["player_id"] == 101, "position"].iloc[0] == "MID"
    assert updated.loc[updated["player_id"] == 202, "team"].iloc[0] == "Nottingham Forest"
    assert updated.loc[updated["player_id"] == 202, "position"].iloc[0] == "DEF"


def test_live_identity_preserves_unregistered_snapshot_rows_until_filtering():
    snapshot = pd.DataFrame([
        {"player_id": 999, "team": "Historical FC", "position": "FWD"},
    ])
    bootstrap = {"teams": [], "elements": []}

    updated = run_week.apply_live_identity(snapshot, bootstrap)

    assert updated[["team", "position"]].iloc[0].tolist() == ["Historical FC", "FWD"]


def test_future_predictions_use_upcoming_opponents_form(monkeypatch):
    """Live opp_* staleness guard (TODO 3.6): the opp_* features on a future-GW prediction
    row must describe the UPCOMING opponent's trailing form, not whatever opponent the
    player happened to face in his last played match (which is what the snapshot rows
    carry)."""
    import numpy as np
    from fpl import features

    # Arsenal last played Chelsea (leaky defence: concedes 3/game). Upcoming fixture is
    # vs Wolves (concedes 1/game). Every team needs its own rows so team_form_asof sees it.
    rows = []
    for gw in (1, 2, 3):
        rows.append({"player_id": 1, "GW_global": gw, "position": "MID", "team": "Arsenal",
                     "name": "A", "was_home": 1, "total_points": 5.0, "minutes": 90,
                     "value": 60, "opponent_team": "Chelsea",
                     "goals_scored": 1.0, "goals_conceded": 0.0})
        rows.append({"player_id": 2, "GW_global": gw, "position": "DEF", "team": "Chelsea",
                     "name": "C", "was_home": 0, "total_points": 2.0, "minutes": 90,
                     "value": 50, "opponent_team": "Arsenal",
                     "goals_scored": 0.0, "goals_conceded": 3.0})
        rows.append({"player_id": 3, "GW_global": gw, "position": "DEF", "team": "Wolves",
                     "name": "W", "was_home": 1, "total_points": 2.0, "minutes": 90,
                     "value": 50, "opponent_team": "Brentford",
                     "goals_scored": 0.0, "goals_conceded": 1.0})
        rows.append({"player_id": 4, "GW_global": gw, "position": "DEF", "team": "Brentford",
                     "name": "B", "was_home": 0, "total_points": 2.0, "minutes": 90,
                     "value": 50, "opponent_team": "Wolves",
                     "goals_scored": 0.0, "goals_conceded": 2.0})
    raw = pd.DataFrame(rows)

    snapshot = run_week.build_live_snapshot(raw)
    feature_cols = ["opp_attack_roll6", "opp_defense_roll6", "opp_cs_rate_roll6"]

    bootstrap = {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Wolves"},
                           {"id": 3, "name": "Chelsea"}, {"id": 4, "name": "Brentford"}]}
    # GW4 fixtures: Arsenal (h) v Wolves, Chelsea (h) v Brentford.
    monkeypatch.setattr(run_week, "fetch_fixtures", lambda gw: [
        {"team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
        {"team_h": 3, "team_a": 4, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ])

    class _Stub:
        def predict(self, X):
            return np.zeros(len(X))

    models = {pos: _Stub() for pos in ("GK", "DEF", "MID", "FWD")}
    opp_form = features.team_form_asof(raw)
    preds = run_week.build_future_predictions(snapshot, feature_cols, models, bootstrap,
                                              start_gw=4, horizon=1, opp_form=opp_form)

    ars = preds[preds["player_id"] == 1].iloc[0]
    assert ars["opponent_team"] == "Wolves"
    # Wolves concede 1/game - the stale snapshot value would be Chelsea's 3/game.
    assert ars["opp_defense_roll6"] == 1.0
    assert snapshot.loc[snapshot["player_id"] == 1, "opp_defense_roll6"].iloc[0] == 3.0


def test_dgw_team_gets_one_prediction_row_per_fixture(monkeypatch):
    """Live DGW handling (TODO 3.3): a team with two fixtures in the target GW must yield
    TWO prediction rows per player (per-fixture opponent/home/FDR), matching the per-fixture
    representation backtest CSVs use - the optimizer sums per (player, GW)."""
    import numpy as np

    rows = []
    for gw in (1, 2, 3):
        for pid, team, opp in ((1, "Arsenal", "Chelsea"), (2, "Chelsea", "Arsenal"),
                               (3, "Wolves", "Brentford"), (4, "Brentford", "Wolves")):
            rows.append({"player_id": pid, "GW_global": gw, "position": "MID", "team": team,
                         "name": f"P{pid}", "was_home": 1, "total_points": 3.0, "minutes": 90,
                         "value": 50, "opponent_team": opp,
                         "goals_scored": 0.0, "goals_conceded": 1.0})
    raw = pd.DataFrame(rows)
    snapshot = run_week.build_live_snapshot(raw)

    bootstrap = {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Wolves"},
                           {"id": 3, "name": "Chelsea"}, {"id": 4, "name": "Brentford"}]}
    # GW4: Arsenal plays TWICE (h v Wolves, a v Chelsea); Brentford blanks.
    monkeypatch.setattr(run_week, "fetch_fixtures", lambda gw: [
        {"team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"team_h": 3, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 5},
    ])

    class _Stub:
        def predict(self, X):
            return np.full(len(X), 2.0)

    models = {pos: _Stub() for pos in ("GK", "DEF", "MID", "FWD")}
    preds = run_week.build_future_predictions(snapshot, ["value"], models, bootstrap,
                                              start_gw=4, horizon=1)

    ars = preds[preds["player_id"] == 1]
    assert len(ars) == 2                                    # one row per fixture
    assert sorted(ars["opponent_team"]) == ["Chelsea", "Wolves"]
    assert sorted(ars["was_home"]) == [0, 1]
    assert sorted(ars["fixture_difficulty"]) == [2, 5]      # per-fixture FDR, not the mean
    assert (preds[preds["player_id"] == 2]["opponent_team"] == "Arsenal").all()
    assert len(preds[preds["player_id"] == 3]) == 1         # single fixture -> single row
    assert len(preds[preds["player_id"] == 4]) == 0         # blank GW -> dropped


def test_live_future_predictions_routes_mid_rows_through_frozen_gate(monkeypatch):
    raw = _raw_frame([
        (1, 1, "MID", "Arsenal", "A", 1, 2.0, 90, 60),
        (1, 2, "MID", "Arsenal", "A", 1, 2.0, 90, 60),
        (1, 3, "MID", "Arsenal", "A", 1, 2.0, 90, 60),
    ])
    snapshot = run_week.build_live_snapshot(raw)
    monkeypatch.setattr(run_week, "fetch_fixtures", lambda gw: [
        {"team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ])
    bootstrap = {"teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}]}

    class Stub:
        def __init__(self, value): self.value = value
        def predict(self, X): return [self.value] * len(X)

    models = {pos: Stub(1.0) for pos in ("GK", "DEF", "MID", "FWD")}
    preds = run_week.build_future_predictions(
        snapshot, ["value"], models, bootstrap, start_gw=4, horizon=1,
        mid_gate=_always_alt_mid_gate(),
        mid_experts={"catboost": Stub(2.0), "lightgbm": Stub(9.0)},
    )
    assert preds["predicted_total_points"].tolist() == [9.0]
