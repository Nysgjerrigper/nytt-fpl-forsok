"""Regression tests for the richer predictor groups added to fpl.features.

These use tiny hand-built frames rather than the real dataset so the expected values
can be reasoned about by hand - the point is to pin the leakage discipline and the
per-90 / opponent-merge arithmetic, not to check accuracy on real data.
"""
import numpy as np
import pandas as pd
import pytest

from fpl import features


def _base_cols(df):
    """Fill in the columns build_feature_frame touches but a given test doesn't care about,
    so we can keep each synthetic frame minimal and readable."""
    df = df.copy()
    for col in features.FORM_STATS + ["xP"]:
        if col not in df.columns:
            df[col] = 0.0
    if "was_home" not in df.columns:
        df["was_home"] = 1
    if "team" not in df.columns:
        df["team"] = "T"
    if "opponent_team" not in df.columns:
        df["opponent_team"] = "OPP"
    return df


def test_per90_handles_zero_minutes():
    """A player with 0 minutes across the whole rolling window must get a 0.0 rate, not NaN/inf."""
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 1],
        "GW_global": [1, 2, 3],
        "minutes": [0.0, 0.0, 0.0],
        "goals_scored": [0.0, 0.0, 0.0],
    }))
    out = features.build_feature_frame(df)
    per90 = out["goals_scored_per90_roll5"]
    assert per90.notna().all()
    assert np.isfinite(per90.to_numpy()).all()
    assert (per90 == 0.0).all()


def test_per90_computes_expected_rate():
    """Sanity-check the rate arithmetic on non-zero minutes: shifted rolling(5) sums, then *90.

    GW3's feature sees only GW1+GW2 (shift-by-one): 1 goal in 135 minutes -> 1/135*90 = 0.6667.
    """
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 1],
        "GW_global": [1, 2, 3],
        "minutes": [90.0, 45.0, 90.0],
        "goals_scored": [1.0, 0.0, 5.0],  # GW3's own 5 goals must NOT enter its feature
    }))
    out = features.build_feature_frame(df).sort_values("GW_global")
    gw3 = out.loc[out["GW_global"] == 3, "goals_scored_per90_roll5"].iloc[0]
    assert gw3 == pytest.approx(90.0 / 135.0)  # 1 goal over 135 mins, current GW excluded


def test_ewma_and_rolling_are_leakage_free():
    """Altering GW t's own total_points must leave every shifted feature at GW t unchanged.

    This is the core leakage guard: features at row t are built from strictly-earlier rows only.
    """
    frame = pd.DataFrame({
        "player_id": [1, 1, 1, 1],
        "GW_global": [1, 2, 3, 4],
        "minutes": [90.0, 90.0, 90.0, 90.0],
        "total_points": [2.0, 5.0, 8.0, 3.0],
        "goals_scored": [0.0, 1.0, 0.0, 1.0],
    })
    base = features.build_feature_frame(_base_cols(frame))

    tampered = frame.copy()
    tampered.loc[tampered["GW_global"] == 3, "total_points"] = 999.0  # explode GW3's own outcome
    after = features.build_feature_frame(_base_cols(tampered))

    checked = ["total_points_ewm3", "total_points_roll3", "total_points_roll5",
               "total_points_prev", "total_points_season_avg"]
    b = base.loc[base["GW_global"] == 3, checked].reset_index(drop=True)
    a = after.loc[after["GW_global"] == 3, checked].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)

    # And the change MUST show up from GW4 onward, proving the features aren't just frozen.
    b4 = base.loc[base["GW_global"] == 4, "total_points_ewm3"].iloc[0]
    a4 = after.loc[after["GW_global"] == 4, "total_points_ewm3"].iloc[0]
    assert a4 != b4


def test_opponent_strength_uses_prior_form_only():
    """The opp_* merge must attach the OPPONENT's trailing form and exclude the current GW.

    Team B concedes 3 then 1 then 0 over GW1-3. When team A plays B in GW3, A's players should
    see B's clean-sheet rate / defense computed from GW1-2 ONLY (rolling(6) mean of goals conceded
    = (3+1)/2 = 2.0, clean-sheet rate = 0.0), never GW3's own 0 conceded.
    """
    rows = []
    # Team B: one defender per GW carrying the team's goals_conceded, plus a scorer for goals_scored.
    b_conceded = {1: 3.0, 2: 1.0, 3: 0.0}
    b_scored = {1: 0.0, 2: 2.0, 3: 4.0}
    for gw in (1, 2, 3):
        rows.append({"player_id": 100 + gw, "GW_global": gw, "team": "B", "opponent_team": "X",
                     "goals_conceded": b_conceded[gw], "goals_scored": b_scored[gw], "minutes": 90.0})
    # Team A plays B in GW3 - this is the row whose opp_* features we inspect.
    rows.append({"player_id": 200, "GW_global": 3, "team": "A", "opponent_team": "B",
                 "goals_conceded": 0.0, "goals_scored": 1.0, "minutes": 90.0})

    df = _base_cols(pd.DataFrame(rows))
    out = features.build_feature_frame(df)
    a_row = out[(out["player_id"] == 200) & (out["GW_global"] == 3)].iloc[0]

    # defense = mean of B's conceded over GW1-2 = (3+1)/2 = 2.0; GW3's 0 excluded by the shift.
    assert a_row["opp_defense_roll6"] == 2.0
    # clean-sheet rate over GW1-2 = 0 (B conceded in both); GW3's clean sheet excluded.
    assert a_row["opp_cs_rate_roll6"] == 0.0
    # attack = mean of B's goals scored over GW1-2 = (0+2)/2 = 1.0; GW3's 4 excluded.
    assert a_row["opp_attack_roll6"] == 1.0


def test_opponent_strength_conceded_is_team_max():
    """goals_conceded is a per-player (minutes-dependent) value, so the team's true concession that
    GW is the max across its rows - verify the aggregation picks that up before rolling/merging.

    B must field a GW2 row of its own (against X) for the (opponent=B, GW2) merge key to exist; its
    trailing form there reflects GW1 only (shifted), which is what team A sees when it meets B in GW2.
    """
    rows = [
        # Team B, GW1: two players report different conceded (one subbed early); max=2 is the truth.
        {"player_id": 301, "GW_global": 1, "team": "B", "opponent_team": "X",
         "goals_conceded": 0.0, "goals_scored": 0.0, "minutes": 30.0},
        {"player_id": 302, "GW_global": 1, "team": "B", "opponent_team": "X",
         "goals_conceded": 2.0, "goals_scored": 0.0, "minutes": 90.0},
        # Team B also plays in GW2 (vs X) so it has a GW2 row to merge onto; its GW2 form is GW1-only.
        {"player_id": 303, "GW_global": 2, "team": "B", "opponent_team": "X",
         "goals_conceded": 5.0, "goals_scored": 0.0, "minutes": 90.0},
        # Team A meets B in GW2, so it sees B's GW1 form only.
        {"player_id": 400, "GW_global": 2, "team": "A", "opponent_team": "B",
         "goals_conceded": 0.0, "goals_scored": 0.0, "minutes": 90.0},
    ]
    df = _base_cols(pd.DataFrame(rows))
    out = features.build_feature_frame(df)
    a_row = out[(out["player_id"] == 400) & (out["GW_global"] == 2)].iloc[0]
    assert a_row["opp_defense_roll6"] == 2.0  # max(0,2), not mean or first
    assert a_row["opp_cs_rate_roll6"] == 0.0  # conceded 2 -> not a clean sheet


def test_xp_features_are_shifted():
    """xP_prev/xP_roll3 must be lagged: GW t's own xP never enters GW t's feature."""
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 1],
        "GW_global": [1, 2, 3],
        "xP": [4.0, 6.0, 100.0],  # GW3's own 100 must not appear at GW3
        "minutes": [90.0, 90.0, 90.0],
    }))
    out = features.build_feature_frame(df).sort_values("GW_global")
    gw3 = out[out["GW_global"] == 3].iloc[0]
    assert gw3["xP_prev"] == 6.0                 # previous GW's xP
    assert gw3["xP_roll3"] == (4.0 + 6.0) / 2.0  # mean of GW1-2, current excluded


def test_all_new_columns_registered_in_feature_columns():
    """Every new column must be picked up by feature_columns() or downstream models never see it."""
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "GW_global": [1, 2, 1, 2],
        "team": ["A", "A", "B", "B"],
        "opponent_team": ["B", "B", "A", "A"],
        "xP": [3.0, 4.0, 2.0, 5.0],
        "minutes": [90.0, 90.0, 90.0, 90.0],
        "goals_scored": [1.0, 0.0, 0.0, 2.0],
        "goals_conceded": [0.0, 2.0, 1.0, 0.0],
        "total_points": [5.0, 2.0, 1.0, 8.0],
    }))
    out = features.build_feature_frame(df)
    cols = set(features.feature_columns(out))

    expected = (features.EWMA_FEATURES + features.PER90_FEATURES
                + features.OPPONENT_FEATURES + features.XP_FEATURES)
    missing = [c for c in expected if c not in cols]
    assert not missing, f"new feature columns not registered: {missing}"
    # And they must actually exist as columns on the built frame, not just in the name lists.
    assert all(c in out.columns for c in expected)


def test_dgw_second_fixture_sees_no_same_round_information():
    """Double-gameweek leakage guard (audit A3): in a DGW a player has two rows with the
    same GW_global, and per-row shift(1) used to let the second fixture's features include
    the first fixture of the SAME round. After the GW-level shift, tampering with the first
    fixture's outcome must leave BOTH GW3 rows' features unchanged - the round is predicted
    entirely from strictly-earlier gameweeks, like a real deadline."""
    frame = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 1],
        "GW_global": [1, 2, 3, 3, 4],  # GW3 is a double gameweek
        "minutes": [90.0] * 5,
        "total_points": [2.0, 4.0, 10.0, 6.0, 5.0],
    })
    base = features.build_feature_frame(_base_cols(frame))

    tampered = frame.copy()
    tampered.loc[2, "total_points"] = 999.0  # first fixture of the DGW
    after = features.build_feature_frame(_base_cols(tampered))

    checked = ["total_points_prev", "total_points_roll3", "total_points_roll5",
               "total_points_ewm3", "total_points_season_avg", "total_points_career_avg",
               "games_played_so_far"]
    b3 = base.loc[base["GW_global"] == 3, checked].reset_index(drop=True)
    a3 = after.loc[after["GW_global"] == 3, checked].reset_index(drop=True)
    pd.testing.assert_frame_equal(a3, b3)

    # Both fixtures of the round carry identical deadline-time form...
    pd.testing.assert_series_equal(b3.iloc[0], b3.iloc[1], check_names=False)
    # ...built from GW1-2 only: prev = GW2's 4.0, roll3 = mean(2, 4) = 3.0.
    assert b3.loc[0, "total_points_prev"] == 4.0
    assert b3.loc[0, "total_points_roll3"] == pytest.approx(3.0)

    # And GW4 must still see the (tampered) DGW - features move forward, not frozen.
    b4 = base.loc[base["GW_global"] == 4, "total_points_roll3"].iloc[0]
    a4 = after.loc[after["GW_global"] == 4, "total_points_roll3"].iloc[0]
    assert a4 != b4
    assert b4 == pytest.approx((4.0 + 10.0 + 6.0) / 3.0)  # GW2 + both GW3 fixtures


def test_dgw_keeps_per_fixture_known_ahead_columns():
    """The GW-level shift must only touch player-form features: the two fixtures of a DGW
    legitimately differ in opponent/home-away (known before the round), and each keeps its
    own outcome as the target."""
    frame = pd.DataFrame({
        "player_id": [1, 1, 1],
        "GW_global": [1, 2, 2],
        "minutes": [90.0] * 3,
        "total_points": [2.0, 8.0, 1.0],
        "was_home": [1, 1, 0],
        "opponent_team": ["OPP", "OPP_A", "OPP_B"],
    })
    out = features.build_feature_frame(_base_cols(frame))
    gw2 = out[out["GW_global"] == 2].sort_values("was_home", ascending=False)
    assert list(gw2["opponent_team"]) == ["OPP_A", "OPP_B"]
    assert list(gw2["was_home"]) == [1, 0]
    assert set(gw2["total_points"]) == {8.0, 1.0}  # targets stay per-fixture


def test_no_current_gw_xp_column_is_ever_built():
    """The same-GW raw xP is a confirmed post-match leak (vaastav scrapes ep_this after the
    round - RESEARCH_LOG 2026-07-16). Nothing unshifted may reach the feature list."""
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 1],
        "GW_global": [1, 2, 3],
        "xP": [4.0, 6.0, 2.5],
        "minutes": [90.0, 90.0, 90.0],
    }))
    out = features.build_feature_frame(df)
    assert "xP_current" not in out.columns
    assert "xP" not in features.feature_columns(out)


def test_xp_zero_filled_rounds_are_masked():
    """A round where EVERY row's xP is exactly 0 is an unfilled dump round, not a forecast of 0:
    it must stay out of the lagged forms (which previously averaged the fake zeros in)."""
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 2],
        "GW_global": [1, 2, 3, 1, 2, 3],
        "xP": [4.0, 0.0, 6.0, 2.0, 0.0, 3.0],  # GW2 all-zero -> unfilled round
        "minutes": [90.0] * 6,
    }))
    out = features.build_feature_frame(df)
    p1 = out[out["player_id"] == 1].sort_values("GW_global")
    assert np.isnan(p1["xP_prev"].iloc[2])                 # prev GW was unfilled -> NaN
    assert p1["xP_roll3"].iloc[2] == 4.0                   # mean skips the masked round


def test_xp_genuine_zero_in_populated_round_is_kept():
    """A single player's xP of 0 in a round where others have real xP is a genuine value and
    must survive into the next GW's lagged features."""
    df = _base_cols(pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "GW_global": [1, 2, 1, 2],
        "xP": [0.0, 1.0, 5.0, 4.0],
        "minutes": [0.0, 0.0, 90.0, 90.0],
    }))
    out = features.build_feature_frame(df)
    p1 = out[out["player_id"] == 1].sort_values("GW_global")
    assert p1["xP_prev"].iloc[1] == 0.0                    # genuine 0, not masked
