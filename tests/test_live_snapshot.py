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
