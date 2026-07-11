"""
Sanity check that the MILP optimizer's output always satisfies its own
constraints. A squad that violates budget/position/club rules isn't a valid
FPL team at all, so a violation here is a modelling bug, not an edge case to
shrug off - this test exists to catch that class of regression automatically
instead of relying on eyeballing a squad_selections CSV.

Uses a small synthetic predictions CSV (not real data) so the test is fast
and self-contained: 5 clubs, more players per position than the squad needs
(3 GK / 8 DEF / 8 MID / 5 FWD, max 3 per club) so the 2/5/5/3-with-max-3
constraints are actually binding, not trivially satisfied.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.milp import optimize

CLUBS = ["ARS", "MCI", "LIV", "CHE", "TOT"]

# (player_id, position, club index into CLUBS, value in raw 0.1m units - same scale as
# master_dataset.csv's "value" column and BS=1000.0 in optimize.py, i.e. NOT pounds-millions).
PLAYERS = (
    [(f"gk{i}", "GK", i % len(CLUBS), 45 + i) for i in range(3)]
    + [(f"def{i}", "DEF", i % len(CLUBS), 40 + i * 2) for i in range(8)]
    + [(f"mid{i}", "MID", i % len(CLUBS), 55 + i * 3) for i in range(8)]
    + [(f"fwd{i}", "FWD", i % len(CLUBS), 60 + i * 4) for i in range(5)]
)

POSITION_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _build_predictions_csv(tmp_path):
    rows = []
    for gw in (1, 2):
        for idx, (pid, pos, club_idx, value) in enumerate(PLAYERS):
            # Vary predicted points by GW so the optimizer isn't just re-picking an
            # identical squad twice - exercises the transfer-decision logic too.
            points = (idx * 7 + gw * 3) % 11
            rows.append({
                "player_id": pid,
                "GW": gw,
                "name": pid,
                "position": pos,
                "team": CLUBS[club_idx],
                "value": value,
                "predicted_total_points": points,
            })
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_optimizer_output_satisfies_all_constraints(tmp_path):
    predictions_csv = _build_predictions_csv(tmp_path)
    output_csv = tmp_path / "squad_selection.csv"

    args = optimize.parse_args([
        "--predictions-csv", str(predictions_csv),
        "--points-col", "predicted_total_points",
        "--start-gw", "1",
        "--max-gw", "2",
        "--horizon", "2",
        "--output", str(output_csv),
    ])
    results_df = optimize.run(args)

    position_by_id = {pid: pos for pid, pos, _, _ in PLAYERS}
    club_by_id = {pid: CLUBS[club_idx] for pid, _, club_idx, _ in PLAYERS}

    assert len(results_df) == 2
    for _, row in results_df.iterrows():
        squad = row["squad"]
        lineup = row["lineup"]
        captain = row["captain"]

        assert len(squad) == 15
        counts = {pos: 0 for pos in POSITION_COUNTS}
        for pid in squad:
            counts[position_by_id[pid]] += 1
        assert counts == POSITION_COUNTS

        club_counts = {}
        for pid in squad:
            club_counts[club_by_id[pid]] = club_counts.get(club_by_id[pid], 0) + 1
        assert all(count <= 3 for count in club_counts.values())

        assert len(lineup) == 11
        lineup_counts = {pos: 0 for pos in POSITION_COUNTS}
        for pid in lineup:
            lineup_counts[position_by_id[pid]] += 1
        assert lineup_counts["GK"] == 1
        assert lineup_counts["DEF"] >= 3
        assert lineup_counts["FWD"] >= 1
        assert set(lineup) <= set(squad)

        assert set(captain) <= set(lineup)

        # Same 0.1m-unit scale as the fixture's "value" column and BS=1000.0 in
        # optimize.py - NOT a pounds-millions (<=100.0) scale.
        assert row["budget_end"] <= 1000.0


def _build_origin_predictions_csv(tmp_path):
    """Origin-based CSV (audit B2): one forecast set per origin gameweek.

    Origin 1 covers GW1-2, origin 2 covers GW2. The two origins DISAGREE about GW2 on
    purpose: origin 1 thinks mid0 hauls and mid7 blanks; origin 2 (the set the GW2 solve
    must use) thinks the opposite. Which player the optimizer captains at GW2 therefore
    reveals which forecast set it consumed.
    """
    rows = []
    def add(origin, gw, pid, points):
        idx = [p[0] for p in PLAYERS].index(pid)
        _, pos, club_idx, value = PLAYERS[idx]
        rows.append({
            "player_id": pid, "GW": gw, "origin_gw": origin, "name": pid, "position": pos,
            "team": CLUBS[club_idx], "value": value,
            "predicted_total_points": points, "actual_total_points": points,
        })
    for pid, _, _, _ in PLAYERS:
        base = 2.0
        add(1, 1, pid, 50.0 if pid == "mid0" else base)
        add(1, 2, pid, 50.0 if pid == "mid0" else base)
        add(2, 2, pid, 50.0 if pid == "mid7" else (0.0 if pid == "mid0" else base))
    csv_path = tmp_path / "predictions_origin.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_origin_based_solve_uses_each_gameweeks_own_forecast_set(tmp_path):
    predictions_csv = _build_origin_predictions_csv(tmp_path)
    output_csv = tmp_path / "squad_selection_origin.csv"

    args = optimize.parse_args([
        "--predictions-csv", str(predictions_csv),
        "--points-col", "predicted_total_points",
        "--start-gw", "1",
        "--max-gw", "2",
        "--horizon", "2",
        "--output", str(output_csv),
    ])
    results_df = optimize.run(args)
    assert len(results_df) == 2
    by_gw = results_df.set_index("gameweek")

    # GW1 solve sees only origin 1, where mid0 is the star and mid7 is nothing.
    assert "mid0" in by_gw.loc[1, "squad"]
    assert by_gw.loc[1, "captain"] == ["mid0"]
    assert "mid7" not in by_gw.loc[1, "squad"]

    # GW2 solve must switch to origin 2's set: mid7 is now the star, mid0 worthless.
    # If the old single-matrix path were still in effect, GW2 would reuse origin 1's
    # forecasts and keep captaining mid0.
    assert "mid7" in by_gw.loc[2, "squad"]
    assert by_gw.loc[2, "captain"] == ["mid7"]

    # Structural sanity on both rows (full constraint sweep lives in the test above).
    for gw in (1, 2):
        assert len(by_gw.loc[gw, "squad"]) == 15
        assert len(by_gw.loc[gw, "lineup"]) == 11
