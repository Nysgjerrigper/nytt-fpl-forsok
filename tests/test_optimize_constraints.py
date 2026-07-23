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


def test_initial_ft_above_policy_cap_is_honored(tmp_path):
    """A real squad can arrive with more banked FTs than the solver's banking policy
    (config.MILP_MAX_FREE_TRANSFERS=2 vs the site's cap of 5, TODO 3.4): the run must
    be feasible and start from the full stack, not clamp it or die infeasible."""
    # Same fixture as _build_predictions_csv but with integer player_ids, since
    # --initial-squad round-trips ids through int().
    int_id = {pid: i + 1 for i, (pid, _, _, _) in enumerate(PLAYERS)}
    rows = []
    for gw in (1, 2):
        for idx, (pid, pos, club_idx, value) in enumerate(PLAYERS):
            rows.append({"player_id": int_id[pid], "GW": gw, "name": pid, "position": pos,
                         "team": CLUBS[club_idx], "value": value,
                         "predicted_total_points": (idx * 7 + gw * 3) % 11})
    predictions_csv = tmp_path / "predictions_int_ids.csv"
    pd.DataFrame(rows).to_csv(predictions_csv, index=False)
    # A valid 15: 2 GK, 5 DEF, 5 MID, 3 FWD (PLAYERS is grouped by position).
    by_pos = {}
    for pid, pos, _, _ in PLAYERS:
        by_pos.setdefault(pos, []).append(int_id[pid])
    squad_ids = by_pos["GK"][:2] + by_pos["DEF"][:5] + by_pos["MID"][:5] + by_pos["FWD"][:3]

    args = optimize.parse_args([
        "--predictions-csv", str(predictions_csv),
        "--points-col", "predicted_total_points",
        "--start-gw", "1",
        "--max-gw", "2",
        "--horizon", "2",
        "--initial-squad", ",".join(str(pid) for pid in squad_ids),
        "--initial-budget", "0",
        "--initial-ft", "5",
        "--output", str(tmp_path / "squad_selection.csv"),
    ])
    results_df = optimize.run(args)
    assert not results_df.empty
    assert results_df.iloc[0]["q_start"] == 5  # full banked stack available at GW1


def test_free_transfer_rollover_banks_and_caps(tmp_path):
    """TODO 3.5: pin the plain-Python FT rollover in optimize.py against the solver's
    own transfer/hit variables.

    optimize.py re-solves a rolling horizon each GW but only locks in the first GW's
    decision; the FT balance for the NEXT GW is then advanced outside the solver
    (previous_ft -> next_ft, around optimize.py's `ft_used_eff`/`ft_carry_val`/`next_ft`
    block) rather than being read off a solver variable. That duplicated bookkeeping can
    silently diverge from the constraints the solver actually enforced, and nothing in
    the existing suite exercises more than one GW of it.

    Fixture: unlike the module-level PLAYERS pool (which deliberately has MORE players
    than the squad needs, so transfer alternatives exist), this test uses a pool of
    EXACTLY 15 players - 2 GK, 5 DEF, 5 MID, 3 FWD, 3 per club across the 5 CLUBS so the
    max-3-per-club constraint is satisfied but not slack. With no eligible substitute at
    any position, transferring is structurally impossible (there is no other player to
    buy), not merely unattractive - so 0 transfers is forced regardless of predicted
    points or solver tie-breaking, giving a fully deterministic FT trajectory.

    With config.MILP_MAX_FREE_TRANSFERS=2 (Q_bar) and config.MILP_FT_PER_GW=1
    (Q_under_bar), a fresh build starts at q_start=1 (default --initial-ft):
      GW1: q_start=1, 0 transfers -> carry=max(0,1-0)=1 -> next_ft=min(2, 1+1)=2
      GW2: q_start=2, 0 transfers -> carry=max(0,2-0)=2 -> next_ft=min(2, 2+1)=2 (capped)
      GW3: q_start=2 (cap holds)
    i.e. the q_start sequence across GW1-3 must be [1, 2, 2], pinning both the
    "bank an unused FT up to the Q_bar=2 cap" behaviour and the "cap never exceeds
    Q_bar" behaviour.
    """
    pool = (
        [(f"rtgk{i}", "GK", 45 + i) for i in range(2)]
        + [(f"rtdef{i}", "DEF", 40 + i * 2) for i in range(5)]
        + [(f"rtmid{i}", "MID", 55 + i * 3) for i in range(5)]
        + [(f"rtfwd{i}", "FWD", 60 + i * 4) for i in range(3)]
    )
    assert len(pool) == 15
    rows = []
    for gw in (1, 2, 3):
        for idx, (pid, pos, value) in enumerate(pool):
            club_idx = idx % len(CLUBS)  # 15 players over 5 clubs -> exactly 3 per club
            rows.append({
                "player_id": pid,
                "GW": gw,
                "name": pid,
                "position": pos,
                "team": CLUBS[club_idx],
                "value": value,
                # Points are identical across players and GWs - irrelevant here since
                # the pool leaves no room to transfer at all, but kept simple/uniform
                # rather than reusing a formula meant to force particular rankings.
                "predicted_total_points": 10,
            })
    predictions_csv = tmp_path / "predictions_ft_rollover.csv"
    pd.DataFrame(rows).to_csv(predictions_csv, index=False)

    args = optimize.parse_args([
        "--predictions-csv", str(predictions_csv),
        "--points-col", "predicted_total_points",
        "--start-gw", "1",
        "--max-gw", "3",
        "--horizon", "2",
        "--output", str(tmp_path / "squad_selection_ft_rollover.csv"),
    ])
    results_df = optimize.run(args)
    assert len(results_df) == 3

    by_gw = results_df.set_index("gameweek")
    # No incentive to transfer at any GW given identical, GW-invariant predicted points.
    assert by_gw.loc[1, "transfers_in"] == []
    assert by_gw.loc[2, "transfers_in"] == []
    assert by_gw.loc[3, "transfers_in"] == []

    assert [by_gw.loc[gw, "q_start"] for gw in (1, 2, 3)] == [1, 2, 2]
