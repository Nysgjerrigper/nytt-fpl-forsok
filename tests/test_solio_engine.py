"""Solve actual pinned compatibility MILPs, independently validating modern state."""
import pandas as pd
import pytest

from fpl.milp.solio import chip_options, run_schedule, score_decisions, solve_window

ROLES = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


def forecasts(start=1, end=3, upgrade_from=99):
    return pd.DataFrame([{"player_id": p+1, "GW": gw, "name": f"p{p+1}",
        "position": ROLES[p % 15], "team": f"club{p % 5}", "value": 50,
        "predicted_total_points": 40 if p >= 15 and gw >= upgrade_from else (0 if p >= 15 else 2),
        "actual_total_points": 1, "minutes": 90}
        for gw in range(start, end+1) for p in range(30)])


def continuing(frame, start=1, horizon=3, **kwargs):
    return solve_window(frame, season="2020-21", start_gw=start, horizon=horizon,
                        initial_squad=list(range(1,16)), bank=250, **kwargs)


def test_fresh_solver_and_five_transfer_hold():
    fresh = solve_window(forecasts(), season="2020-21", start_gw=1, horizon=3)
    assert len(fresh) == 3 and fresh.iloc[0].q_start == 0
    assert fresh.iloc[0].budget_end == 250
    held = continuing(forecasts(), free_transfers=5)
    assert held.q_start.tolist() == [5,5,5] and held.q_end.tolist() == [5,5,5]
    # Reserve-GK weight is zero, so equally valuable bench swaps may tie.
    assert all(len(ids) <= 1 for ids in held.transfers_in)
    assert held.alpha.eq(0).all()
    assert held.solver_status.eq("Optimal").all() and held.mip_gap.eq(0).all()


def test_wildcard_preserves_banked_transfers_and_forces_many_moves():
    result = continuing(forecasts(upgrade_from=2), free_transfers=4,
                        chip_inventory={"wc": {1:1}}, forced_chips={"wc":[2]})
    row = result.iloc[1]
    assert row.chip_played == "WC" and len(row.transfers_in) >= 10
    assert row.alpha == 0 and row.q_end == row.q_start


def test_free_hit_restores_squad_bank_and_banked_transfers():
    frame = forecasts(2,4)
    frame.loc[(frame.GW == 3) & (frame.player_id > 15), "predicted_total_points"] = 50
    result = continuing(frame, start=2, horizon=3, free_transfers=4,
                        chip_inventory={"fh": {1:1}}, forced_chips={"fh":[3]})
    row = result.iloc[1]
    assert row.chip_played == "FH" and len(set(row.squad)-set(range(1,16))) >= 10
    assert row.budget_start == row.budget_end and not row.transfers_in
    after = result.iloc[2]
    assert set(after.squad) == (set(result.iloc[0].squad)-set(after.transfers_out)) | set(after.transfers_in)
    assert result.iloc[2].q_start == row.q_start


def test_both_half_chip_sets_are_distinct_and_bb_keeps_legal_xi():
    result = continuing(forecasts(19,20), start=19, horizon=2, free_transfers=5,
                        chip_inventory={"bb":{1:1,2:1}}, forced_chips={"bb":[19,20]})
    assert result.chip_played.tolist() == ["BB","BB"]
    assert all(len(xi)==11 for xi in result.lineup)
    assert all(len(bench)==4 for bench in result.bench_order)


def test_tiny_time_limit_fails_closed():
    with pytest.raises(RuntimeError, match="solve rejected"):
        continuing(forecasts(), time_limit=1e-9)


@pytest.mark.parametrize("inventory,forced", [({"bb":{1:1}}, {"bb":[19,20]}),
    ({"bb":{1:1}}, {"bb":[18,19]}), ({"fh":{1:1,2:1}}, {"fh":[19,20]}),
    ({"wc":{1:1},"bb":{1:1}}, {"wc":[19],"bb":[19]}),
    ({"bb":{1:1.0}}, {})])
def test_bad_chip_schedules_fail_before_solving(inventory, forced):
    with pytest.raises(ValueError):
        chip_options(18,3,inventory,forced)


def test_rolling_state_and_round_scoring():
    result = run_schedule(forecasts(), season="2020-21", start_gw=1,max_gw=3,horizon=2)
    assert result.actual_total_points.tolist() == [12,12,12]
    assert result.actual_net_points.tolist() == [12,12,12]
    assert result.q_start.tolist() == [0,1,2]


def test_origin_dgw_realized_rows_sum_once_across_origins():
    decision = solve_window(forecasts(),season="2020-21",start_gw=1,horizon=1).iloc[:1]
    rows = forecasts(1,1)
    panel = pd.concat([rows.assign(origin_gw=origin, fixture=fixture) for origin in (1,2) for fixture in (1,2)], ignore_index=True)
    scored = score_decisions(decision,panel)
    assert scored.actual_total_points.iloc[0] == 24
    panel.loc[0,"actual_total_points"] = 9
    with pytest.raises(ValueError,match="disagree across"):
        score_decisions(decision,panel)


@pytest.mark.parametrize("previous", [True, 1.5, 4, 0, "2"])
def test_previous_fh_state_is_validated_for_direct_calls(previous):
    with pytest.raises(ValueError, match="previous Free Hit"):
        chip_options(4, 1, None, None, previous)


def test_wholly_missing_round_is_not_silently_assumed_blank():
    with pytest.raises(ValueError, match="every requested round"):
        continuing(forecasts(1,1), horizon=2)


def test_mismatched_bootstrap_season_is_rejected_before_inputs():
    with pytest.raises(ValueError, match="bootstrap active season"):
        solve_window(forecasts(1,1), season="2020-21", start_gw=1, horizon=1,
                     bootstrap={"events":[{"id":1,"deadline_time":"2026-08-21T18:30:00Z"}]})


def test_public_window_cannot_admit_two_origins():
    rows = forecasts(1,1)
    panel = pd.concat([rows.assign(origin_gw=origin) for origin in (1,2)])
    with pytest.raises(ValueError, match="origin_gw"):
        solve_window(panel,season="2020-21",start_gw=1,horizon=1)
