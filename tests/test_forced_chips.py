"""Explicit chip schedules obey budget/state and transfer-hit contracts."""
import pandas as pd
import pytest

from fpl.milp import optimize

ROLES = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


def _source(tmp_path, start=1, end=3, upgrade_from=1, expensive=False, negative=False):
    rows = []
    for gw in range(start, end + 1):
        for i in range(30):
            upgrade = i >= 15
            rows.append({"player_id": i + 1, "name": f"p{i+1}", "position": ROLES[i % 15],
                         "team": f"club{i % 5}", "GW": gw,
                         "value": 350 if upgrade and expensive else 50,
                         "predicted_total_points": (-1. if negative else
                             (40. if upgrade and gw >= upgrade_from else (0. if upgrade else 2.))),
                         "actual_total_points": 1., "minutes": 90})
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(tmp_path, chip, target, *, start=1, end=3, ft=0, **data):
    source = _source(tmp_path, start, end, upgrade_from=target, **data)
    args = optimize.parse_args(["--predictions-csv", str(source), "--start-gw", str(start),
        "--max-gw", str(end), "--horizon", "2", "--initial-squad",
        ",".join(map(str, range(1, 16))), "--initial-budget", "250",
        "--initial-ft", str(ft), f"--{chip}-gw", str(target),
        "--scoring-mode", "corrected", "--output", str(tmp_path / "output.csv")])
    return optimize.run(args)


@pytest.mark.parametrize("chip,start,target", [("wc1", 1, 1), ("wc1", 1, 2),
                                              ("wc2", 20, 20), ("wc2", 20, 21)])
def test_wildcards_are_forced_in_both_halves_and_horizon_periods(tmp_path, chip, start, target):
    result = _run(tmp_path, chip, target, start=start, end=start+2)
    assert len(result) == 3
    row = result[result.gameweek == target].iloc[0]
    assert row.chip_played == "WC"
    assert row.alpha == 0
    assert len(row.transfers_in) >= 10
    assert len(row.transfers_in) > row.q_start
    assert result[result.gameweek == target+1].iloc[0].q_start == 1
    assert sum(result.chip_played == "WC") == 1


@pytest.mark.parametrize("chip,start", [("wc1", 1), ("wc2", 20), ("fh", 1)])
def test_chip_with_maximum_banked_transfers_and_no_changes_is_feasible(tmp_path, chip, start):
    # No upgrades are affordable, so the legal squad is unchanged.
    result = _run(tmp_path, chip, start, start=start, end=start+1, ft=2, expensive=True)
    assert len(result) == 2
    first, second = result.iloc[0], result.iloc[1]
    assert first.chip_played == ("FH" if chip == "fh" else "WC")
    assert first.transfers_in == first.transfers_out == []
    assert first.alpha == 0
    assert second.q_start == 1


@pytest.mark.parametrize("target", [1, 2])
def test_free_hit_budget_and_permanent_state_are_preserved(tmp_path, target):
    result = _run(tmp_path, "fh", target, ft=1, expensive=True)
    assert len(result) == 3
    row = result[result.gameweek == target].iloc[0]
    assert row.chip_played == "FH"
    assert row.transfers_in == row.transfers_out == []
    assert row.alpha == 0
    prices = {i: 50 if i <= 15 else 350 for i in range(1, 31)}
    assert sum(prices[i] for i in row.squad) <= 1000
    following = result[result.gameweek == target+1].iloc[0]
    assert following.budget_start == row.budget_start
    assert following.q_start == 1
    prior = set(range(1, 16)) if target == 1 else set(result[result.gameweek == target-1].iloc[0].squad)
    restored = (set(following.squad) - set(following.transfers_in)) | set(following.transfers_out)
    assert restored == prior


@pytest.mark.parametrize("chip", ["bb", "tc"])
def test_forced_scoring_chips_cannot_be_skipped_for_negative_forecasts(tmp_path, chip):
    result = _run(tmp_path, chip, 1, end=1, negative=True)
    assert len(result) == 1
    assert result.iloc[0].chip_played == "BB" if chip == "bb" else result.iloc[0].chip_played.startswith("TC_")


def test_disabled_chips_preserve_ordinary_transfer_penalty(tmp_path):
    result = _run(tmp_path, "wc1", 0, end=1)
    row = result.iloc[0]
    assert pd.isna(row.chip_played)
    assert len(row.transfers_in) >= 10
    assert row.alpha == len(row.transfers_in) - row.q_start
    # Constant forecasts make the objective decomposable; all hit deductions remain present.
    assert row.alpha > 0


@pytest.mark.parametrize("arguments,message", [
    (["--wc1-gw", "-1"], "nonnegative"),
    (["--wc1-gw", "1", "--bb-gw", "1"], "one chip"),
    (["--wc2-gw", "1"], "wrong season half"),
    (["--wc1-gw", "20"], "wrong season half"),
    (["--wc1-gw", "39"], "starting season"),
    (["--tc-gw", "51"], "run range"),
])
def test_invalid_chip_targets_fail_before_solver_or_csv(monkeypatch, arguments, message):
    monkeypatch.setattr(optimize, "make_solver", lambda *a, **k: pytest.fail("invalid target reached solver"))
    args = optimize.parse_args(["--start-gw", "1", "--max-gw", "50", *arguments])
    with pytest.raises(ValueError, match=message):
        optimize.run(args)
