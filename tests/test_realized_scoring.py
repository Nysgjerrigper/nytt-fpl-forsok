"""Realized scoring must respect starting slots, chips, and whole-round minutes."""
import pandas as pd
import pytest

from fpl.milp import optimize


def _lineup():
    positions = {"gk": "GK", "bgk": "GK"}
    positions.update({f"d{i}": "DEF" for i in range(4)})
    positions.update({f"m{i}": "MID" for i in range(6)})
    positions.update({f"f{i}": "FWD" for i in range(3)})
    return ["gk", "d0", "d1", "d2", "m0", "m1", "m2", "m3", "m4", "f0", "f1"], positions


def test_partial_xi_still_allows_a_legal_defender_replacement():
    lineup, positions = _lineup()
    minutes = {p: 90 for p in positions}
    minutes.update({"d0": 0, "d1": 0})
    actual = optimize.apply_auto_substitutions(lineup, ["m5", "d3", "f2"], minutes, positions)
    assert "d3" in actual and "m5" not in actual and "f2" not in actual
    assert len(actual) == 10
    assert sum(positions[p] == "DEF" for p in actual) == 2


def test_bench_order_and_goalkeeper_cover_are_preserved():
    lineup, positions = _lineup()
    minutes = {p: 90 for p in positions}
    minutes.update({"gk": 0, "f0": 0, "m5": 0})
    actual = optimize.apply_auto_substitutions(lineup, ["bgk", "m5", "d3", "f2"], minutes, positions)
    assert "bgk" in actual and "gk" not in actual
    assert "d3" in actual and "f2" not in actual


def test_captain_cameo_prevents_vice_activation():
    assert optimize.resolve_active_captain(["cap"], ["vice"], ["cap", "vice"],
                                           {"cap": 1, "vice": 90}) == "cap"
    assert optimize.resolve_active_captain(["cap"], ["vice"], ["vice"],
                                           {"cap": 0, "vice": 90}) == "vice"


def _predictions(tmp_path):
    # Exactly 15 players make the squad known; club limits remain feasible.
    roles = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    frame = pd.DataFrame([
        {"player_id": f"p{i}", "GW": 1, "name": f"p{i}", "position": role,
         "team": f"club{i % 5}", "value": 50,
         "predicted_total_points": i + 1, "actual_total_points": 0 if i == 14 else i + 1,
         "minutes": 0 if i == 14 else 90}
        for i, role in enumerate(roles)
    ])
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    return path, frame


@pytest.mark.parametrize("chip", ["bb", "tc"])
def test_chip_scoring_uses_actual_minutes_and_active_captain(tmp_path, chip):
    source, frame = _predictions(tmp_path)
    output = tmp_path / "squads.csv"
    result = optimize.run(optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "1",
        "--horizon", "1", "--scoring-mode", "corrected", f"--{chip}-gw", "1",
        "--output", str(output),
    ]))
    row = result.iloc[0]
    saved = pd.read_csv(output).iloc[0]
    actual = frame.set_index("player_id").actual_total_points
    assert row.captain == ["p14"]  # Chosen by forecast, but did not play.
    assert saved.actual_captain == row.vice_captain[0]
    assert saved.scoring_protocol == "autosubs_v2"
    bonus = actual.loc[row.vice_captain[0]] * (2 if chip == "tc" else 1)
    if chip == "bb":
        assert row.chip_played == "BB"
        assert saved.actual_total_points == actual.sum() + bonus
    else:
        assert row.chip_played.startswith("TC_")
        ids = saved.actual_lineup.split(", ")
        assert "p14" not in ids
        assert saved.actual_total_points == actual.loc[ids].sum() + bonus


@pytest.mark.parametrize("minutes", [None, float("nan"), -1, float("inf")])
def test_corrected_scoring_rejects_unknown_or_invalid_minutes(tmp_path, minutes):
    source, frame = _predictions(tmp_path)
    if minutes is None:
        frame = frame.drop(columns="minutes")
    else:
        frame["minutes"] = frame["minutes"].astype(float)
        frame.loc[0, "minutes"] = minutes
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="minutes"):
        optimize.run(optimize.parse_args([
            "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "1",
            "--scoring-mode", "corrected", "--output", str(tmp_path / "out.csv"),
        ]))


def test_explicit_legacy_scoring_keeps_original_captain_convention(tmp_path):
    source, frame = _predictions(tmp_path)
    output = tmp_path / "legacy.csv"
    result = optimize.run(optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "1",
        "--horizon", "1", "--scoring-mode", "legacy", "--output", str(output),
    ]))
    row = result.iloc[0]
    saved = pd.read_csv(output).iloc[0]
    actual = frame.set_index("player_id").actual_total_points
    assert saved.scoring_protocol == "legacy_no_autosubs"
    assert saved.actual_total_points == actual.loc[row.lineup].sum() + actual.loc[row.captain].sum()


def test_double_gameweek_minutes_prevent_false_captain_substitution(tmp_path):
    source, frame = _predictions(tmp_path)
    second_fixture = frame.iloc[[-1]].copy()
    second_fixture["predicted_total_points"] = 0
    second_fixture["actual_total_points"] = 5
    second_fixture["minutes"] = 45
    pd.concat([frame, second_fixture], ignore_index=True).to_csv(source, index=False)
    output = tmp_path / "double.csv"
    optimize.run(optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "1",
        "--horizon", "1", "--scoring-mode", "corrected", "--output", str(output),
    ]))
    row = pd.read_csv(output).iloc[0]
    assert row.actual_captain == "p14"
    assert row.actual_captain_points == 5


def test_free_hit_reports_and_scores_the_temporary_squad(tmp_path):
    source, frame = _predictions(tmp_path)
    frame["player_id"] = range(len(frame))
    extra = frame.iloc[[-1]].copy()
    extra["player_id"] = 15
    extra["name"] = "temporary_star"
    extra["predicted_total_points"] = 100
    extra["actual_total_points"] = 7
    extra["minutes"] = 90
    pd.concat([frame, extra], ignore_index=True).to_csv(source, index=False)
    output = tmp_path / "freehit.csv"
    result = optimize.run(optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "1",
        "--horizon", "1", "--scoring-mode", "corrected", "--fh-gw", "1",
        "--initial-squad", ",".join(map(str, frame.player_id)), "--initial-budget", "250", "--initial-ft", "0",
        "--output", str(output),
    ]))
    row = result.iloc[0]
    saved = pd.read_csv(output).iloc[0]
    assert row.chip_played == "FH"
    assert 15 in row.squad
    assert set(row.lineup) <= set(row.squad)
    assert "temporary_star" in saved.actual_lineup
    assert saved.actual_captain == "temporary_star"
