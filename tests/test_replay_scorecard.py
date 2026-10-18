"""Settlement evidence summaries use saved bytes, never optimizer claims as proof."""
import hashlib
import json

import pandas as pd
import pytest

from fpl.milp.replay import run_replay
from fpl.replay_scorecard import summarize_replay_pair


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(tmp_path, name, *, alpha=0, chip=None, timed=True, points=1., rounds=1):
    root = tmp_path / name
    root.mkdir()
    row = {"gameweek": 141, "squad": list(range(1, 16)),
           "lineup": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
           "bench_order": [2, 6, 7, 12], "captain": [8], "vice_captain": [9],
           "alpha": alpha, "chip_played": chip, "transfers_in": [18, 19],
           "transfers_out": [20, 21], "solver_status": "Optimal", "mip_gap": 0,
           "price_policy": "raw-price-claim", "pool_policy": "raw-pool-claim",
           "solver_outcome_columns": "ABSENT"}
    if timed:
        row["solver_seconds"] = 1.25
    state = {"schema_version": 1, "settlement_completed": False, "decisions": [row]}
    source = root / "source.json"
    source.write_text(json.dumps(state))
    frame = pd.DataFrame({"player_id": range(1, 16), "GW": 141, "fixture": 1001,
                          "position": ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                          "team": [f"club{p % 5}" for p in range(15)],
                          "actual_total_points": points, "minutes": 90.})
    forecasts = root / "source.csv"
    if rounds == 2:
        second = dict(row, gameweek=142)
        second.pop("solver_seconds", None)
        state["decisions"].append(second)
        source.write_text(json.dumps(state))
        frame = pd.concat([frame, frame.assign(GW=142, fixture=1002)], ignore_index=True)
    frame.to_csv(forecasts, index=False)
    return run_replay(source, forecasts, root / "replay")


@pytest.fixture
def pair(tmp_path):
    return bundle(tmp_path, "candidate", alpha=1, chip="TC"), bundle(tmp_path, "control", timed=False)


def summary(pair, **kwargs):
    candidate, control = pair
    params = {"candidate_sha256": digest(candidate / "replayed.csv"),
              "control_sha256": digest(control / "replayed.csv"),
              "gameweeks": [141], "metric": "actual_net_points"}
    params.update(kwargs)
    return summarize_replay_pair(candidate, control, **params)


def repin(root, name, content):
    """Keep archive hashing valid so independent semantic checks are exercised."""
    path = root / name
    path.write_text(content)
    receipt = json.loads((root / "complete.json").read_text())
    receipt["output_sha256"][name] = digest(path)
    if name in receipt["input_sha256"]:
        receipt["input_sha256"][name] = digest(path)
    (root / "complete.json").write_text(json.dumps(receipt))


def test_known_settlement_and_recorded_claims_never_upgrade_gates(pair):
    result = summary(pair)
    candidate = result["sides"]["candidate"]
    assert (candidate["gross_points"], candidate["net_points"], candidate["hit_points"]) == (13, 9, 4)
    assert candidate["chip_usage"] == {"TC": 1}
    assert candidate["decision_rounds"] == 1
    assert candidate["transfers_in"] == candidate["transfers_out"] == 2
    assert candidate["receipt_sha256"] == digest(pair[0] / "complete.json")
    assert candidate["snapshot_sha256"] == digest(pair[0] / "decisions.json")
    assert candidate["prediction_sha256"] == digest(pair[0] / "predictions.csv")
    assert candidate["recorded"]["solver_status"] == {"status": "RECORDED", "values": ["Optimal"]}
    assert candidate["recorded"]["mip_gap"]["values"] == [0]
    assert candidate["recorded"]["price_policy"]["status"] == "RECORDED"
    assert candidate["recorded"]["pool_policy"]["decisions"] == ["raw-pool-claim"]
    assert candidate["recorded"]["solver_outcome_columns"]["decisions"] == ["ABSENT"]
    assert result["sides"]["control"]["recorded"]["solver_seconds"] == {"status": "NOT MEASURED"}
    assert set(result["checks"].values()) == {"NOT ASSESSED"}
    assert result["promotion_eligible"] is False
    assert result["status"] == "VERIFIED_SETTLEMENT_INPUTS"
    assert "blank planning" in result["limitations"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("file", ["decisions.json", "replayed.csv", "predictions.csv"])
def test_changed_bytes_fail(pair, file):
    with (pair[0] / file).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError):
        summary(pair)


@pytest.mark.parametrize("where", ["receipt", "csv", "snapshot"])
def test_exact_coverage_even_when_hashes_rebound(pair, where):
    root = pair[0]
    if where == "receipt":
        receipt = json.loads((root / "complete.json").read_text())
        receipt["gameweeks"] = [142]
        (root / "complete.json").write_text(json.dumps(receipt))
    elif where == "csv":
        frame = pd.read_csv(root / "replayed.csv").assign(gameweek=142)
        repin(root, "replayed.csv", frame.to_csv(index=False))
    else:
        state = json.loads((root / "decisions.json").read_text())
        state["decisions"][0]["gameweek"] = 142
        repin(root, "decisions.json", json.dumps(state))
    with pytest.raises(ValueError, match="coverage"):
        summary(pair)


def test_comparison_hash_is_expected_input_not_self_generated(pair):
    with pytest.raises(ValueError, match="comparison input SHA256"):
        summary(pair, candidate_sha256="0" * 64)


@pytest.mark.parametrize("field,value", [("alpha", .5), ("alpha", True), ("solver_seconds", -1),
                                         ("mip_gap", float("inf")), ("transfers_in", [1, 1]),
                                         ("transfers_out", [True])])
def test_invalid_snapshot_values(pair, field, value):
    root = pair[0]
    state = json.loads((root / "decisions.json").read_text())
    state["decisions"][0][field] = value
    repin(root, "decisions.json", json.dumps(state))
    with pytest.raises(ValueError):
        summary(pair)


@pytest.mark.parametrize("field,value", [("actual_net_points", 10), ("alpha", .5),
                                         ("actual_total_points", float("inf"))])
def test_settlement_accounting_is_checked_after_byte_admission(pair, field, value):
    root = pair[0]
    frame = pd.read_csv(root / "replayed.csv")
    frame[field] = value
    repin(root, "replayed.csv", frame.to_csv(index=False))
    with pytest.raises(ValueError):
        summary(pair)


def test_receipt_totals_are_checked(pair):
    root = pair[0]
    receipt = json.loads((root / "complete.json").read_text())
    receipt["net_points"] = 100
    (root / "complete.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt net_points"):
        summary(pair)


@pytest.mark.parametrize("weeks", [[141, 141], [142, 141], [True], []])
def test_invalid_requested_coverage(pair, weeks):
    with pytest.raises(ValueError):
        summary(pair, gameweeks=weeks)


def test_invalid_metric(pair):
    with pytest.raises(ValueError, match="metric"):
        summary(pair, metric="predicted_total_points")


@pytest.mark.parametrize("points", [-1., 0.])
def test_negative_and_zero_totals_are_finite_unclamped(tmp_path, points):
    pair = (bundle(tmp_path, "negative", points=points, alpha=1),
            bundle(tmp_path, "control", points=points))
    result = summary(pair)
    assert result["sides"]["candidate"]["gross_points"] == 12 * points
    assert result["sides"]["candidate"]["net_points"] == 12 * points - 4
    json.dumps(result, allow_nan=False)


def test_partial_missing_timing_is_not_measured(tmp_path):
    pair = (bundle(tmp_path, "candidate", rounds=2), bundle(tmp_path, "control", rounds=2))
    result = summary(pair, gameweeks=[141, 142])
    assert result["sides"]["candidate"]["decision_rounds"] == 2
    assert result["sides"]["candidate"]["recorded"]["solver_seconds"] == {"status": "NOT MEASURED"}
