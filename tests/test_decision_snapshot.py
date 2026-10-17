"""Stable decision identities, immutable receipts and pre-settlement integration."""

import json

import numpy as np
import pandas as pd
import pytest

from fpl.artifacts import write_json_new
from fpl.milp import optimize


def _decisions():
    return pd.DataFrame([{
        "gameweek": 153, "squad": [91, 12, 40, 8], "lineup": [91, 12],
        "captain": [91], "vice_captain": [12], "bench_order": [40, 8], "chip_played": "TC_91",
    }])


def test_snapshot_preserves_ids_order_and_normalizes_chip_without_mutating_source(tmp_path):
    decisions = _decisions()
    original = decisions.copy(deep=True)
    target = tmp_path / "decision.json"
    optimize.write_decision_snapshot(decisions, target)
    payload = json.loads(target.read_text())
    assert payload["schema_version"] == 1
    assert payload["settlement_completed"] is False
    assert payload["outcome_access_assessment"] == "NOT ASSESSED"
    assert payload["native_optimality_assessment"] == "NOT ASSESSED"
    row = payload["decisions"][0]
    assert row["squad"] == [91, 12, 40, 8]
    assert row["captain"] == [91]
    assert row["vice_captain"] == [12]
    assert row["bench_order"] == [40, 8]
    assert row["chip_played"] == "TC"
    pd.testing.assert_frame_equal(decisions, original)


def test_snapshot_refuses_overwrite_and_preserves_existing_bytes(tmp_path):
    target = tmp_path / "decision.json"
    optimize.write_decision_snapshot(_decisions(), target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        optimize.write_decision_snapshot(_decisions(), target)
    assert target.read_bytes() == original


@pytest.mark.parametrize("column", ["actual_total_points", "actual_net_points", "actual_lineup"])
def test_snapshot_rejects_settled_outcome_fields(tmp_path, column):
    frame = _decisions()
    frame[column] = 0
    target = tmp_path / "decision.json"
    with pytest.raises(ValueError, match="outcome"):
        optimize.write_decision_snapshot(frame, target)
    assert not target.exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_json_artifact_rejects_nonfinite_nested_values_without_creating_file(tmp_path, bad):
    target = tmp_path / "artifact.json"
    with pytest.raises(ValueError):
        write_json_new(target, {"measurements": [{"value": bad}]})
    assert not target.exists()


def test_optimizer_writes_decisions_before_settlement_and_exports_consistent_costs(tmp_path):
    roles = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    frame = pd.DataFrame([
        {"player_id": i, "GW": 1, "name": f"display_{i}", "position": role,
         "team": f"club{i % 5}", "value": 50, "predicted_total_points": i + 1,
         "actual_total_points": i + 1, "minutes": 90}
        for i, role in enumerate(roles)
    ])
    original_frame = frame.copy(deep=True)
    source = tmp_path / "predictions.csv"
    output = tmp_path / "squads.csv"
    snapshot = tmp_path / "decisions.json"
    frame.to_csv(source, index=False)
    original_bytes = source.read_bytes()
    result = optimize.run(optimize.parse_args([
        "--predictions-csv", str(source), "--start-gw", "1", "--max-gw", "1",
        "--horizon", "1", "--scoring-mode", "corrected", "--output", str(output),
        "--decisions-json", str(snapshot),
    ]))
    payload = json.loads(snapshot.read_text())
    locked = payload["decisions"][0]
    assert payload["settlement_completed"] is False
    assert payload["outcome_access_assessment"] == "NOT ASSESSED"
    assert not any(key.startswith("actual_") for key in locked)
    assert locked["squad"] == result.iloc[0].squad
    assert locked["lineup"] == result.iloc[0].lineup
    assert locked["captain"] == result.iloc[0].captain
    assert locked["bench_order"] == result.iloc[0].bench_order
    assert all(isinstance(player_id, int) for player_id in locked["squad"])
    saved = pd.read_csv(output).iloc[0]
    assert np.isfinite(saved[["actual_total_points", "actual_net_points", "alpha", "transfer_hit_points"]].astype(float)).all()
    assert saved.transfer_hit_points == 4 * saved.alpha
    assert saved.actual_net_points == saved.actual_total_points - saved.transfer_hit_points
    assert source.read_bytes() == original_bytes
    pd.testing.assert_frame_equal(frame, original_frame)
