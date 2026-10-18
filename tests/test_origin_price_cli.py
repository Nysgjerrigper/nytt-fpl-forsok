"""Mechanical CLI serialization tests use mocked schedules, never solver evidence."""

import gzip
import hashlib
import json
import sys

import pandas as pd
import pytest

from fpl.milp import optimize, origin_prices, solio


def _paths(tmp_path):
    source = tmp_path / "predictions.csv"
    pd.DataFrame({"player_id": [1001], "GW": [153], "origin_gw": [153], "predicted_total_points": [3.]}).to_csv(source, index=False)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"purchase_prices": {"1001": 50, "1002": 47}}))
    return {"source": source, "state": state, "descriptor": tmp_path / "captures.json",
            "output": tmp_path / "decisions.csv", "snapshot": tmp_path / "decisions.json"}


def _argv(paths, *, capture=True, snapshot=True):
    args = ["fpl.milp.solio", "--predictions-csv", str(paths["source"]), "--season", "2024-25",
            "--start-gw", "153", "--max-gw", "154", "--initial-squad", "1001,1002",
            "--state-json", str(paths["state"]), "--output", str(paths["output"])]
    if capture: args += ["--origin-price-captures", str(paths["descriptor"])]
    if snapshot: args += ["--decisions-json", str(paths["snapshot"])]
    return args


def _decisions():
    receipt = {"origin_gw": 153, "captured_at": "2024-08-16T16:30:00Z",
               "event_deadline": "2024-08-16T17:30:00Z", "excluded_player_ids": [9999],
               "input_sha256": {"manifest.json": "a" * 64, "bootstrap-static.json.gz": "b" * 64}}
    return pd.DataFrame([{"gameweek": 153, "squad": [1001, 1002], "lineup": [1001, 1002], "bench_order": [],
        "captain": [1001], "vice_captain": [1002], "transfers_in": [], "transfers_out": [],
        "permanent_squad_start": [1001, 1002], "permanent_squad_end": [1001, 1002],
        "budget_start": 20., "budget_end": 20., "q_start": 1, "q_end": 2, "alpha": 0,
        "chip_played": "TC_player1", "price_input_receipt": receipt, "solver_outcome_columns": "ABSENT"}])


def test_cli_passes_exact_capture_origins_and_integer_basis_and_serializes_receipt(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    calls = []
    captured = {153: {"test_capture": 153}, 154: {"test_capture": 154}}
    def load(path, **kwargs):
        assert path == paths["descriptor"]
        assert kwargs == {"season": "2024-25", "origins": [153, 154]}
        calls.append("capture")
        return captured
    monkeypatch.setattr(origin_prices, "load_origin_captures", load)
    def schedule(frame, **kwargs):
        assert kwargs["origin_price_captures"] is captured
        assert kwargs["purchase_prices"] == {1001: 50, 1002: 47}
        assert all(type(key) is int for key in kwargs["purchase_prices"])
        assert kwargs["season"] == "2024-25" and kwargs["start_gw"] == 153 and kwargs["max_gw"] == 154
        assert kwargs["initial_squad"] == [1001, 1002]
        calls.append("schedule")
        return _decisions()
    monkeypatch.setattr(solio, "run_schedule", schedule)
    monkeypatch.setattr(solio, "score_decisions", lambda *a, **k: pytest.fail("CLI settled decisions"))
    original_write = optimize.write_decision_snapshot
    def write(frame, path):
        assert not paths["output"].exists()
        assert not any(column.startswith("actual_") for column in frame)
        calls.append("snapshot")
        original_write(frame, path)
    monkeypatch.setattr(optimize, "write_decision_snapshot", write)
    monkeypatch.setattr(sys, "argv", _argv(paths))
    solio.main()
    assert calls == ["capture", "schedule", "snapshot"]
    payload = json.loads(paths["snapshot"].read_text())
    assert payload["settlement_completed"] is False
    assert payload["decisions"][0]["captain"] == [1001]
    assert payload["decisions"][0]["chip_played"] == "TC"
    assert payload["decisions"][0]["price_input_receipt"] == _decisions().iloc[0].price_input_receipt
    exported = pd.read_csv(paths["output"]).iloc[0]
    assert json.loads(exported.price_input_receipt) == payload["decisions"][0]["price_input_receipt"]
    assert json.loads(exported.permanent_squad_start) == [1001, 1002]
    assert json.loads(exported.permanent_squad_end) == [1001, 1002]
    assert json.loads(exported.captain) == [1001]
    assert "actual_total_points" not in exported.index


@pytest.mark.parametrize("problem", ["no_capture", "output_exists", "snapshot_exists", "same_path"])
def test_cli_rejects_snapshot_capture_or_overwrite_conflicts_before_schedule(tmp_path, monkeypatch, problem):
    paths = _paths(tmp_path)
    capture = problem != "no_capture"
    if problem == "output_exists": paths["output"].write_text("preserve output")
    elif problem == "snapshot_exists": paths["snapshot"].write_text("preserve snapshot")
    elif problem == "same_path": paths["snapshot"] = paths["output"]
    before = {path: path.read_bytes() for path in paths.values() if path.exists()}
    monkeypatch.setattr(solio, "run_schedule", lambda *a, **k: pytest.fail("invalid CLI reached schedule"))
    monkeypatch.setattr(origin_prices, "load_origin_captures", lambda *a, **k: pytest.fail("invalid CLI loaded capture"))
    monkeypatch.setattr(sys, "argv", _argv(paths, capture=capture))
    with pytest.raises((ValueError, FileExistsError)):
        solio.main()
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("basis", [{"01001": 50}, {"+1001": 50}, {"1001.0": 50}, {"0": 50},
                                    {"-1001": 50}, {"true": 50}, [50]])
def test_cli_rejects_noncanonical_purchase_price_json_keys_before_capture(tmp_path, monkeypatch, basis):
    paths = _paths(tmp_path)
    paths["state"].write_text(json.dumps({"purchase_prices": basis}))
    monkeypatch.setattr(origin_prices, "load_origin_captures", lambda *a, **k: pytest.fail("bad basis reached capture loading"))
    monkeypatch.setattr(solio, "run_schedule", lambda *a, **k: pytest.fail("bad basis reached schedule"))
    monkeypatch.setattr(sys, "argv", _argv(paths))
    with pytest.raises(ValueError, match="canonical"):
        solio.main()
    assert not paths["output"].exists() and not paths["snapshot"].exists()


def _admitted_capture(tmp_path):
    folder = tmp_path / "admitted"
    folder.mkdir()
    bootstrap = {"teams": [{"id": i, "name": f"club{i}"} for i in range(1, 21)],
        "elements": [{"id": 1, "code": 1001, "team": 1, "element_type": 1, "now_cost": 54, "web_name": "Player"}],
        "events": [{"id": 1, "deadline_time": "2024-08-16T17:30:00Z"}]}
    payload = json.dumps(bootstrap).encode()
    sha = lambda content: hashlib.sha256(content).hexdigest()
    (folder / "bootstrap-static.json.gz").write_bytes(gzip.compress(payload))
    manifest = {"event": 1, "event_deadline": "2024-08-16T17:30:00Z", "captured_at": "2024-08-16T16:30:00Z",
                "backfill": {"season": "2024-25"}, "files": [{"name": "bootstrap-static.json.gz", "endpoint": "/bootstrap-static/", "sha256": sha(payload)}]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    path = tmp_path / "admitted-captures.json"
    path.write_text(json.dumps({"schema_version": 1, "season": "2024-25", "captures": {"153": {
        "folder": folder.name, "input_sha256": {name: sha((folder / name).read_bytes()) for name in ("manifest.json", "bootstrap-static.json.gz")}}}}))
    return origin_prices.load_origin_captures(path, season="2024-25", origins=[153])[153]


@pytest.mark.parametrize("basis_value", [True, False])
def test_cli_boolean_purchase_basis_is_rejected_by_actual_selling_helper(tmp_path, monkeypatch, basis_value):
    paths = _paths(tmp_path)
    capture = _admitted_capture(tmp_path)
    paths["state"].write_text(json.dumps({"purchase_prices": {"1001": basis_value}}))
    monkeypatch.setattr(origin_prices, "load_origin_captures", lambda *a, **k: {153: capture})
    def schedule(frame, **kwargs):
        assert kwargs["purchase_prices"] == {1001: basis_value}
        origin_prices.selling_prices(capture, kwargs["initial_squad"], kwargs["purchase_prices"])
        pytest.fail("boolean basis passed selling helper")
    monkeypatch.setattr(solio, "run_schedule", schedule)
    argv = _argv(paths)
    argv[argv.index("--max-gw") + 1] = "153"
    argv[argv.index("--initial-squad") + 1] = "1001"
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="purchase price"):
        solio.main()
    assert not paths["output"].exists() and not paths["snapshot"].exists()
