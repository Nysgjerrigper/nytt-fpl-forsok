"""Immutable wrapper evidence survives failures without weakening native acceptance."""
import hashlib
import json
import sys

import pandas as pd
import pytest

from fpl.milp import solio
from fpl.milp.solio_receipts import ScheduleReceipts


def forecasts():
    roles = ["GK"]*2 + ["DEF"]*5 + ["MID"]*5 + ["FWD"]*3
    return pd.DataFrame([{"player_id": p+1, "GW": gw, "name": f"p{p+1}",
        "position": roles[p], "team": f"club{p % 5}", "value": 50,
        "predicted_total_points": 2, "actual_total_points": 1, "minutes": 90}
        for gw in (1, 2) for p in range(15)])


def run(frame, **kwargs):
    return solio.run_schedule(frame, season="2020-21", start_gw=1, max_gw=2, horizon=1, **kwargs)


def read(path):
    return json.loads(path.read_text())


def events(folder, name):
    return [read(p) for p in sorted(folder.glob(f"*-{name}.json"))]


def verify_terminal(folder, name):
    terminal = read(folder / name)
    actual = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob('*') if p.is_file() and p.name != name}
    assert actual == terminal["receipt_sha256"]
    assert terminal["started_sha256"] == actual["started.json"]
    for p in folder.glob('[0-9]*.json'):
        assert read(p)["started_sha256"] == actual["started.json"]
    return terminal


def test_checkpoint_output_matches_default_and_binds_pre_settlement_decisions(tmp_path):
    frame = forecasts()
    ordinary = run(frame)
    folder = tmp_path / "recorded"
    recorded = run(frame, checkpoint_dir=folder)
    columns = [c for c in ordinary if c not in ("solver_seconds", "window_seconds")]
    pd.testing.assert_frame_equal(ordinary[columns], recorded[columns])
    assert recorded.actual_net_points.tolist() == [12, 12]
    receipt = verify_terminal(folder, "complete.json")
    assert receipt["attempted_vendor_calls"] == receipt["completed_vendor_calls"] == 2
    assert receipt["completed_origins"] == 2 and receipt["requested_origins"] == [1, 2]
    assert receipt["outcome_access_assessment"] == "NOT ASSESSED"
    assert not (folder / "failed.json").exists()
    for event in events(folder, "decision"):
        assert event["settlement_completed"] is False
        assert not any(c.startswith("actual_") for c in event["decision"])
        assert len(event["decision"]["squad"]) == 15
        assert len(event["next_state"]["squad"]) == 15
    for event in events(folder, "vendor-complete"):
        assert event["native_seconds"] >= 0
        assert event["vendor_wall_seconds"] >= 0
        assert event["overhead_basis"] == "NON_NATIVE_OVERHEAD_NOT_BUILD_TIME"
    assert all(e["preparation_seconds"] >= 0 for e in events(folder, "prepared"))
    started = read(folder / "started.json")
    assert started["input_frame"]["rows"] == 30
    assert "fpl/milp/_vendor/solio/solver.py" in started["source_sha256"]


def test_second_origin_failure_preserves_one_decision_and_exact_wrapper_counts(tmp_path, monkeypatch):
    original = solio.solve_multi_period_fpl
    calls = []
    def fail_second(prepared, options):
        calls.append(prepared["next_gw"])
        if len(calls) == 2:
            raise RuntimeError("controlled second-origin native failure")
        return original(prepared, options)
    monkeypatch.setattr(solio, "solve_multi_period_fpl", fail_second)
    folder = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="second-origin"):
        run(forecasts(), checkpoint_dir=folder)
    receipt = verify_terminal(folder, "failed.json")
    assert calls == [1, 2]
    assert receipt["failed_origin"] == 2 and receipt["failed_stage"] == "vendor-failed"
    assert receipt["attempted_vendor_calls"] == 2
    assert receipt["completed_vendor_calls"] == receipt["completed_origins"] == 1
    assert receipt["failed_native_seconds"] == receipt["failed_build_seconds"] == "NOT MEASURED"
    assert len(events(folder, "decision")) == 1
    assert not (folder / "complete.json").exists()


def test_admission_failure_records_zero_vendor_calls(tmp_path):
    folder = tmp_path / "admission"
    with pytest.raises(ValueError, match="missing prediction round"):
        run(forecasts().query('GW == 1'), checkpoint_dir=folder)
    receipt = verify_terminal(folder, "failed.json")
    assert receipt["failed_origin"] == 2
    assert receipt["attempted_vendor_calls"] == receipt["completed_origins"] == 1
    # First-origin invalid input must not reach the vendor at all.
    folder = tmp_path / "first-admission"
    with pytest.raises(ValueError, match="missing metadata"):
        run(forecasts().drop(columns="name"), checkpoint_dir=folder)
    receipt = verify_terminal(folder, "failed.json")
    assert receipt["attempted_vendor_calls"] == receipt["completed_origins"] == 0


def test_nonoptimal_vendor_result_and_tiny_limit_still_fail_closed(tmp_path, monkeypatch):
    folder = tmp_path / "limit"
    with pytest.raises(RuntimeError, match="solve rejected"):
        run(forecasts(), checkpoint_dir=folder, time_limit=1e-9)
    receipt = verify_terminal(folder, "failed.json")
    assert receipt["attempted_vendor_calls"] == 1 and receipt["completed_origins"] == 0
    monkeypatch.setattr(solio, "solve_multi_period_fpl", lambda *args: [{"solver_status": "Feasible"}])
    folder = tmp_path / "nonoptimal"
    with pytest.raises(RuntimeError, match="exactly one optimal"):
        run(forecasts(), checkpoint_dir=folder)
    receipt = verify_terminal(folder, "failed.json")
    assert receipt["completed_vendor_calls"] == 1 and receipt["completed_origins"] == 0
    assert events(folder, "vendor-complete")[0]["native_seconds"] == "NOT MEASURED"


def test_existing_and_production_destinations_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "exists"
    folder.mkdir()
    with pytest.raises(FileExistsError): run(forecasts(), checkpoint_dir=folder)
    assert list(folder.iterdir()) == []
    monkeypatch.setattr(solio.config, "DATASETT_DIR", tmp_path / "dataset")
    with pytest.raises(ValueError, match="production"):
        run(forecasts(), checkpoint_dir=tmp_path / "dataset/run")
    assert not (tmp_path / "dataset").exists()


def test_interrupted_call_has_no_terminal_and_no_automatic_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(solio, "solve_multi_period_fpl", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    folder = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt): run(forecasts(), checkpoint_dir=folder)
    assert (folder / "started.json").exists()
    assert len(events(folder, "vendor-start")) == 1
    assert not (folder / "failed.json").exists() and not (folder / "complete.json").exists()
    with pytest.raises(FileExistsError): run(forecasts(), checkpoint_dir=folder)


def test_decision_checkpoint_survives_later_scoring_failure(tmp_path, monkeypatch):
    def bad_score(*args, **kwargs): raise ValueError("scoring failed")
    monkeypatch.setattr(solio, "score_decisions", bad_score)
    folder = tmp_path / "scoring"
    with pytest.raises(ValueError, match="scoring failed"):
        run(forecasts(), checkpoint_dir=folder)
    receipt = verify_terminal(folder, "failed.json")
    assert receipt["completed_origins"] == 2
    assert receipt["failed_stage"] == "schedule-scoring-start"
    assert len(events(folder, "decision")) == 2


def test_cli_propagates_checkpoint_directory(tmp_path, monkeypatch):
    frame = forecasts()
    input_path, output = tmp_path / 'input.csv', tmp_path / 'out.csv'
    frame.to_csv(input_path, index=False)
    monkeypatch.setattr(sys, 'argv', ['solio', '--predictions-csv', str(input_path), '--season', '2020-21',
        '--start-gw', '1', '--max-gw', '2', '--horizon', '1', '--output', str(output),
        '--checkpoint-dir', str(tmp_path / 'cli')])
    solio.main()
    assert output.exists()
    assert verify_terminal(tmp_path / 'cli', 'complete.json')['completed_origins'] == 2


def test_checkpoint_state_keeps_free_hit_permanent_holdings(tmp_path):
    frame = forecasts()
    folder = tmp_path / "free-hit"
    frame["GW"] += 1
    result = solio.run_schedule(frame, season="2020-21", start_gw=2, max_gw=3, horizon=1,
        checkpoint_dir=folder, initial_squad=list(range(1, 16)), bank=250,
        free_transfers=4, chip_inventory={"fh": {1: 1}}, forced_chips={"fh": [2]})
    first, second = events(folder, "decision")
    assert first["decision"]["chip_played"] == "FH"
    assert first["next_state"]["squad"] == list(range(1, 16))
    assert first["next_state"]["bank"] == 250
    assert first["next_state"]["free_transfers"] == 4
    assert first["next_state"]["previous_free_hit_gw"] == 2
    assert second["decision"]["temporary_selection_bank_end"] is None
    assert result.iloc[0].q_end == 4
    verify_terminal(folder, "complete.json")


def test_source_change_cannot_seal_completed_schedule(tmp_path, monkeypatch):
    from fpl.milp import solio_receipts
    original = solio_receipts.source_runtime
    calls = []
    def changing_source():
        value = original()
        calls.append(1)
        if len(calls) >= 2:
            value["runtime"]["changed"] = True
        return value
    monkeypatch.setattr(solio_receipts, 'source_runtime', changing_source)
    folder = tmp_path / 'changed-source'
    with pytest.raises(ValueError, match='source or runtime changed'):
        run(forecasts(), checkpoint_dir=folder)
    assert not (folder / 'complete.json').exists()
    assert verify_terminal(folder, 'failed.json')['completed_origins'] == 2


def test_nonfinite_initial_state_cannot_create_receipt_directory(tmp_path):
    folder = tmp_path / 'nonfinite'
    with pytest.raises(ValueError): run(forecasts(), bank=float('nan'), checkpoint_dir=folder)
    assert not folder.exists()


def test_invalid_vendor_return_is_counted_as_returned_not_completed_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(solio, "solve_multi_period_fpl", lambda *args: None)
    folder = tmp_path / 'invalid-return'
    with pytest.raises(TypeError): run(forecasts(), checkpoint_dir=folder)
    receipt = verify_terminal(folder, 'failed.json')
    assert receipt['attempted_vendor_calls'] == receipt['completed_vendor_calls'] == 1
    assert receipt['completed_origins'] == 0
    assert receipt['failed_stage'] == 'normalization-start'
    assert events(folder, 'vendor-complete')[0]['native_seconds'] == 'NOT MEASURED'
