"""Synthetic workflow guards; these checks supply no model-performance evidence."""
import json
import subprocess
from pathlib import Path

import pytest

from fpl import research_workflow as workflow
from fpl.artifacts import sha256_file, write_json_new


@pytest.fixture
def registered(tmp_path):
    plan = {"schema_version": 1, "promotion_eligible": False,
            "window": {"start_gw": 141, "end_gw": 144, "retrain_every": 4,
                       "horizon": 3, "phase": "selection_diagnostic"},
            "dataset_sha256": "a" * 64, "source_sha256": {"example.py": "b" * 64},
            "runtime": {"python": "synthetic"}, "seed": 0,
            "arms": {"control": {"model": "catboost_mae", "features": ["minutes_roll5", "form_roll5"]},
                     "candidate": {"model": "catboost_rmse", "features": ["minutes_roll5", "form_roll5"]}},
            "budget": {"max_seconds": 10, "solve_seconds": 1, "max_fit_calls": 16, "max_solver_calls": 16}}
    for name in ("first.json", "second.json"):
        write_json_new(tmp_path / name, plan)
    spec = {key: f"Test {key}" for key in workflow._TEXT}
    spec.update(max_worker_seconds=20, plans=["first.json", "second.json"])
    path = tmp_path / "spec.json"
    write_json_new(path, spec)
    return path, tmp_path / "campaign", spec, plan


def fake_run(plan, dataset, output):
    """Seal minimal synthetic outputs with the real runner's receipt format."""
    output.mkdir()
    (output / "plan.json").write_bytes(plan.read_bytes())
    comparisons = {}
    for protocol in ("standard", "origin"):
        comparisons[protocol] = {"total_a": 43, "total_b": 42, "total_diff": 1, "ci_low": -4, "ci_high": 5}
        stats = {"mae": 1.2, "rmse": 2.3, "bias": -0.1, "total_calibration": 0.9, "spearman": 0.2, "top1_capture": 0.5}
        write_json_new(output / f"{protocol}-forecast-summary.json",
                       {"summaries": [{"lead": None if protocol == "standard" else 0,
                                       "candidate": {"overall": stats}, "control": {"overall": stats}}]})
    write_json_new(output / "complete.json", {"schema_version": 1, "status": "COMPLETED",
        "promotion_eligible": False, "fresh_holdout": False, "plan_sha256": sha256_file(plan),
        "comparisons": comparisons, "outputs": {p.name: sha256_file(p) for p in output.iterdir()}})
    return output


def test_register_captures_exact_bytes_without_execution(registered, monkeypatch):
    path, root, _, _ = registered
    monkeypatch.setattr(workflow.runner, "run_plan", lambda *a: pytest.fail("registration executed compute"))
    assert workflow.register(path, root) == root
    assert (root / "plan-1.json").read_bytes() == (path.parent / "first.json").read_bytes()
    assert "0/2 launches" in workflow.report(root)
    with pytest.raises(FileExistsError):
        workflow.register(path, root)


@pytest.mark.parametrize("change", ["budget", "nan", "bool", "empty", "final", "forged_phase", "mismatched_control", "mismatched_seed"])
def test_invalid_designs_rejected_before_output(registered, change):
    path, root, spec, plan = registered
    if change in ("budget", "nan", "bool"):
        spec["max_worker_seconds"] = {"budget": 19, "nan": float("nan"), "bool": True}[change]
    elif change == "empty":
        spec["hypothesis"] = "  "
    elif change in ("final", "forged_phase"):
        plan["window"].update(start_gw=153, end_gw=156)
        if change == "final":
            plan["window"]["phase"] = "historical_evaluation"
    elif change == "mismatched_control":
        plan["arms"]["control"]["model"] = "different"
    else:
        plan["seed"] = 1
    path.write_text(json.dumps(spec))
    (path.parent / "second.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        workflow.register(path, root)
    assert not root.exists()


@pytest.mark.parametrize("artifact", ["registration.json", "plan-1.json"])
def test_changed_registration_or_plan_stops_before_compute(registered, monkeypatch, artifact):
    path, root, _, _ = registered
    workflow.register(path, root)
    (root / artifact).write_text((root / artifact).read_text() + "\n")
    monkeypatch.setattr(workflow.runner, "run_plan", lambda *a: pytest.fail("changed plan reached worker"))
    with pytest.raises(ValueError, match="bytes changed"):
        workflow.run_next(root, "unused.csv", "First contrast")


def test_sequential_evidence_and_iteration_budget(registered, monkeypatch):
    path, root, _, _ = registered
    workflow.register(path, root)
    monkeypatch.setattr(workflow.runner, "run_plan", fake_run)
    with pytest.raises(ValueError, match="reason"):
        workflow.run_next(root, "unused.csv", "")
    assert not (root / "launch-1.json").exists()
    for number in (1, 2):
        workflow.run_next(root, "unused.csv", f"Evidence for trial {number}")
        claim = json.loads((root / f"launch-{number}.json").read_text())
        assert claim["reason"] == f"Evidence for trial {number}"
        previous = None if number == 1 else sha256_file(root / "trial-1" / "complete.json")
        assert claim["previous_complete_sha256"] == previous
    with pytest.raises(ValueError, match="exhausted"):
        workflow.run_next(root, "unused.csv", "No budget left")
    report = workflow.report(root)
    assert "43/42" in report and "paired CI [-4, 5]" in report and "rmse=2.3" in report
    (root / "trial-1" / "standard-forecast-summary.json").write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        workflow.report(root)


@pytest.mark.parametrize("failed", [False, True])
def test_incomplete_or_failed_attempt_is_preserved_and_blocks_continuation(registered, monkeypatch, failed):
    path, root, _, _ = registered
    workflow.register(path, root)
    def fail(plan, dataset, output):
        if failed:
            write_json_new(output / "failed.json", {"status": "FAILED"})
        raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(workflow.runner, "run_plan", fail)
    with pytest.raises(RuntimeError):
        workflow.run_next(root, "unused.csv", "First contrast")
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.run_next(root, "unused.csv", "Must not retry")
    assert (root / "launch-1.json").exists() and not (root / "launch-2.json").exists()
    assert ("FAILED" if failed else "INCOMPLETE") in workflow.report(root)
    assert "candidate/control net points" not in workflow.report(root)


def test_existing_lock_and_broken_chronology_refuse_launch(registered, monkeypatch):
    path, root, _, _ = registered
    workflow.register(path, root)
    monkeypatch.setattr(workflow.runner, "run_plan", lambda *a: pytest.fail("unsafe launch"))
    write_json_new(root / "launch.lock", {"reason": "another process"})
    with pytest.raises(FileExistsError):
        workflow.run_next(root, "unused.csv", "Must not steal lock")
    assert (root / "launch.lock").exists()
    (root / "launch.lock").unlink()
    write_json_new(root / "launch-2.json", {})
    with pytest.raises(ValueError, match="chronology"):
        workflow.run_next(root, "unused.csv", "Must not skip launch")


def test_same_model_ablation_report_explains_feature_change(registered, monkeypatch):
    path, root, _, plan = registered
    plan["arms"]["candidate"] = {"model": "catboost_mae", "features": ["form_roll5"]}
    (path.parent / "first.json").write_text(json.dumps(plan))
    workflow.register(path, root)
    monkeypatch.setattr(workflow.runner, "run_plan", fake_run)
    workflow.run_next(root, "unused.csv", "Registered minutes ablation")
    assert "dropped ['minutes_roll5']; added []" in workflow.report(root)


def test_cli_timeout_stops_concisely_and_keeps_claim(registered, monkeypatch, capsys):
    path, root, _, _ = registered
    workflow.register(path, root)
    def timeout(*args):
        raise subprocess.TimeoutExpired("synthetic worker", 10)
    monkeypatch.setattr(workflow.runner, "run_plan", timeout)
    with pytest.raises(SystemExit) as stopped:
        workflow.main(["next", str(root), "--dataset", "unused.csv", "--reason", "Registered trial"])
    assert stopped.value.code == 2
    assert "Research workflow stopped:" in capsys.readouterr().err
    assert (root / "launch-1.json").exists()
    assert not (root / "launch.lock").exists()
    assert "INCOMPLETE" in workflow.report(root)
