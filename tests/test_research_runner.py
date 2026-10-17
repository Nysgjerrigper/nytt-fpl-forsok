"""Bounded runner tests use synthetic files and stubs, never forecasting evidence."""

from contextlib import redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pandas as pd
import pytest

from fpl import research_runner as runner
from fpl.artifacts import sha256_file, write_json_new


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    path = tmp_path / "tiny.csv"
    frame = pd.DataFrame([
        {"player_id": i, "GW_global": gw, "position": position, "name": f"p{i}",
         "team": f"club{i}", "value": 50, "total_points": i + 1, "minutes": 90,
         "fixture": gw * 10, "form_roll5": float(i), "was_home": 1}
        for gw in (136, 137, 138, 139, 140, 153, 154, 155, 156, 232)
        for i, position in enumerate(("GK", "DEF", "MID", "FWD"), 1)
    ])
    frame.to_csv(path, index=False)
    seen = []
    def load(source):
        result = pd.read_csv(source)
        result.attrs = {"frozen_dataset_sha256": sha256_file(source), "frozen_dataframe_fingerprint": "complete"}
        return result
    def build(raw):
        seen.append(raw.copy())
        return raw.copy()
    monkeypatch.setattr(runner, "load_frozen_research_dataset", load)
    monkeypatch.setattr(runner.features, "build_feature_frame", build)
    monkeypatch.setattr(runner.features, "feature_columns", lambda _: ["form_roll5", "was_home"])
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"fpl/example.py": "a" * 64})
    monkeypatch.setattr(runner, "_runtime_versions", lambda: {"python": "test-runtime"})
    return path, seen


def _plan(admitted, **kwargs):
    return runner.create_plan(admitted[0], start_gw=153, end_gw=156, **kwargs)


def _save(tmp_path, plan):
    path = tmp_path / "registered-plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_prepare_resolves_complete_recipes_without_fits_or_solves(admitted, monkeypatch):
    monkeypatch.setattr(runner.models, "fit_model", lambda *a, **k: pytest.fail("prepare fitted a model"))
    monkeypatch.setattr(runner.optimize, "run", lambda *a, **k: pytest.fail("prepare solved a MILP"))
    plan = _plan(admitted, seed=np.int64(3), parameters={"iterations": 7})
    assert plan["window"]["phase"] == "historical_evaluation"
    assert plan["promotion_eligible"] is False
    assert plan["dataset_sha256"] == sha256_file(admitted[0])
    assert plan["budget"]["max_fit_calls"] == plan["budget"]["max_solver_calls"] == 16
    for side, loss in (("control", "MAE"), ("candidate", "RMSE")):
        arm = plan["arms"][side]
        assert arm["features"] == ["form_roll5", "was_home"]
        assert set(arm["params_by_pos"]) == {"GK", "DEF", "MID", "FWD"}
        for by_model in arm["params_by_pos"].values():
            params = by_model[arm["model"]]
            assert params["iterations"] == 7
            assert params["loss_function"] == loss
            assert params["random_seed"] == 3
            assert params["learning_rate"] == .05
    assert admitted[1][-1].GW_global.max() == 156
    assert admitted[1][-1].GW_global.min() == 136
    assert "frozen_dataframe_fingerprint" not in admitted[1][-1].attrs
    json.dumps(plan, allow_nan=False)


def test_selection_phase_and_same_model_feature_ablation_are_explicit(admitted):
    selection = runner.create_plan(admitted[0], start_gw=137, end_gw=140)
    assert selection["window"]["phase"] == "selection_diagnostic"
    ablation = _plan(admitted, candidate_model="catboost_mae", candidate_drop=["was_home"])
    assert ablation["arms"]["candidate"]["features"] == ["form_roll5"]


@pytest.mark.parametrize("start,end", [(136, 140), (150, 155), (191, 194), (232, 235), (156, 153), (True, 154)])
def test_unregistered_spent_or_invalid_windows_rejected_before_features(admitted, start, end):
    with pytest.raises(ValueError):
        runner.create_plan(admitted[0], start_gw=start, end_gw=end)
    assert admitted[1] == []


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": -1}, {"seed": 1.5}, {"retrain_every": 0},
                                    {"horizon": False}, {"max_seconds": float("nan")}, {"solve_seconds": 0}])
def test_invalid_seed_schedule_or_budget_rejected(admitted, kwargs):
    with pytest.raises(ValueError):
        _plan(admitted, **kwargs)


@pytest.mark.parametrize("kwargs", [{"candidate_drop": ["was_home"]}, {"control_model": "lightgbm"},
                                    {"candidate_drop": ["missing"]}, {"candidate_drop": ["was_home", "was_home"]},
                                    {"parameters": {"iterations": float("inf")}}])
def test_confounded_or_unsupported_recipes_rejected(admitted, kwargs):
    with pytest.raises(ValueError):
        _plan(admitted, **kwargs)


@pytest.mark.parametrize("change", ["missing_round", "missing_training_position"])
def test_contiguous_rounds_and_complete_training_positions_required(admitted, change):
    frame = pd.read_csv(admitted[0])
    if change == "missing_round":
        frame = frame.loc[frame.GW_global != 154]
    else:
        frame = frame.loc[~((frame.GW_global < 153) & (frame.position == "GK"))]
    frame.to_csv(admitted[0], index=False)
    with pytest.raises(ValueError):
        _plan(admitted)


@pytest.mark.parametrize("change", ["source", "runtime", "dataset", "feature_order", "incomplete_recipe", "mismatched_depth", "unknown_field"])
def test_preexecution_identity_and_recipe_failures_create_no_run(tmp_path, admitted, monkeypatch, change):
    plan = _plan(admitted)
    if change == "source":
        monkeypatch.setattr(runner, "_source_hashes", lambda: {"fpl/example.py": "b" * 64})
    elif change == "runtime":
        monkeypatch.setattr(runner, "_runtime_versions", lambda: {"python": "changed"})
    elif change == "dataset":
        admitted[0].write_text(admitted[0].read_text() + "\n")
    elif change == "feature_order":
        plan["arms"]["candidate"]["features"].reverse()
    elif change == "incomplete_recipe":
        plan["arms"]["candidate"]["params_by_pos"]["GK"]["catboost_rmse"].pop("depth")
    elif change == "mismatched_depth":
        plan["arms"]["candidate"]["params_by_pos"]["GK"]["catboost_rmse"]["depth"] += 1
    else:
        plan["execute"] = "arbitrary code"
    path = _save(tmp_path, plan)
    output = tmp_path / "output"
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("invalid plan launched a worker"))
    with pytest.raises(ValueError):
        runner.run_plan(path, admitted[0], output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_subprocess_failure_is_preserved_and_existing_attempt_cannot_retry(tmp_path, admitted, monkeypatch, failure):
    plan = _plan(admitted)
    path = _save(tmp_path, plan)
    calls = []
    def process(command, **kwargs):
        calls.append((command, kwargs))
        assert command[1:4] == ["-m", "fpl.research_runner", "_worker"]
        assert kwargs["timeout"] == plan["budget"]["max_seconds"]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=9)
    monkeypatch.setattr(runner.subprocess, "run", process)
    root = tmp_path / "output"
    with pytest.raises((subprocess.TimeoutExpired, RuntimeError)):
        runner.run_plan(path, admitted[0], root)
    failed = json.loads((root / "failed.json").read_text())
    assert failed["status"] == "FAILED" and failed["promotion_eligible"] is False
    assert not (root / "complete.json").exists()
    assert (root / "plan.json").read_bytes() == path.read_bytes()
    assert sha256_file(root / "dataset.csv") == plan["dataset_sha256"]
    with pytest.raises(FileExistsError):
        runner.run_plan(path, admitted[0], root)
    assert len(calls) == 1


def _receipt(root):
    root.mkdir()
    (root / "plan.json").write_text('{"example":"synthetic receipt validation"}')
    (root / "report.txt").write_text("test only")
    receipt = {"schema_version": 1, "status": "COMPLETED", "promotion_eligible": False,
               "fresh_holdout": False, "plan_sha256": sha256_file(root / "plan.json"),
               "outputs": {path.name: sha256_file(path) for path in root.iterdir()}}
    write_json_new(root / "complete.json", receipt)
    return receipt


def test_forecast_failure_stops_before_solving_and_restores_fit_hook(tmp_path, admitted, monkeypatch):
    plan = _plan(admitted)
    path = _save(tmp_path, plan)
    original_fit = runner.models.fit_model
    def forecast(*args, **kwargs):
        raise RuntimeError("synthetic forecast failure")
    monkeypatch.setattr(runner.predict, "walk_forward_predictions", forecast)
    monkeypatch.setattr(runner.optimize, "run", lambda *a, **k: pytest.fail("failed forecast reached optimizer"))
    def process(command, **kwargs):
        with redirect_stdout(kwargs["stdout"]):
            runner._worker(command[-1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", process)
    root = tmp_path / "failed-forecast"
    with pytest.raises(RuntimeError, match="synthetic forecast failure"):
        runner.run_plan(path, admitted[0], root)
    assert runner.models.fit_model is original_fit
    assert (root / "failed.json").exists()
    assert not (root / "complete.json").exists()


def test_read_result_accepts_exact_file_coverage(tmp_path):
    root = tmp_path / "receipt"
    expected = _receipt(root)
    assert runner.read_result(root) == expected


@pytest.mark.parametrize("change", ["extra", "missing", "changed", "planhash", "failed", "escape"])
def test_read_result_rejects_corruption_coverage_and_unsafe_paths(tmp_path, change):
    root = tmp_path / "receipt"
    receipt = _receipt(root)
    if change == "extra":
        (root / "extra.txt").write_text("extra")
    elif change == "missing":
        (root / "report.txt").unlink()
    elif change == "changed":
        (root / "report.txt").write_text("changed")
    elif change == "planhash":
        receipt["plan_sha256"] = "a" * 64
        (root / "complete.json").write_text(json.dumps(receipt))
    elif change == "failed":
        write_json_new(root / "failed.json", {"reason": "late validation failed"})
    else:
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")
        (root / "alias.txt").symlink_to(outside)
        receipt["outputs"]["alias.txt"] = sha256_file(outside)
        (root / "complete.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        runner.read_result(root)


def test_fake_worker_integrates_ledgers_forecasts_solves_and_real_scorecards(tmp_path, admitted, monkeypatch):
    plan = _plan(admitted, parameters={"iterations": 2})
    path = _save(tmp_path, plan)
    root = tmp_path / "output"
    calls = []
    def fitted(*args, **kwargs):
        return SimpleNamespace(get_all_params=lambda: {"iterations": 2, "synthetic_test_model": True})
    monkeypatch.setattr(runner.models, "fit_model", fitted)
    original_fit = runner.models.fit_model
    def forecast(frame, feature_cols, start, end, **kwargs):
        history = frame.loc[frame.GW_global < start]
        model = kwargs["weight_strategy"].split(":", 1)[1]
        for position in ("GK", "DEF", "MID", "FWD"):
            training = history.loc[history.position == position]
            runner.models.fit_model(model, training[feature_cols], training.total_points, position=position,
                minutes=training.minutes, gw=training.GW_global,
                params=kwargs["params_by_pos"][position][model], seed=kwargs["seed"])
        return frame.loc[frame.GW_global.between(start, end)].rename(
            columns={"GW_global": "GW", "total_points": "actual_total_points"}).assign(predicted_total_points=3.)
    monkeypatch.setattr(runner.predict, "walk_forward_predictions", forecast)
    def origin(frame, raw, feature_cols, start, end, horizon, **kwargs):
        return forecast(frame, feature_cols, start, end, **kwargs).assign(origin_gw=lambda frame: frame.GW)
    monkeypatch.setattr(runner.predict, "origin_based_predictions", origin)
    def solve(args):
        assert len(list(root.glob("*-predictions.csv"))) == 4
        assert args.solver == "highs" and args.gap_rel == 0 and args.scoring_mode == "corrected"
        calls.append(args)
        result = pd.DataFrame({"gameweek": list(range(args.start_gw, args.max_gw + 1)),
                               "actual_total_points": [40.] * 4, "alpha": [0] * 4,
                               "actual_net_points": [40.] * 4, "scoring_protocol": ["autosubs_v2"] * 4,
                               "native_optimality_assessment": ["PROVEN_OPTIMAL"] * 4})
        result.to_csv(args.output, index=False)
        write_json_new(args.decisions_json, {"native_optimality_assessment": "PROVEN_OPTIMAL",
                                             "decisions": result.to_dict(orient="records")})
        return result
    monkeypatch.setattr(runner.optimize, "run", solve)
    # The CSV logger shells out to git; keep this test scoped to the worker subprocess.
    monkeypatch.setattr(runner.experiment, "_current_git_hash", lambda: "synthetic-test")
    def process(command, **kwargs):
        with redirect_stdout(kwargs["stdout"]):
            runner._worker(Path(command[-1]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", process)
    assert runner.run_plan(path, admitted[0], root) == root
    result = runner.read_result(root)
    assert result["fit_calls"] == result["solver_calls"] == 16
    assert len(calls) == 4
    assert runner.models.fit_model is original_fit
    assert len(list(root.glob("fit-start-*.json"))) == 16
    assert len(list(root.glob("fit-complete-*.json"))) == 16
    for log in root.glob("fit-complete-*.json"):
        record = json.loads(log.read_text())
        assert record["train_max_gw"] < 153
        assert record["features"] == ["form_roll5", "was_home"]
        assert record["seed"] == 0 and record["seconds"] >= 0
        assert record["fitted_params"]["synthetic_test_model"] is True
    assert set(result["comparisons"]) == {"standard", "origin"}
    assert all(value["evidence_status"] == "UNVERIFIED" for value in result["comparisons"].values())
    registry = pd.read_csv(root / "registry.csv")
    assert len(registry) == 2
    assert "reports/training-standard/first/complete.json" in result["outputs"]
    assert result["fresh_holdout"] is result["promotion_eligible"] is False
