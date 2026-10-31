"""Mocked cached temporal blend admission, pilot gating and immutable provenance."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpl import config
from fpl.model import blend_research as runner
from fpl.model.sequence_features import sequence_columns


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _forecasts(start, end, prediction, origin=True):
    rows = []
    for initial in range(start, end + 1):
        for gw in range(initial, min(initial + 3, end + 1)) if origin else [initial]:
            for player, position in enumerate(config.ONFIELD_POSITIONS, 1):
                row = {"player_id": player, "position": position, "GW": gw, "fixture": gw * 10 + player,
                       "predicted_total_points": float(prediction), "actual_total_points": 2.,
                       "value": 50., "minutes": 90.}
                if origin:
                    row["origin_gw"] = initial
                rows.append(row)
    return pd.DataFrame(rows)


def _raw(end):
    rows = []
    for player, position in enumerate(config.ONFIELD_POSITIONS, 1):
        for gw in range(1, end + 1):
            rows.append({"player_id": player, "position": position, "GW_global": gw,
                         "fixture": gw * 10 + player, "base": 1.})
    frame = pd.DataFrame(rows)
    for stat in config.CNN_SEQUENCE_STATS:
        frame[stat] = 2.
    return frame


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "BLEND_WEIGHT_WEEKS", 2)
    monkeypatch.setattr(config, "CNN_RETRAIN_EVERY", 100)
    current_source = tmp_path / "fpl" / "current.py"
    current_source.parent.mkdir()
    current_source.write_text("new source code\n")
    dataset = tmp_path / "frozen.csv"
    dataset.write_bytes(b"frozen bytes\n")
    digest = runner.research_package.file_hash(dataset)
    source_dir, output_dir = tmp_path / "old_run", tmp_path / "new_run"
    cutoffs = {"discovery_max_gw": 1, "selection_min_gw": 2, "selection_max_gw": 5,
               "backtest_min_gw": 6, "backtest_max_gw": 8}
    monkeypatch.setattr(runner, "protocol_cutoffs", lambda: cutoffs)
    old_source = source_dir / "source_archive" / "fpl" / "current.py"
    old_source.parent.mkdir(parents=True)
    old_source.write_text("different old source code\n")
    parameters = {model: {position: {model: {"fixed_setting": model, "seed": 0}}
                          for position in config.ONFIELD_POSITIONS}
                  for model in ("catboost_mae", "catboost_rmse", "sequence_cnn")}
    registration = {"seed": 0, "cutoffs": cutoffs, "horizon": 3, "retrain_every": 100,
                    "points_accounting": "net_after_transfer_hits", "dataset_sha256": digest,
                    "base_feature_columns": ["base"], "sequence_feature_columns": sequence_columns(),
                    "parameters": parameters, "source_archive": "source_archive",
                    "code_sha256": {"fpl/current.py": runner.research_package.file_hash(old_source)}}
    _write(source_dir / "registration.json", registration)
    selection = {}
    member_paths = {}
    for name, prediction in (("cnn_sequence", 3.), ("rmse_sequence", 1.)):
        path = source_dir / "selection" / f"{name}.csv"
        path.parent.mkdir(exist_ok=True)
        _forecasts(2, 5, prediction).to_csv(path, index=False)
        member_paths[name] = path
        selection[name] = {"predictions": path.name, "prediction_sha256": runner.research_package.file_hash(path),
                           "total_points": 100}
    summary = {"status": "selection_rejected", "selection": selection}
    _write(source_dir / "summary.json", summary)
    _write(source_dir / "selection_summary.json", {"status": "running", "selection": selection})
    _write(source_dir / "selection_lock.json", {
        "selection_summary_sha256": runner.research_package.file_hash(source_dir / "selection_summary.json")})
    _write(source_dir / "verification.json", {
        "status": "VERIFIED", "registration_sha256": runner.research_package.file_hash(source_dir / "registration.json"),
        "summary_sha256": runner.research_package.file_hash(source_dir / "summary.json"),
        "selection_lock_sha256": runner.research_package.file_hash(source_dir / "selection_lock.json")})
    monkeypatch.setattr(runner.experiment, "_library_versions", lambda: {"mock": "1"})
    records, loaded, evaluations, comparisons = [], [], [], []
    monkeypatch.setattr(runner.experiment, "log_result", lambda *args: records.append(args))
    points = {"rmse_sequence": 100., "blend": 101.}

    def load(end, path):
        assert path == dataset
        if end == 8:
            lock = json.loads((output_dir / "pilot_lock.json").read_text())
            assert lock["advances"] is True
            assert lock["pilot_summary_sha256"] == runner.research_package.file_hash(output_dir / "pilot_summary.json")
        loaded.append(end)
        raw = _raw(end)
        raw.attrs["frozen_dataset_sha256"] = digest
        return raw, raw.copy(), ["base"]
    monkeypatch.setattr(runner, "load_research_features", load)

    def evaluate(raw, frame, columns, variant, params, start, end, origin, directory, seed, **kwargs):
        assert records[0][0] == "blend_research_registration"
        assert seed == 0 and kwargs["retrain_every"] == 100
        assert kwargs["model_name"] == runner.VARIANTS[variant]
        assert columns == ["base"] + ([] if variant == "mae_full" else sequence_columns())
        weights_lock = json.loads((output_dir / "weights_lock.json").read_text())
        assert weights_lock["weights_sha256"] == runner.research_package.file_hash(output_dir / "weights.json")
        evaluations.append((directory.name, variant, origin, start, end))
        if directory.name == "pilot":
            assert (start, end, origin) == (4, 5, True)
            assert kwargs["forecast_provider"] is not None
            forecasts = kwargs["forecast_provider"]()
            assert set(zip(forecasts.origin_gw, forecasts.GW)) == {(4, 4), (4, 5), (5, 5)}
            assert set(forecasts.predicted_total_points) == ({2.} if variant == "blend" else {1.})
            score = points[variant]
        else:
            assert (start, end) == (6, 8) and "forecast_provider" not in kwargs
            forecasts = _forecasts(start, end, 2., origin)
            score = 200.
        if variant == "mae_full":
            assert kwargs["snapshot_builder"] is None and kwargs["extra_player_features"] == ()
        else:
            assert kwargs["snapshot_builder"] is runner.cnn_research.build_sequence_snapshot
            assert kwargs["extra_player_features"] == tuple(sequence_columns())
        prediction_name, squad_name = f"{variant}_{origin}.predictions.csv", f"{variant}_{origin}.squads.csv"
        forecasts.to_csv(directory / prediction_name, index=False)
        pd.DataFrame({"gameweek": range(start, end + 1), "actual_total_points": 3.}).to_csv(directory / squad_name, index=False)
        return {"total_points": score, "predictions": prediction_name, "squads": squad_name,
                "prediction_sha256": runner.research_package.file_hash(directory / prediction_name),
                "squad_sha256": runner.research_package.file_hash(directory / squad_name)}
    monkeypatch.setattr(runner.research_package, "evaluate_variant", evaluate)

    def compare(candidate, baseline, **kwargs):
        assert Path(candidate).is_file() and Path(baseline).is_file()
        assert kwargs == {"n_boot": 10000, "block_len": 3, "seed": 0}
        comparisons.append((candidate, baseline))
        return {"total_diff": points["blend"] - points["rmse_sequence"]}
    monkeypatch.setattr(runner.research_package.compare_backtests, "compare", compare)
    return SimpleNamespace(dataset=dataset, source_dir=source_dir, out=output_dir, points=points,
                           records=records, loaded=loaded, evaluations=evaluations, comparisons=comparisons,
                           current_source=current_source, old_source=old_source, member_paths=member_paths,
                           original_params=parameters)


def _run(study):
    return runner.run(study.dataset, study.out, source_dir=study.source_dir)


def test_cached_weight_split_pilot_lock_and_full_reference(study):
    result = _run(study)
    assert result["status"] == "complete" and result["promotion_eligible"] is False
    assert study.loaded == [5, 8]
    assert study.evaluations[:2] == [("pilot", "rmse_sequence", True, 4, 5), ("pilot", "blend", True, 4, 5)]
    assert len(study.evaluations) == 8 and len(study.comparisons) == 5
    assert set(result["evaluation"]) == {"blend", "rmse_sequence", "mae_full"}
    weights = json.loads((study.out / "weights.json").read_text())
    assert weights["window"] == [2, 3]
    assert weights["weights"] == {pos: .5 for pos in config.ONFIELD_POSITIONS}
    lock = json.loads((study.out / "pilot_lock.json").read_text())
    assert lock["pilot_summary_sha256"] == runner.research_package.file_hash(study.out / "pilot_summary.json")
    assert json.loads((study.out / "pilot_summary.json").read_text())["evaluation"] == {}
    assert study.records[-1][1]["summary_sha256"] == runner.research_package.file_hash(study.out / "summary.json")
    assert result["elapsed_seconds"] >= 0


@pytest.mark.parametrize("points", [100., 99.])
def test_pilot_tie_or_loss_skips_reference(study, points):
    study.points["blend"] = points
    result = _run(study)
    assert result["status"] == "pilot_rejected" and result["advances"] is False
    assert study.loaded == [5] and len(study.evaluations) == 2
    assert len(study.comparisons) == 1
    assert result["evaluation"] == {} and not (study.out / "evaluation").exists()


def test_old_archive_is_admitted_independently_of_current_source(study):
    assert study.old_source.read_bytes() != study.current_source.read_bytes()
    _run(study)
    registration = json.loads((study.out / "registration.json").read_text())
    archived = study.out / "source_archive" / "fpl" / "current.py"
    assert archived.read_bytes() == study.current_source.read_bytes()
    assert registration["source_registration"]["code_sha256"]["fpl/current.py"] == runner.research_package.file_hash(study.old_source)
    assert registration["parameters"]["sequence_cnn"] == study.original_params["sequence_cnn"]
    assert registration["parameters"]["catboost_rmse"] == study.original_params["catboost_rmse"]
    for position in config.ONFIELD_POSITIONS:
        settings = registration["parameters"][runner.BLEND_NAME][position][runner.BLEND_NAME]
        assert settings["cnn_weight"] == .5
        assert settings["cnn_params"] == study.original_params["sequence_cnn"][position]["sequence_cnn"]
        assert settings["tree_params"] == study.original_params["catboost_rmse"][position]["catboost_rmse"]
    assert any("not untouched independent validation" in value for value in registration["limitations"])


@pytest.mark.parametrize("artifact", ["old_source", "cached_forecast", "verification", "selection_summary"])
def test_cached_provenance_tampering_refused_before_fit(study, artifact):
    path = {"old_source": study.old_source, "cached_forecast": study.member_paths["cnn_sequence"],
            "verification": study.source_dir / "verification.json", "selection_summary": study.source_dir / "selection_summary.json"}[artifact]
    if artifact == "verification":
        payload = json.loads(path.read_text())
        payload["summary_sha256"] = "0" * 64
        _write(path, payload)
    elif artifact == "selection_summary":
        payload = json.loads(path.read_text())
        payload["selection"]["cnn_sequence"]["total_points"] = 999
        _write(path, payload)
    else:
        path.write_text("tampered bytes\n")
    with pytest.raises(ValueError):
        _run(study)
    assert not study.evaluations and study.loaded == []


@pytest.mark.parametrize("drift", ["current_source", "dataset", "cached_forecast"])
def test_phase_drift_incomplete_checkpoint(study, monkeypatch, drift):
    original = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = {"current_source": study.current_source, "dataset": study.dataset,
                "cached_forecast": study.member_paths["cnn_sequence"]}[drift]
        path.write_text("changed after evaluation\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="changed"):
        _run(study)
    summary = json.loads((study.out / "summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["elapsed_seconds"] >= 0
    assert len(study.evaluations) == 1 and study.loaded == [5]
    assert not (study.out / "pilot_lock.json").exists()


def test_paired_pilot_metadata_refusal_prevents_advancement(study, monkeypatch):
    original = runner.research_package.evaluate_variant
    def mismatched(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == "blend":
            path = args[8] / result["predictions"]
            frame = pd.read_csv(path)
            frame.loc[0, "value"] += 1
            frame.to_csv(path, index=False)
            result["prediction_sha256"] = runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", mismatched)
    with pytest.raises(ValueError, match="differ in universe"):
        _run(study)
    assert study.loaded == [5] and not study.comparisons
    assert not (study.out / "pilot_lock.json").exists()


def test_no_overwriting_completed_study(study):
    _run(study)
    preserved = {path: path.read_bytes() for path in study.out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already exists"):
        _run(study)
    assert {path: path.read_bytes() for path in preserved} == preserved
    assert study.loaded == [5, 8]


def test_final_drift_changes_completion_to_incomplete(study, monkeypatch):
    log = runner.experiment.log_result
    def changed(name, params, metrics):
        log(name, params, metrics)
        if name == "blend_research_completed":
            study.current_source.write_text("final drift\n")
    monkeypatch.setattr(runner.experiment, "log_result", changed)
    with pytest.raises(ValueError, match="source changed"):
        _run(study)
    assert json.loads((study.out / "summary.json").read_text())["status"] == "incomplete"
    assert study.records[-1][0] == "blend_research_incomplete"


def test_cli_portable_source_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda dataset, output, **kwargs: calls.append((dataset, output, kwargs)))
    runner.main(["--dataset", "frozen.csv", "--output-dir", "new", "--source-dir", "old"])
    assert calls == [(Path("frozen.csv"), Path("new"), {"source_dir": Path("old")})]


@pytest.mark.parametrize("artifact,phase", [("weights.json", "pilot"), ("pilot_lock.json", "evaluation")])
def test_locked_outputs_cannot_mutate_between_evaluations(study, monkeypatch, artifact, phase):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[8].name == phase:
            path = study.out / artifact
            payload = json.loads(path.read_text())
            payload["changed_after_lock"] = True
            _write(path, payload)
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="immutable blend output changed"):
        _run(study)
    summary = json.loads((study.out / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert len(study.evaluations) == (1 if phase == "pilot" else 3)
    assert study.records[-1][0] == "blend_research_incomplete"


@pytest.mark.parametrize("artifact", ["predictions", "squads"])
def test_scored_outputs_remain_hash_bound_before_pilot_ci(study, monkeypatch, artifact):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[3] == "blend" and args[8].name == "pilot":
            control = args[8] / f"rmse_sequence_True.{artifact}.csv"
            control.write_text(control.read_text() + "\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="immutable blend output changed"):
        _run(study)
    assert not study.comparisons and not (study.out / "pilot_lock.json").exists()
    assert json.loads((study.out / "summary.json").read_text())["status"] == "incomplete"
