"""Mocked fixed-fit CNN selection, origin adapters and immutable run admission."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpl import config
from fpl.model import cnn_research as runner
from fpl.model.sequence_features import sequence_columns


def _raw(end):
    frame = pd.DataFrame({"player_id": [1] * end, "GW_global": range(1, end + 1),
                          "fixture": range(10, 10 + end), "base": np.arange(end, dtype=float)})
    for stat in config.CNN_SEQUENCE_STATS:
        frame[stat] = np.arange(end, dtype=float) + 1
    return frame


def test_attach_sequences_preserves_feature_rows_and_duplicate_indices():
    raw = _raw(3)
    frame = raw.iloc[[2, 0, 1]][["player_id", "GW_global", "base"]].copy()
    frame.index = [8, 8, 4]
    result = runner.attach_sequences(raw, frame)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.iloc[0]["cnn_total_points_t7"] == 2
    assert np.isnan(result.iloc[1]["cnn_total_points_t7"])
    with pytest.raises(ValueError, match="already contains"):
        runner.attach_sequences(raw, result)
    frame.iloc[0, frame.columns.get_loc("player_id")] = 99
    with pytest.raises(ValueError, match="lack matching"):
        runner.attach_sequences(raw, frame)


def test_snapshot_appends_exact_ids_without_fetch_or_extra_players(monkeypatch):
    raw = _raw(3)
    raw = pd.concat([raw, _raw(2).assign(player_id=2)], ignore_index=True)
    base = pd.DataFrame({"player_id": [2, 1], "base": [20., 10.]}, index=[7, 7])
    monkeypatch.setattr(runner, "build_live_snapshot", lambda hist: base)
    result = runner.build_sequence_snapshot(raw)
    pd.testing.assert_frame_equal(result[base.columns], base)
    assert result.iloc[0]["cnn_total_points_t7"] == 2
    assert result.iloc[1]["cnn_total_points_t7"] == 3
    assert len(result.columns) == 2 + len(sequence_columns())


@pytest.fixture
def package_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "fpl" / "models")
    source = tmp_path / "fpl" / "fake.py"
    source.parent.mkdir()
    source.write_text("registered source bytes\n")
    artifact = tmp_path / "fpl" / "models" / "tuned.json"
    artifact.parent.mkdir()
    artifact.write_text('{"fixed": true}\n')
    dataset = tmp_path / "frozen.csv"
    dataset.write_bytes(b"verified frozen fixture bytes\n")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    cutoffs = {"discovery_max_gw": 1, "selection_min_gw": 2, "selection_max_gw": 4,
               "backtest_min_gw": 5, "backtest_max_gw": 7}
    monkeypatch.setattr(runner, "protocol_cutoffs", lambda: cutoffs)
    monkeypatch.setattr(config, "CNN_RETRAIN_EVERY", 100)
    monkeypatch.setattr(runner.experiment, "_library_versions", lambda: {"fake": "1"})
    monkeypatch.setattr(runner.models, "EXPLORATORY_SPECS", {
        "sequence_cnn": SimpleNamespace(default_params=lambda seed: {"epochs": 2, "seed": seed})})
    records, loaded, evaluated, comparisons = [], [], [], []
    scores = {"mae_full": 100, "mae_sequence": 110, "rmse_sequence": 105, "cnn_sequence": 120}
    out = tmp_path / "run"

    def log(*args):
        records.append(args)
    monkeypatch.setattr(runner.experiment, "log_result", log)

    def resolve(mode, seed):
        assert (mode, seed) == ("selection", 0)
        params = {model: {pos: {model: {"iterations": 3, "random_seed": seed}}
                          for pos in config.ONFIELD_POSITIONS}
                  for model in ("catboost_mae", "catboost_rmse")}
        return params, {str(artifact.relative_to(tmp_path)): runner.research_package.file_hash(artifact)}
    monkeypatch.setattr(runner.research_package, "resolve_parameters", resolve)

    def load(end, path):
        assert path == dataset
        if end == 7:
            lock = json.loads((out / "selection_lock.json").read_text())
            assert lock["cnn_advances"] is True
            assert runner.research_package.file_hash(out / "selection_summary.json") == lock["selection_summary_sha256"]
        loaded.append(end)
        raw = _raw(end)
        raw.attrs["frozen_dataset_sha256"] = digest
        return raw, raw.copy(), ["base"]
    monkeypatch.setattr(runner, "load_research_features", load)

    def evaluate(raw, frame, columns, variant, parameters, start, end, origin, directory, seed, **kwargs):
        assert records[0][0] == "cnn_research_registration"
        assert records[0][2]["status"] == "REGISTERED"
        assert kwargs["model_name"] == runner.MODEL_NAMES[variant]
        assert kwargs["retrain_every"] == 100
        assert seed == 0
        assert list(columns) == ["base"] + ([] if variant == "mae_full" else sequence_columns())
        if variant == "mae_full":
            assert kwargs["snapshot_builder"] is None and kwargs["extra_player_features"] == ()
        else:
            assert kwargs["snapshot_builder"] is runner.build_sequence_snapshot
            assert kwargs["extra_player_features"] == tuple(sequence_columns())
        name = kwargs["model_name"]
        assert set(parameters[name]) == set(config.ONFIELD_POSITIONS)
        assert int(raw.GW_global.max()) == end
        evaluated.append((directory.name, variant, origin, start, end))
        if directory.name == "selection":
            assert origin is True and (start, end) == (2, 4)
            points = scores[variant]
        else:
            lock = json.loads((out / "selection_lock.json").read_text())
            assert variant in lock["finalists"]
            assert lock["settings"][variant]["parameters"] == parameters[name]
            assert (start, end) == (5, 7)
            points = 99 if variant == "cnn_sequence" else 101
        forecasts = pd.DataFrame({"player_id": [1, 1, 1], "position": "MID", "GW": range(start, end + 1),
                                  "fixture": range(start * 10, start * 10 + 3), "value": 50.,
                                  "actual_total_points": [1., 2., 3.], "minutes": 90.,
                                  "predicted_total_points": 4. if variant == "cnn_sequence" else 3.})
        if origin:
            forecasts["origin_gw"] = forecasts.GW
        prediction_name = f"{variant}_{origin}.predictions.csv"
        forecasts.to_csv(directory / prediction_name, index=False)
        squad_name = f"{variant}_{origin}.squads.csv"
        pd.DataFrame({"gameweek": range(start, end + 1), "actual_total_points": 3.}).to_csv(directory / squad_name, index=False)
        pd.DataFrame({"gameweek": range(start, end + 1), "actual_total_points": points / 3}).to_csv(directory / squad_name, index=False)
        return {"total_points": points, "predictions": prediction_name, "squads": squad_name,
                "prediction_sha256": runner.research_package.file_hash(directory / prediction_name),
                "squad_sha256": runner.research_package.file_hash(directory / squad_name)}
    monkeypatch.setattr(runner.research_package, "evaluate_variant", evaluate)

    def compare(candidate, control, **kwargs):
        assert Path(candidate).is_file() and Path(control).is_file()
        assert kwargs == {"n_boot": 10000, "block_len": 3, "seed": 0}
        comparisons.append((candidate, control))
        return {"total_diff": -2.}
    monkeypatch.setattr(runner.research_package.compare_backtests, "compare", compare)
    return SimpleNamespace(dataset=dataset, out=out, source=source, artifact=artifact,
                           scores=scores, loaded=loaded, evaluated=evaluated,
                           comparisons=comparisons, records=records)


def test_selection_first_lock_and_reference_forwarding(package_mock):
    p = package_mock
    result = runner.run(p.dataset, p.out)
    assert p.loaded == [4, 7]
    assert result["status"] == "complete" and result["promotion_eligible"] is False
    assert np.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= 0
    assert result["tree_best"] == "mae_sequence"
    assert [item[1] for item in p.evaluated[:4]] == list(runner.MODEL_NAMES)
    assert len(p.evaluated) == 10
    assert len(p.comparisons) == 4
    assert set(result["evaluation"]) == {"mae_full", "mae_sequence", "cnn_sequence"}
    lock = json.loads((p.out / "selection_lock.json").read_text())
    assert lock["finalists"] == ["mae_full", "mae_sequence", "cnn_sequence"]
    assert runner.research_package.file_hash(p.out / "selection_summary.json") == lock["selection_summary_sha256"]
    selection = json.loads((p.out / "selection_summary.json").read_text())
    assert selection["evaluation"] == {} and selection["status"] == "running"
    assert selection["selection"] == result["selection"]
    assert p.records[-1][1]["summary_sha256"] == runner.research_package.file_hash(p.out / "summary.json")


@pytest.mark.parametrize("cnn_points", [110, 109])
def test_cnn_tie_or_loss_rejects_without_reference_load(package_mock, cnn_points):
    p = package_mock
    p.scores["cnn_sequence"] = cnn_points
    result = runner.run(p.dataset, p.out)
    assert result["status"] == "selection_rejected"
    assert np.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= 0
    assert p.loaded == [4]
    assert len(p.evaluated) == 4 and not p.comparisons
    assert result["evaluation"] == {}
    assert not (p.out / "evaluation").exists()
    lock = json.loads((p.out / "selection_lock.json").read_text())
    assert lock["cnn_advances"] is False and lock["finalists"] == []


def test_tree_tie_prefers_simpler_predeclared_order(package_mock):
    p = package_mock
    p.scores.update(mae_full=110, mae_sequence=110, rmse_sequence=110)
    result = runner.run(p.dataset, p.out)
    assert result["tree_best"] == "mae_full"
    assert len(p.evaluated) == 8
    assert set(result["evaluation"]) == {"mae_full", "cnn_sequence"}
    assert len(p.comparisons) == 4


def test_source_archive_and_exact_registration(package_mock):
    p = package_mock
    runner.run(p.dataset, p.out)
    registration = json.loads((p.out / "registration.json").read_text())
    for relative, digest in registration["code_sha256"].items():
        archived = p.out / registration["source_archive"] / relative
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
        assert archived.read_bytes() == (config.ROOT / relative).read_bytes()
    assert registration["tuned_artifact_sha256"] == {str(p.artifact.relative_to(config.ROOT)): runner.research_package.file_hash(p.artifact)}
    assert registration["train_max_gw"] == {"selection": 1, "reference": 4}
    assert registration["base_feature_columns"] == ["base"]
    assert registration["sequence_feature_columns"] == sequence_columns()
    assert registration["sequence_length"] == config.CNN_SEQUENCE_LENGTH
    assert registration["sequence_stats"] == list(config.CNN_SEQUENCE_STATS)
    assert registration["architecture"] == {
        "convolution_layers": 2, "kernel_size": 3, "padding": 1,
        "channels": config.CNN_CHANNELS, "hidden": config.CNN_HIDDEN,
        "activation": "relu", "pooling": "presence_masked_mean",
        "sequence_missing_flags": True, "context_missing_flags": True}
    assert registration["training_objective"] == {
        "loss": "MSE", "target_standardization": "training_only_mean_std",
        "preprocessing": "training_only_imputation_and_scaling"}
    assert registration["points_accounting"] == "net_after_transfer_hits"
    assert registration["retrain_every"] == 100
    assert registration["runtime_policy"] == {"cnn_backend": "isolated_cpu_worker", "device": "cpu", "pytabkit_attempts": 0}


def test_existing_outputs_never_overwritten(package_mock):
    p = package_mock
    runner.run(p.dataset, p.out)
    preserved = {path: path.read_bytes() for path in p.out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already exists"):
        runner.run(p.dataset, p.out)
    assert {path: path.read_bytes() for path in preserved} == preserved
    assert p.loaded == [4, 7]


@pytest.mark.parametrize("drift", ["source", "artifact", "dataset"])
def test_fit_phase_drift_preserves_incomplete_checkpoint(package_mock, monkeypatch, drift):
    p = package_mock
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        {"source": p.source, "artifact": p.artifact, "dataset": p.dataset}[drift].write_text("changed\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="changed"):
        runner.run(p.dataset, p.out)
    assert len(p.evaluated) == 1 and p.loaded == [4]
    summary = json.loads((p.out / "summary.json").read_text())
    assert summary["status"] == "incomplete" and "mae_full" in summary["selection"]
    assert np.isfinite(summary["elapsed_seconds"]) and summary["elapsed_seconds"] >= 0
    assert p.records[-1][0] == "cnn_research_incomplete"


def test_final_source_drift_not_silently_completed(package_mock, monkeypatch):
    p = package_mock
    log = runner.experiment.log_result
    def changed(name, params, metrics):
        log(name, params, metrics)
        if name == "cnn_research_completed":
            p.source.write_text("changed at completion\n")
    monkeypatch.setattr(runner.experiment, "log_result", changed)
    with pytest.raises(ValueError, match="source changed"):
        runner.run(p.dataset, p.out)
    assert json.loads((p.out / "summary.json").read_text())["status"] == "incomplete"
    assert p.records[-1][0] == "cnn_research_incomplete"


def test_refit_interval_must_leave_each_window_fixed(package_mock, monkeypatch):
    monkeypatch.setattr(config, "CNN_RETRAIN_EVERY", 3)
    with pytest.raises(ValueError, match="exceed both windows"):
        runner.run(package_mock.dataset, package_mock.out)
    assert package_mock.loaded == []


def test_cli_forwards_dataset_and_output(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "run", lambda dataset, out: seen.append((dataset, out)))
    runner.main(["--dataset", "frozen.csv", "--output-dir", "new-run"])
    assert seen == [(Path("frozen.csv"), Path("new-run"))]


@pytest.mark.parametrize("change", ["value", "missing_row"])
def test_selection_metadata_or_pool_mismatch_fails_before_ranking_and_lock(package_mock, monkeypatch, change):
    p = package_mock
    evaluate = runner.research_package.evaluate_variant
    def mismatched(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[3] == "mae_sequence":
            path = args[8] / result["predictions"]
            frame = pd.read_csv(path)
            if change == "value":
                frame.loc[0, "value"] += 1
            else:
                frame = frame.iloc[1:]
            frame.to_csv(path, index=False)
            result["prediction_sha256"] = runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", mismatched)
    with pytest.raises(ValueError, match="differ in universe"):
        runner.run(p.dataset, p.out)
    assert p.loaded == [4] and len(p.evaluated) == 2
    assert not (p.out / "selection_lock.json").exists()
    assert not (p.out / "selection_summary.json").exists()
    summary = json.loads((p.out / "summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["elapsed_seconds"] >= 0
    assert summary["evaluation"] == {} and not p.comparisons


@pytest.mark.parametrize("artifact,phase", [("registration.json", "selection"),
                                            ("selection_lock.json", "evaluation"),
                                            ("selection_summary.json", "evaluation")])
def test_cnn_immutable_metadata_mutation_fails_between_scores(package_mock, monkeypatch, artifact, phase):
    p = package_mock
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[8].name == phase:
            path = p.out / artifact
            payload = json.loads(path.read_text())
            payload["mutated_after_lock"] = True
            path.write_text(json.dumps(payload))
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="immutable CNN output changed"):
        runner.run(p.dataset, p.out)
    assert json.loads((p.out / "summary.json").read_text())["status"] == "incomplete"
    assert len(p.evaluated) == (1 if phase == "selection" else 5)


@pytest.mark.parametrize("artifact", ["predictions", "squads"])
def test_cnn_completed_selection_outputs_cannot_mutate_before_ranking(package_mock, monkeypatch, artifact):
    p = package_mock
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[3] == "mae_sequence" and args[8].name == "selection":
            path = args[8] / f"mae_full_True.{artifact}.csv"
            path.write_text(path.read_text() + "\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="immutable CNN output changed"):
        runner.run(p.dataset, p.out)
    assert not (p.out / "selection_lock.json").exists()
    assert p.loaded == [4] and not p.comparisons
    assert json.loads((p.out / "summary.json").read_text())["status"] == "incomplete"
