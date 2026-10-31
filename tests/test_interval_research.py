"""Mechanical chronology/provenance tests, not real model evaluation evidence."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fpl import config
from fpl.model import interval_research as runner


def frame():
    rows = []
    for position_index, position in enumerate(config.ONFIELD_POSITIONS):
        for gw in range(1, 25):
            rows.append({"player_id": position_index + 1, "position": position, "GW_global": gw,
                         "fixture": gw * 10, "total_points": 2., "minutes": 90., "seen_gw": float(gw)})
        if position == "MID":
            rows.append({"player_id": position_index + 1, "position": position, "GW_global": 14,
                         "fixture": 141, "total_points": 3., "minutes": 0., "seen_gw": 14.})
    return pd.DataFrame(rows)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "fpl").mkdir(parents=True)
    source = root / "fpl" / "sample.py"
    source.write_text("# frozen test source\n")
    dataset = tmp_path / "dataset.csv"
    dataset.write_bytes(b"synthetic verified input")
    frozen_manifest = tmp_path / "frozen_manifest.json"
    frozen_manifest.write_text('{"max_gw_global": 231}')
    monkeypatch.setattr(runner.provenance, "FROZEN_MANIFEST_PATH", frozen_manifest)
    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(runner, "protocol_cutoffs", lambda: {
        "discovery_max_gw": 12, "selection_min_gw": 13, "selection_max_gw": 20,
        "backtest_min_gw": 21, "backtest_max_gw": 24})
    monkeypatch.setattr(config, "INTERVAL_CALIBRATION_WEEKS", 8)
    monkeypatch.setattr(config, "INTERVAL_ALPHA", .2)
    monkeypatch.setattr(config, "INTERVAL_SEED", 0)
    state = {"loads": [], "fits": [], "logs": [], "frame": frame(), "dataset": dataset,
             "output": tmp_path / "run", "source": source, "frozen_manifest": frozen_manifest}
    def loader(cap, path):
        assert path == dataset
        state["loads"].append(cap)
        admitted = state["frame"].loc[state["frame"].GW_global <= cap].copy()
        admitted.attrs["frozen_dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
        return admitted.copy(), admitted, ["seen_gw"]
    monkeypatch.setattr(runner, "load_research_features", loader)
    monkeypatch.setattr(runner.experiment, "_library_versions", lambda: {"mock": "1"})
    monkeypatch.setattr(runner.experiment, "log_result", lambda name, params, metrics: state["logs"].append((name, params, metrics)))
    class Estimator:
        def __init__(self, **params):
            self.params = params
        def fit(self, X, y):
            assert state["logs"][0][0] == "interval_research_registration"
            assert (state["output"] / "registration.json").exists()
            assert (state["output"] / "evaluated_sources" / "fpl" / "sample.py").exists()
            state["fits"].append((X.copy(), y.copy(), self.params.copy()))
            return self
        def predict(self, X):
            # Deliberately crossed raw predictions to test rowwise sorting.
            return np.full(len(X), {.1: 8., .5: 4., .9: 0.}[self.params["alpha"]])
    monkeypatch.setattr(runner.models.lgb, "LGBMRegressor", Estimator)
    return state


def test_registered_caps_splits_and_models_with_unchanged_median(sandbox):
    summary = runner.run(sandbox["dataset"], sandbox["output"])
    assert summary["status"] == "complete" and not summary["promotion_eligible"]
    assert sandbox["loads"] == [20, 24]
    assert len(sandbox["fits"]) == 24
    assert [int(X.seen_gw.max()) for X, _, _ in sandbox["fits"]] == [4] * 12 + [12] * 12
    for X, y, params in sandbox["fits"]:
        assert (y == 2).all() and params["objective"] == "quantile" and "metric" not in params
        assert params["random_state"] == 0 and params["n_jobs"] == config.RESEARCH_MODEL_THREADS
        assert params["deterministic"] and params["force_col_wise"]
    registration = json.loads((sandbox["output"] / "registration.json").read_text())
    assert registration["feature_columns"] == ["seen_gw"]
    assert registration["splits"]["selection"] == {
        "train_max_gw": 4, "calibration_min_gw": 5, "calibration_max_gw": 12,
        "evaluation_min_gw": 13, "evaluation_max_gw": 20}
    assert registration["splits"]["reference"]["calibration_max_gw"] == 20
    for phase in ("selection", "reference"):
        path = sandbox["output"] / phase
        cal = pd.read_csv(path / "calibration.csv")
        pred = pd.read_csv(path / "predictions.csv")
        assert cal.GW_global.min() == (5 if phase == "selection" else 13)
        assert cal.GW_global.max() == (12 if phase == "selection" else 20)
        assert (pred["median"] == 4 * pred.fixture_count).all()
        assert (pred.calibrated_lower <= pred["median"]).all()
        assert (pred["median"] <= pred.calibrated_upper).all()
        assert set(summary["phases"][phase]["metrics"]) == {"pooled", *config.ONFIELD_POSITIONS}
    complete = sandbox["logs"][-1]
    assert complete[0] == "interval_research_complete"
    assert complete[1]["summary_sha256"] == runner.file_hash(sandbox["output"] / "summary.json")


def test_calibration_has_no_same_phase_evaluation_label_leakage(sandbox):
    calls = []
    real_adjustment = runner.conformal_adjustment
    def capture(y, lower, upper, *, alpha):
        calls.append(np.asarray(y).copy())
        return real_adjustment(y, lower, upper, alpha=alpha)
    # Change only reference evaluation outcomes; training and earlier calibration stay fixed.
    sandbox["frame"].loc[sandbox["frame"].GW_global >= 21, "total_points"] = 9999.
    from unittest.mock import patch
    with patch.object(runner, "conformal_adjustment", capture):
        runner.run(sandbox["dataset"], sandbox["output"])
    assert len(calls) == 8
    assert all(np.max(values) < 9999 for values in calls)
    reference = pd.read_csv(sandbox["output"] / "reference" / "predictions.csv")
    assert (reference.actual_total_points == 9999).all()


def test_round_aggregation_sums_distinct_dgw_fixtures():
    fixtures = pd.DataFrame({"player_id": [1, 1], "position": ["MID", "MID"], "GW_global": [14, 14],
                             "fixture": [140, 141], "total_points": [2., 3.], "minutes": [90., 0.]})
    result = runner.aggregate_round_predictions(fixtures, np.array([[0., 4., 8.], [1., 5., 9.]]))
    assert len(result) == 1
    assert result.actual_total_points.iloc[0] == 5 and result.fixture_count.iloc[0] == 2
    assert result.minutes.iloc[0] == 90 and result["median"].iloc[0] == 9
    assert result.raw_lower.iloc[0] == 1 and result.raw_upper.iloc[0] == 17
    with pytest.raises(ValueError, match="duplicate fixture"):
        runner.aggregate_round_predictions(fixtures.assign(fixture=140), np.array([[0., 4., 8.]] * 2))


def test_existing_output_refused_without_mutation(sandbox):
    sandbox["output"].mkdir()
    marker = sandbox["output"] / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        runner.run(sandbox["dataset"], sandbox["output"])
    assert marker.read_text() == "preserve" and sandbox["loads"] == [] and sandbox["logs"] == []


def test_source_drift_after_final_phase_preserves_incomplete_and_outputs(sandbox, monkeypatch):
    original = runner._evaluate_phase
    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[-1].name == "reference":
            sandbox["source"].write_text("# drifted after evaluation\n")
        return result
    monkeypatch.setattr(runner, "_evaluate_phase", drift)
    with pytest.raises(ValueError, match="source changed"):
        runner.run(sandbox["dataset"], sandbox["output"])
    summary = json.loads((sandbox["output"] / "summary.json").read_text())
    assert summary["status"] == "incomplete" and "ValueError" in summary["error"]
    assert "selection" in summary["phases"]
    assert (sandbox["output"] / "reference" / "predictions.csv").exists()
    assert sandbox["logs"][-1][0] == "interval_research_incomplete"
    assert not any(name == "interval_research_complete" for name, _, _ in sandbox["logs"])


def test_loader_cap_violation_fails_before_fitting(sandbox, monkeypatch):
    def broken(cap, path):
        admitted = sandbox["frame"].copy()
        admitted.attrs["frozen_dataset_sha256"] = runner.file_hash(path)
        return admitted, admitted, ["seen_gw"]
    monkeypatch.setattr(runner, "load_research_features", broken)
    with pytest.raises(ValueError, match="phase cap"):
        runner.run(sandbox["dataset"], sandbox["output"])
    assert sandbox["fits"] == []
    assert json.loads((sandbox["output"] / "summary.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize('artifact', ['archive', 'registration', 'earlier_predictions', 'earlier_calibration', 'dataset'])
def test_immutable_artifact_mutation_during_reference_refuses_completion(sandbox, monkeypatch, artifact):
    evaluate = runner._evaluate_phase
    def changed(*args, **kwargs):
        if args[-1].name == 'reference':
            path = {
                'archive': sandbox['output'] / 'evaluated_sources/fpl/sample.py',
                'registration': sandbox['output'] / 'registration.json',
                'earlier_predictions': sandbox['output'] / 'selection/predictions.csv',
                'earlier_calibration': sandbox['output'] / 'selection/calibration.csv',
                'dataset': sandbox['dataset'],
            }[artifact]
            path.write_bytes(path.read_bytes() + b'\n')
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(runner, '_evaluate_phase', changed)
    with pytest.raises(ValueError, match='artifact changed|dataset changed'):
        runner.run(sandbox['dataset'], sandbox['output'])
    summary = json.loads((sandbox['output'] / 'summary.json').read_text())
    assert summary['status'] == 'incomplete'
    assert 'selection' in summary['phases'] and summary['immutable_output_sha256']
    assert not any(name == 'interval_research_complete' for name, _, _ in sandbox['logs'])


def test_completed_manifest_binds_registration_archive_and_all_phase_csvs(sandbox):
    summary = runner.run(sandbox['dataset'], sandbox['output'])
    output = sandbox['output']
    manifest = summary['immutable_output_sha256']
    expected = [output / 'registration.json', output / 'evaluated_sources/fpl/sample.py', sandbox['frozen_manifest']]
    expected += [output / phase / name for phase in ('selection', 'reference')
                 for name in ('calibration.csv', 'predictions.csv')]
    assert set(manifest) == {str(path.resolve()) for path in expected}
    assert summary['registration_sha256'] == manifest[str((output / 'registration.json').resolve())]
    registration = json.loads((output / 'registration.json').read_text())
    assert registration['input_sha256'] == {
        str(sandbox['frozen_manifest'].resolve()): runner.file_hash(sandbox['frozen_manifest'])}
    for path in expected:
        assert manifest[str(path.resolve())] == runner.file_hash(path)


def test_first_csv_is_bound_before_second_csv_write(sandbox, monkeypatch):
    original = pd.DataFrame.to_csv
    def changed(frame, path, *args, **kwargs):
        original(frame, path, *args, **kwargs)
        if str(path).endswith('selection/predictions.csv'):
            calibration = sandbox['output'] / 'selection/calibration.csv'
            calibration.write_bytes(calibration.read_bytes() + b'\n')
    monkeypatch.setattr(pd.DataFrame, 'to_csv', changed)
    with pytest.raises(ValueError, match='artifact changed'):
        runner.run(sandbox['dataset'], sandbox['output'])
    summary = json.loads((sandbox['output'] / 'summary.json').read_text())
    assert summary['status'] == 'incomplete' and not summary['phases']


def test_optional_phase_json_and_calibration_lock_are_sealed(sandbox, monkeypatch):
    evaluate = runner._evaluate_phase
    def changed(*args, **kwargs):
        directory = args[-1]
        if directory.name == 'reference':
            lock = sandbox['output'] / 'selection/calibration_lock.json'
            lock.write_text('{"changed":true}')
        result = evaluate(*args, **kwargs)
        for name in ('phase.json', 'calibration_lock.json'):
            (directory / name).write_text('{"fixed":true}')
        return result
    monkeypatch.setattr(runner, '_evaluate_phase', changed)
    with pytest.raises(ValueError, match='artifact changed'):
        runner.run(sandbox['dataset'], sandbox['output'])
    summary = json.loads((sandbox['output'] / 'summary.json').read_text())
    assert summary['status'] == 'incomplete'
    for name in ('phase.json', 'calibration_lock.json'):
        assert str((sandbox['output'] / 'selection' / name).resolve()) in summary['immutable_output_sha256']


def test_manifest_is_bound_before_first_load_and_drift_stops_before_next_phase(sandbox, monkeypatch):
    loader = runner.load_research_features
    def mutate(cap, path):
        result = loader(cap, path)
        sandbox['frozen_manifest'].write_text('changed frozen manifest')
        return result
    monkeypatch.setattr(runner, 'load_research_features', mutate)
    with pytest.raises(ValueError, match='Sealed interval artifact changed'):
        runner.run(sandbox['dataset'], sandbox['output'])
    assert sandbox['loads'] == [20] and not sandbox['fits']
    assert not (sandbox['output'] / 'reference').exists()
    summary = json.loads((sandbox['output'] / 'summary.json').read_text())
    assert summary['status'] == 'incomplete'
    assert str(sandbox['frozen_manifest'].resolve()) in summary['immutable_output_sha256']


def test_final_summary_mutation_by_completion_logger_is_incomplete(sandbox, monkeypatch):
    def mutate(name, params, metrics):
        sandbox['logs'].append((name, params, metrics))
        if name == 'interval_research_complete':
            assert params['summary_sha256'] == runner.file_hash(sandbox['output'] / 'summary.json')
            (sandbox['output'] / 'summary.json').write_text('tampered final summary')
    monkeypatch.setattr(runner.experiment, 'log_result', mutate)
    with pytest.raises(ValueError, match='Final interval summary changed'):
        runner.run(sandbox['dataset'], sandbox['output'])
    summary = json.loads((sandbox['output'] / 'summary.json').read_text())
    assert summary['status'] == 'incomplete' and set(summary['phases']) == {'selection', 'reference'}
    assert str((sandbox['output'] / 'summary.json').resolve()) not in summary['immutable_output_sha256']


@pytest.mark.parametrize('same_fixture', [True, False])
def test_position_conflict_cannot_split_one_player_round_into_two_intervals(same_fixture):
    frame = pd.DataFrame({'player_id': [1, 1], 'position': ['MID', 'FWD'], 'GW_global': [14, 14],
                          'fixture': [140, 140 if same_fixture else 141], 'total_points': [4., 5.], 'minutes': [90., 90.]})
    with pytest.raises(ValueError, match='conflicting position'):
        runner.aggregate_round_predictions(frame, np.array([[0., 4., 8.], [1., 5., 9.]]))
