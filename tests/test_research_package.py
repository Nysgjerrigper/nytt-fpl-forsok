"""Research registration, immutable controls, and temporal finalist boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl import config, features
from fpl.model import research_package as package
from fpl.model import tuning


def _columns():
    return ["ordinary", "minutes_prev", *features.FIXTURE_FEATURES,
            *features.MINUTES_FEATURES, *features.OPPONENT_FEATURES, *features.XP_FEATURES]


def test_ablation_families_are_disjoint_one_at_a_time():
    columns = _columns()
    variants = package.feature_variants(columns)
    assert variants["mae_full"] == variants["rmse_full"] == columns
    removed = []
    for family in config.RESEARCH_ABLATION_FAMILIES:
        reduced = variants[f"mae_without_{family}"]
        drop = set(columns) - set(reduced)
        assert drop
        assert reduced == [column for column in columns if column not in drop]
        assert "ordinary" in reduced
        removed.append(drop)
    assert sum(map(len, removed)) == len(set().union(*removed))
    assert "minutes_prev" not in variants["mae_without_minutes"]


@pytest.mark.parametrize("columns", [[], ["ordinary", "ordinary"], ["ordinary", features.TARGET_COL]])
def test_invalid_features_rejected(columns):
    with pytest.raises(ValueError, match="unique, nonempty"):
        package.feature_variants(columns)


def test_missing_family_cannot_claim_ablation():
    with pytest.raises(ValueError, match="absent"):
        package.feature_variants(["ordinary"])


def test_defaults_bypass_ambient_parameters(monkeypatch):
    monkeypatch.setattr(package.models, "_tuned_params", lambda *args: pytest.fail("ambient params read"))
    monkeypatch.setattr(package, "load_validated_params", lambda *args, **kwargs: pytest.fail("artifact read"))
    resolved, hashes = package.resolve_parameters("defaults", 2)
    assert hashes == {}
    for model, by_position in resolved.items():
        assert set(by_position) == set(config.ONFIELD_POSITIONS)
        for settings in by_position.values():
            params = settings[model]
            assert params["random_seed"] == 2
            assert params["thread_count"] == config.RESEARCH_MODEL_THREADS
            assert params["allow_writing_files"] is False
    assert resolved["catboost_mae"]["MID"]["catboost_mae"]["loss_function"] == "MAE"
    assert resolved["catboost_rmse"]["MID"]["catboost_rmse"]["loss_function"] == "RMSE"


@pytest.mark.parametrize("mode,seed", [("ambient", 0), ("defaults", 3)])
def test_invalid_parameter_registration(mode, seed):
    with pytest.raises(ValueError):
        package.resolve_parameters(mode, seed)


def test_selection_artifacts_hashed_validated_and_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    cutoff = package.protocol_cutoffs()["discovery_max_gw"]
    paths = []
    for name in ("catboost_mae", "catboost_rmse"):
        for position in config.ONFIELD_POSITIONS:
            paths.append(tuning.save_best_params(position, name, {"iterations": 7, "random_seed": 0},
                                                 train_max_gw=cutoff, seed=0, stage="selection"))
    before = {path: path.read_bytes() for path in paths}
    resolved, hashes = package.resolve_parameters("selection", 1)
    assert {path: path.read_bytes() for path in paths} == before
    assert hashes == {str(path.relative_to(tmp_path)): hashlib.sha256(content).hexdigest()
                      for path, content in before.items()}
    assert resolved["catboost_mae"]["MID"]["catboost_mae"]["random_seed"] == 1
    payload = json.loads(paths[0].read_text())
    payload["iterations"] = 999
    paths[0].write_text(json.dumps(payload))
    tampered = paths[0].read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        package.resolve_parameters("selection", 0)
    assert paths[0].read_bytes() == tampered


def test_selection_uses_earlier_points_and_full_wins_ties():
    selection = {"mae_full": {"total_points": 100}, "rmse_full": {"total_points": 999},
                 "mae_without_xp": {"total_points": 100}, "mae_without_minutes": {"total_points": 90}}
    assert package.select_ablation(selection) is None
    selection["mae_without_minutes"]["total_points"] = 101
    assert package.select_ablation(selection) == "mae_without_minutes"
    selection["mae_without_xp"]["total_points"] = 102
    assert package.select_ablation(selection) == "mae_without_xp"


@pytest.fixture(autouse=True)
def frozen_manifest_fixture(tmp_path, monkeypatch):
    manifest=tmp_path/'frozen_manifest.json'
    manifest.write_text('{"max_gw_global":231}')
    monkeypatch.setattr(package.provenance,'FROZEN_MANIFEST_PATH',manifest)
    return manifest


def _forecasts(origin=False):
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)] if origin else [(1, 1), (2, 2), (3, 3)]
    frame = pd.DataFrame({"player_id": [1] * len(pairs), "position": ["MID"] * len(pairs),
                          "GW": [gw for _, gw in pairs], "predicted_total_points": 2.,
                          "actual_total_points": 3., "minutes": 90, "value": 50,
                          "fixture": [gw * 10 for _, gw in pairs]})
    if origin:
        frame["origin_gw"] = [o for o, _ in pairs]
    return frame


@pytest.mark.parametrize("origin", [False, True])
def test_forecast_schema_accepts_distinct_dgw_fixtures(origin):
    frame = _forecasts(origin)
    extra = frame.iloc[[0]].copy()
    extra["fixture"] = 11
    package.validate_forecasts(pd.concat([frame, extra]), 1, 3, origin)
    with pytest.raises(ValueError, match="duplicate fixture"):
        package.validate_forecasts(pd.concat([frame, frame.iloc[[0]]]), 1, 3, origin)


@pytest.mark.parametrize("column", ["player_id", "GW", "predicted_total_points", "actual_total_points", "minutes", "value"])
def test_nonfinite_forecasts_rejected(column):
    frame = _forecasts()
    frame.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        package.validate_forecasts(frame, 1, 3, False)


@pytest.mark.parametrize("column", ["player_id", "position", "GW", "predicted_total_points", "actual_total_points", "minutes", "value"])
def test_missing_forecast_inputs_rejected(column):
    with pytest.raises(ValueError, match="missing forecast"):
        package.validate_forecasts(_forecasts().drop(columns=column), 1, 3, False)


def test_origin_horizon_coverage_and_semantics():
    package.validate_forecasts(_forecasts(True), 1, 3, True)
    with pytest.raises(ValueError, match="horizon coverage"):
        package.validate_forecasts(_forecasts(True).drop(index=1), 1, 3, True)
    with pytest.raises(ValueError, match="origin semantics"):
        package.validate_forecasts(_forecasts(True), 1, 3, False)
    with pytest.raises(ValueError, match="origin semantics"):
        package.validate_forecasts(_forecasts(), 1, 3, True)


def test_missing_round_negative_minutes_and_unidentified_dgw_rejected():
    with pytest.raises(ValueError, match="incomplete forecast"):
        package.validate_forecasts(_forecasts().iloc[:2], 1, 3, False)
    frame = _forecasts()
    frame.loc[0, "minutes"] = -1
    with pytest.raises(ValueError, match="negative minutes"):
        package.validate_forecasts(frame, 1, 3, False)
    frame = _forecasts().drop(columns="fixture")
    with pytest.raises(ValueError, match="require fixture"):
        package.validate_forecasts(pd.concat([frame, frame.iloc[[0]]]), 1, 3, False)


def test_cached_replay_uses_same_admission_and_net_scoring_without_refitting(tmp_path, monkeypatch):
    monkeypatch.setattr(package.predict, "origin_based_predictions",
                        lambda *a, **k: pytest.fail("cached replay unexpectedly refitted"))
    monkeypatch.setattr(package, "prediction_diagnostics", lambda *a: {"cached": True})
    def optimize(args):
        pd.DataFrame({"gameweek": [1, 2, 3], "scoring_protocol": "autosubs_v2",
                      "actual_total_points": [10, 20, 30], "alpha": [0, 1, 0], "solver_status": "Optimal", "solver_solution_status": "Optimal Solution Found"}).to_csv(args.output, index=False)
        for _ in range(3):
            print("Status: Optimal (test)")
    monkeypatch.setattr(package.optimize, "run", optimize)
    result = package.evaluate_variant(pd.DataFrame(), pd.DataFrame(), ["form"], "cached",
                                      {"catboost_rmse": {}}, 1, 3, True, tmp_path, 0,
                                      model_name="catboost_rmse", forecast_provider=lambda: _forecasts(True))
    assert result["total_points"] == 56 and result["gross_points"] == 60
    assert result["transfer_hit_points"] == 4
    admission = json.loads((tmp_path / "cached_origin.admission.json").read_text())
    assert admission["cached_replay"] is True
    with pytest.raises(ValueError, match="callable"):
        package.evaluate_variant(pd.DataFrame(), pd.DataFrame(), ["form"], "invalid_cached",
                                 {"catboost_rmse": {}}, 1, 3, True, tmp_path, 0,
                                 model_name="catboost_rmse", forecast_provider=42)


def test_package_locks_finalists_before_reference_and_preserves_outputs(tmp_path, monkeypatch):
    cutoffs = {"discovery_max_gw": 1, "selection_min_gw": 2, "selection_max_gw": 4,
               "backtest_min_gw": 5, "backtest_max_gw": 7}
    monkeypatch.setattr(package, "protocol_cutoffs", lambda: cutoffs)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "fpl").mkdir()
    source = tmp_path / "fpl" / "fake.py"
    source.write_text("# preserved source\n")
    dataset = tmp_path / "frozen.csv"
    dataset.write_bytes(b"synthetic frozen bytes")
    dataset_digest = package.file_hash(dataset)
    monkeypatch.setattr(package.experiment, "_library_versions", lambda: {"fake": "1"})
    records = []
    monkeypatch.setattr(package.experiment, "log_result", lambda *args: records.append(args))
    monkeypatch.setattr(package, "resolve_parameters", lambda *args: ({"mock": {}}, {}))
    loaded = []

    def load(max_gw, dataset):
        if max_gw == 7:
            assert (out / "selection_lock.json").exists()
        loaded.append(max_gw)
        raw = pd.DataFrame({"GW_global": range(1, max_gw + 1)})
        raw.attrs["frozen_dataset_sha256"] = dataset_digest
        return raw, raw.copy(), _columns()

    monkeypatch.setattr(package, "load_research_features", load)
    out = tmp_path / "run"
    evaluated = []
    selection_scores = {"mae_full": 100, "rmse_full": 110, "mae_without_fixture": 95,
                        "mae_without_minutes": 120, "mae_without_opponent": 105, "mae_without_xp": 99}

    def evaluate(raw, frame, columns, variant, parameters, start, end, origin, directory, seed):
        registration = json.loads((out / "registration.json").read_text())
        assert registration["status"] == "registered"
        assert registration["promotion_eligible"] is False
        assert int(raw.GW_global.max()) == end
        evaluated.append((directory.name, variant, origin))
        if directory.name == "evaluation":
            lock = json.loads((out / "selection_lock.json").read_text())
            assert lock["finalists"] == ["mae_full", "rmse_full", "mae_without_minutes"]
            assert lock["selected_ablation"] == "mae_without_minutes"
            assert variant in lock["finalists"]
            points = 10000 if variant == "mae_full" else 1
        else:
            assert origin is True and (start, end) == (2, 4)
            points = selection_scores[variant]
        prediction_name = f"{variant}_{origin}.predictions.csv"
        _forecasts(origin).to_csv(directory / prediction_name, index=False)
        squad_name = f"{variant}_{origin}.csv"
        pd.DataFrame({"gameweek": [start], "actual_total_points": [points]}).to_csv(directory / squad_name, index=False)
        result = {"total_points": points, "squads": squad_name,
                  "predictions": prediction_name, "prediction_sha256": package.file_hash(directory / prediction_name),
                  "squad_sha256": package.file_hash(directory / squad_name)}
        (directory / f"{variant}_{origin}.result.json").write_text(json.dumps(result))
        (directory / f"{variant}_{origin}.admission.json").write_text("{}")
        (directory / f"{variant}_{origin}.log").write_text("Status: Optimal (mock)")
        return result

    monkeypatch.setattr(package, "evaluate_variant", evaluate)
    comparisons = []
    def compare(candidate, control, **kwargs):
        comparisons.append((candidate, control))
        return {"delta": -9999}
    monkeypatch.setattr(package.compare_backtests, "compare", compare)
    result = package.run_package(tmp_path / "frozen.csv", out, parameter_mode="defaults")
    assert loaded == [4, 7]
    assert result["status"] == "complete" and result["promotion_eligible"] is False
    assert result["selected_ablation"] == "mae_without_minutes"
    registration = json.loads((out / "registration.json").read_text())
    assert registration["source_archive"] == "evaluated_sources"
    assert (out / "evaluated_sources" / "fpl" / "fake.py").read_bytes() == source.read_bytes()
    assert result["registration_sha256"] == package.file_hash(out / "registration.json")
    assert all(package.file_hash(out / name) == digest for name, digest in result["immutable_output_sha256"].items())
    # The finalist lock must remain verifiable after the mutable run summary
    # acquires reference-window results and completion status.
    selection_summary = out / "selection_summary.json"
    assert selection_summary.is_file(), "preserve immutable selection bytes before evaluation"
    lock = json.loads((out / "selection_lock.json").read_text())
    assert hashlib.sha256(selection_summary.read_bytes()).hexdigest() == lock["selection_summary_sha256"]
    saved_selection = json.loads(selection_summary.read_text())
    assert saved_selection["selection"] == result["selection"]
    assert saved_selection["evaluation"] == {} and saved_selection["status"] == "running"
    assert len([row for row in evaluated if row[0] == "selection"]) == 6
    assert len([row for row in evaluated if row[0] == "evaluation"]) == 6
    assert len(comparisons) == 4
    assert [record[0] for record in records] == ["research_package_registration", "research_package_completed"]
    assert records[-1][1]["summary_sha256"] == hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
    preserved = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already exists"):
        package.run_package(tmp_path / "frozen.csv", out)
    assert {path: path.read_bytes() for path in preserved} == preserved
    assert loaded == [4, 7]


def test_explicit_ensemble_settings_reach_fitter_without_ambient_lookup(monkeypatch):
    from fpl.model import train
    calls = []
    sentinel = object()

    def fit(name, X, y, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(train.models, 'fit_model', fit)
    frame = pd.DataFrame({'position':['GK'], 'total_points':[2.], 'x':[1.], 'minutes':[90], 'GW_global':[3]})
    pinned = {'GK': {'catboost_mae': {'iterations':7}}}
    fitted = train.fit_position_ensembles(frame, ['x'], {'GK':{'catboost_mae':1.}}, params_by_pos=pinned, seed=2)
    assert fitted['GK'].members['catboost_mae'] is sentinel
    assert calls[0][1]['params'] == {**package.models.EXPERT_SPECS['catboost_mae'].default_params(2), 'iterations':7}
    assert calls[0][1]['seed'] == 2
    assert calls[0][1]['position'] == 'GK'
    assert calls[0][1]['gw'].tolist() == [3]


@pytest.mark.parametrize('pinned', [{}, {'GK':{}}, {'GK':{'catboost_mae':{}, 'unused':{}}}])
def test_explicit_ensemble_settings_require_exact_positive_members(pinned):
    from fpl.model import train
    frame = pd.DataFrame({'position':['GK'], 'total_points':[2.], 'x':[1.]})
    with pytest.raises(ValueError, match='recipe'):
        train.fit_position_ensembles(frame, ['x'], {'GK':{'catboost_mae':1., 'zero':0.}}, params_by_pos=pinned, seed=0)


@pytest.mark.parametrize("origin", [False, True])
def test_paired_forecasts_allow_prediction_changes_and_row_shuffle(tmp_path, origin):
    """Decision comparisons share the entire input universe, not forecast values."""
    control = _forecasts(origin)
    candidate = control.sample(frac=1, random_state=1).copy()
    candidate["predicted_total_points"] += 10
    control_path, candidate_path = tmp_path / "control.csv", tmp_path / "candidate.csv"
    control.to_csv(control_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    package.validate_paired_forecasts(candidate_path, control_path)


@pytest.mark.parametrize("origin", [False, True])
@pytest.mark.parametrize("change", ["value", "actual_total_points", "minutes", "fixture", "missing_row"])
def test_paired_forecasts_reject_different_scoring_inputs_or_universe(tmp_path, origin, change):
    control = _forecasts(origin)
    candidate = control.copy()
    if change == "missing_row":
        candidate = candidate.iloc[1:]
    else:
        candidate.loc[0, change] += 1
    control_path, candidate_path = tmp_path / "control.csv", tmp_path / "candidate.csv"
    control.to_csv(control_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    with pytest.raises(ValueError):
        package.validate_paired_forecasts(candidate_path, control_path)


def test_source_change_in_last_reference_pass_cannot_complete(tmp_path, monkeypatch):
    cutoffs = {'discovery_max_gw': 1, 'selection_min_gw': 2, 'selection_max_gw': 3,
               'backtest_min_gw': 4, 'backtest_max_gw': 5}
    monkeypatch.setattr(package, 'protocol_cutoffs', lambda: cutoffs)
    monkeypatch.setattr(package, 'feature_variants', lambda columns: {'mae_full': ['x'], 'rmse_full': ['x']})
    monkeypatch.setattr(package, 'resolve_parameters', lambda *args: ({}, {}))
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    (tmp_path / 'fpl').mkdir()
    source = tmp_path / 'fpl' / 'fake.py'
    source.write_text('# registered source\n')
    dataset = tmp_path / 'frozen.csv'
    dataset.write_bytes(b'synthetic frozen bytes')
    dataset_digest = package.file_hash(dataset)
    monkeypatch.setattr(package.experiment, '_library_versions', lambda: {})
    logs = []
    monkeypatch.setattr(package.experiment, 'log_result', lambda *args: logs.append(args))

    def load(max_gw, dataset):
        raw = pd.DataFrame({'GW_global': range(1, max_gw + 1)})
        raw.attrs['frozen_dataset_sha256'] = dataset_digest
        return raw, raw.copy(), ['x']

    def evaluate(raw, frame, columns, variant, parameters, start, end, origin, directory, seed):
        prediction_path = directory / f'{variant}_{origin}.predictions.csv'
        squad_path = directory / f'{variant}_{origin}.squads.csv'
        _forecasts(origin).to_csv(prediction_path, index=False)
        squad_path.write_text('gameweek,actual_total_points\n1,100\n')
        if directory.name == 'evaluation' and variant == 'rmse_full' and origin:
            source.write_text('# changed source\n')
        return {'total_points': 100, 'predictions': prediction_path.name, 'squads': squad_path.name,
                'prediction_sha256': package.file_hash(prediction_path), 'squad_sha256': package.file_hash(squad_path)}

    monkeypatch.setattr(package, 'load_research_features', load)
    monkeypatch.setattr(package, 'evaluate_variant', evaluate)
    monkeypatch.setattr(package, 'validate_paired_forecasts', lambda *args: None)
    monkeypatch.setattr(package.compare_backtests, 'compare', lambda *args, **kwargs: {})
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='source changed'):
        package.run_package(tmp_path / 'frozen.csv', output)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'incomplete' and not summary['promotion_eligible']
    assert [entry[0] for entry in logs] == ['research_package_registration', 'research_package_incomplete']


@pytest.fixture
def provenance_package(tmp_path, monkeypatch):
    """Use real immutable synthetic bytes and only mocked fitting/decision metrics."""
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    (tmp_path / 'fpl' / 'models').mkdir(parents=True)
    source = tmp_path / 'fpl' / 'fake.py'
    source.write_text('# registered synthetic source\n')
    tuned = tmp_path / 'fpl' / 'models' / 'pinned.json'
    tuned.write_text('{"iterations": 7}\n')
    dataset = tmp_path / 'frozen.csv'
    dataset.write_bytes(b'synthetic complete frozen snapshot')
    dataset_sha = package.file_hash(dataset)
    tuned_sha = package.file_hash(tuned)
    cutoffs = {'discovery_max_gw': 1, 'selection_min_gw': 2, 'selection_max_gw': 3,
               'backtest_min_gw': 4, 'backtest_max_gw': 5}
    monkeypatch.setattr(package, 'protocol_cutoffs', lambda: cutoffs)
    monkeypatch.setattr(package, 'feature_variants', lambda columns: {'mae_full': ['x'], 'rmse_full': ['x']})
    monkeypatch.setattr(package, 'resolve_parameters', lambda *args: ({}, {'fpl/models/pinned.json': tuned_sha}))
    monkeypatch.setattr(package.experiment, '_library_versions', lambda: {})
    state = {'source': source, 'tuned': tuned, 'dataset': dataset, 'output': tmp_path / 'run',
             'logs': [], 'evaluations': [], 'comparisons': [], 'after_evaluate': None, 'after_compare': None,
             'manifest': package.provenance.FROZEN_MANIFEST_PATH}
    monkeypatch.setattr(package.experiment, 'log_result', lambda *args: state['logs'].append(args))
    def load(max_gw, path):
        raw = pd.DataFrame({'GW_global': range(1, max_gw + 1)})
        raw.attrs['frozen_dataset_sha256'] = dataset_sha
        return raw, raw.copy(), ['x']
    def evaluate(raw, frame, columns, variant, parameters, start, end, origin, directory, seed):
        prediction = directory / f'{variant}_{origin}.predictions.csv'
        squad = directory / f'{variant}_{origin}.squads.csv'
        _forecasts(origin).to_csv(prediction, index=False)
        squad.write_text('gameweek,actual_total_points\n1,100\n')
        result = {'total_points': 100, 'predictions': prediction.name, 'squads': squad.name,
                  'prediction_sha256': package.file_hash(prediction), 'squad_sha256': package.file_hash(squad)}
        (directory / f'{variant}_{origin}.result.json').write_text(json.dumps(result))
        (directory / f'{variant}_{origin}.admission.json').write_text('{}')
        (directory / f'{variant}_{origin}.log').write_text('Status: Optimal (mock)')
        state['evaluations'].append((directory.name, variant, origin))
        if state['after_evaluate']:
            state['after_evaluate'](directory, variant, origin, result)
        return result
    def compare(candidate, control, **kwargs):
        state['comparisons'].append((candidate, control))
        if state['after_compare']:
            state['after_compare'](candidate, control)
        return {'delta': 0}
    monkeypatch.setattr(package, 'load_research_features', load)
    monkeypatch.setattr(package, 'evaluate_variant', evaluate)
    monkeypatch.setattr(package.compare_backtests, 'compare', compare)
    return state


@pytest.mark.parametrize('target', ['dataset', 'tuned', 'registration', 'source_archive', 'prediction', 'squad', 'result_json'])
def test_registered_bytes_and_prior_outputs_drift_after_checkpoint_fail_closed(provenance_package, target):
    state = provenance_package
    def mutate(directory, variant, origin, result):
        if directory.name != 'selection' or variant != 'rmse_full':
            return
        files = {'dataset': state['dataset'], 'tuned': state['tuned'],
                 'registration': state['output'] / 'registration.json',
                 'source_archive': state['output'] / 'evaluated_sources' / 'fpl' / 'fake.py',
                 'prediction': directory / 'mae_full_True.predictions.csv',
                 'squad': directory / 'mae_full_True.squads.csv',
                 'result_json': directory / 'mae_full_True.result.json'}
        files[target].write_bytes(files[target].read_bytes() + b'\nchanged bytes')
    state['after_evaluate'] = mutate
    with pytest.raises(ValueError, match='changed after registration'):
        package.run_package(state['dataset'], state['output'])
    saved = json.loads((state['output'] / 'summary.json').read_text())
    assert saved['status'] == 'incomplete' and 'mae_full' in saved['selection']
    assert not saved['evaluation'] and not state['comparisons']
    assert [entry[0] for entry in state['logs']] == ['research_package_registration', 'research_package_incomplete']
    assert (state['output'] / 'selection' / 'mae_full_True.predictions.csv').exists()


@pytest.mark.parametrize('target', ['selection_lock.json', 'selection_summary.json'])
def test_immutable_selection_bytes_drift_during_reference_fail_closed(provenance_package, target):
    state = provenance_package
    def mutate(directory, variant, origin, result):
        if directory.name == 'evaluation' and variant == 'mae_full' and not origin:
            file = state['output'] / target
            file.write_bytes(file.read_bytes() + b'\nmutation after lock')
    state['after_evaluate'] = mutate
    with pytest.raises(ValueError, match='immutable research output changed'):
        package.run_package(state['dataset'], state['output'])
    saved = json.loads((state['output'] / 'summary.json').read_text())
    assert saved['status'] == 'incomplete' and len(saved['selection']) == 2
    assert (state['output'] / 'evaluation' / 'mae_full_False.predictions.csv').exists()
    assert state['comparisons'] == []


@pytest.mark.parametrize('target', ['prediction', 'squad', 'dataset', 'tuned'])
def test_ci_checks_detect_mutation_during_comparison_and_cannot_complete(provenance_package, target):
    state = provenance_package
    def mutate(candidate, control):
        if target == 'dataset':
            file = state['dataset']
        elif target == 'tuned':
            file = state['tuned']
        elif target == 'squad':
            file = Path(candidate)
        else:
            file = state['output'] / 'evaluation' / 'mae_full_False.predictions.csv'
        file.write_bytes(file.read_bytes() + b'\nCI-time mutation')
    state['after_compare'] = mutate
    with pytest.raises(ValueError, match='changed after registration'):
        package.run_package(state['dataset'], state['output'])
    saved = json.loads((state['output'] / 'summary.json').read_text())
    assert saved['status'] == 'incomplete' and len(state['comparisons']) == 1
    assert len(saved['evaluation']) == 2
    assert not any(entry[0] == 'research_package_completed' for entry in state['logs'])


def test_results_missing_byte_hashes_cannot_rank_or_complete(provenance_package):
    state = provenance_package
    state['after_evaluate'] = lambda directory, variant, origin, result: result.pop('squad_sha256')
    with pytest.raises(ValueError, match='require prediction/squad byte hashes'):
        package.run_package(state['dataset'], state['output'])
    assert not (state['output'] / 'selection_lock.json').exists()
    assert state['comparisons'] == []

@pytest.mark.parametrize('policy,origin,catalog', [('bad',True,False),('origin-asof',True,False),('target-replay',False,True)])
def test_price_replay_invalid_adapter_settings(tmp_path,policy,origin,catalog):
    path=tmp_path/'prices.csv'
    path.write_text('prices')
    with pytest.raises(ValueError):
        package.evaluate_variant(pd.DataFrame(),pd.DataFrame(),['form'],'cached',{'catboost_mae':{}},
                                 1,3,origin,tmp_path,0,origin_price_catalog=path if catalog else None,
                                 origin_price_policy=policy)

@pytest.mark.parametrize('tamper', [False,True])
def test_price_replay_passes_sealed_catalog_and_rejects_mutation(tmp_path,monkeypatch,tamper):
    path=tmp_path/'prices.csv';path.write_text('sealed catalog bytes')
    monkeypatch.setattr(package,'prediction_diagnostics',lambda *a: [])
    def solve(args):
        assert args.origin_price_catalog==str(path) and args.origin_price_policy=='origin-asof'
        pd.DataFrame({'gameweek':[1,2,3],'scoring_protocol':'autosubs_v2',
                      'actual_total_points':[10,20,30],'alpha':[0,1,0],'solver_status':'Optimal','solver_solution_status':'Optimal Solution Found'}).to_csv(args.output,index=False)
        if tamper:path.write_text('changed')
        for _ in range(3):print('Status: Optimal (test)')
    monkeypatch.setattr(package.optimize,'run',solve)
    def run():return package.evaluate_variant(pd.DataFrame(),pd.DataFrame(),['form'],'cached',{'catboost_mae':{}},
                    1,3,True,tmp_path,0,forecast_provider=lambda:_forecasts(True),
                    origin_price_catalog=path,origin_price_policy='origin-asof')
    if tamper:
        with pytest.raises(ValueError,match='catalog changed during'):run()
    else:
        result=run();assert result['total_points']==56
        assert result['origin_price_catalog_sha256']==package.file_hash(path)
        assert result['origin_price_policy']=='origin-asof'


@pytest.mark.parametrize('column', ['player_id','GW','origin_gw','fixture'])
@pytest.mark.parametrize('value', [True,False,0,-1,1.5,np.inf])
def test_identity_numeric_positive_whole_and_not_bool(column,value):
    frame=_forecasts(True)
    frame[column]=frame[column].astype(object if isinstance(value,bool) else float)
    frame.loc[frame.index[0],column]=value
    with pytest.raises(ValueError):package.validate_forecasts(frame,1,3,True)


@pytest.mark.parametrize('value', [0.,-1.])
def test_positive_prices_required(value):
    frame=_forecasts();frame.loc[0,'value']=value
    with pytest.raises(ValueError,match='positive'):package.validate_forecasts(frame,1,3,False)


def test_empty_duplicate_columns_and_nullable_nonfinite_rejected():
    frame=_forecasts()
    for bad in (frame.iloc[:0],pd.concat([frame,frame[['minutes']]],axis=1)):
        with pytest.raises(ValueError,match='nonempty with unique'):package.validate_forecasts(bad,1,3,False)
    frame['minutes']=frame.minutes.astype('Float64');frame.loc[0,'minutes']=pd.NA
    with pytest.raises(ValueError,match='nonfinite'):package.validate_forecasts(frame,1,3,False)


def test_same_fixture_position_change_cannot_duplicate_or_sum_twice():
    frame=_forecasts()
    duplicate=frame.iloc[[0]].copy();duplicate['position']='FWD'
    with pytest.raises(ValueError,match='conflicting player-round'):
        package.validate_forecasts(pd.concat([frame,duplicate]),1,3,False)
    frame.loc[0,'position']='MANAGER'
    with pytest.raises(ValueError,match='invalid forecast position'):package.validate_forecasts(frame,1,3,False)


@pytest.mark.parametrize('column', ['actual_total_points','minutes','value','name','team','opponent_team','was_home'])
def test_repeated_fixture_metadata_across_origins_must_match(column):
    frame=_forecasts(True)
    if column not in frame:frame[column]='original'
    index=frame.index[(frame.GW==2)&(frame.origin_gw==2)][0]
    frame.loc[index,column]=99 if pd.api.types.is_numeric_dtype(frame[column]) else 'different'
    with pytest.raises(ValueError,match='across origins'):package.validate_forecasts(frame,1,3,True)


def test_position_consistency_spans_dgw_and_origins():
    frame=_forecasts(True)
    frame.loc[(frame.GW==2)&(frame.origin_gw==2),'position']='FWD'
    with pytest.raises(ValueError,match='conflicting player-round'):package.validate_forecasts(frame,1,3,True)


@pytest.mark.parametrize('seed',[True,False])
def test_bool_seed_rejected_without_artifacts(tmp_path,seed):
    with pytest.raises(ValueError):package.resolve_parameters('defaults',seed)
    with pytest.raises(ValueError):package.run_package(tmp_path/'missing.csv',tmp_path/'new',seed=seed)
    with pytest.raises(ValueError):package.evaluate_variant(pd.DataFrame(),pd.DataFrame(),[],'v',{},1,1,False,tmp_path,seed)
    assert not (tmp_path/'new').exists()


def test_selection_pool_mismatch_aborts_before_finalist_lock(provenance_package):
    state=provenance_package
    def change_pool(directory,variant,origin,result):
        if directory.name=='selection' and variant=='rmse_full':
            path=directory/result['predictions'];frame=pd.read_csv(path);frame['player_id']=2
            frame.to_csv(path,index=False);result['prediction_sha256']=package.file_hash(path)
    state['after_evaluate']=change_pool
    with pytest.raises(ValueError,match='universe'):package.run_package(state['dataset'],state['output'])
    assert not (state['output']/'selection_lock.json').exists()
    assert not (state['output']/'evaluation').exists()


@pytest.mark.parametrize('suffix',['admission.json','log'])
def test_admission_and_optimal_log_tampering_detected(provenance_package,suffix):
    state=provenance_package
    def mutate(directory,variant,origin,result):
        if directory.name=='selection' and variant=='rmse_full':
            path=directory/f'mae_full_True.{suffix}';path.write_text(path.read_text()+'tampered')
    state['after_evaluate']=mutate
    with pytest.raises(ValueError,match='immutable research output changed'):package.run_package(state['dataset'],state['output'])
    assert not (state['output']/'selection_lock.json').exists()


@pytest.mark.parametrize('solution',['Integer Solution Found',None])
def test_structured_exact_optimality_required(tmp_path,monkeypatch,solution):
    monkeypatch.setattr(package,'prediction_diagnostics',lambda *_:[])
    def solve(args):
        frame=pd.DataFrame({'gameweek':[1,2,3],'scoring_protocol':'autosubs_v2','actual_total_points':10.,'alpha':0})
        if solution is not None:
            frame['solver_status']='Optimal';frame['solver_solution_status']=solution
        frame.to_csv(args.output,index=False)
        for _ in range(3):print('Status: Optimal (reported text)')
    monkeypatch.setattr(package.optimize,'run',solve)
    with pytest.raises(ValueError,match='structured proven-optimal'):
        package.evaluate_variant(pd.DataFrame(),pd.DataFrame(),['form'],'cached',{'catboost_mae':{}},1,3,True,tmp_path,0,
                                  forecast_provider=lambda:_forecasts(True))


@pytest.mark.parametrize('suffix',['predictions.csv','squads.csv','admission.json','result.json','log'])
def test_existing_prefix_refused_before_forecasting(tmp_path,monkeypatch,suffix):
    path=tmp_path/f'cached_origin.{suffix}';path.write_bytes(b'preserved')
    def forbidden():pytest.fail('forecasting began before overwrite refusal')
    with pytest.raises(ValueError,match='already exist'):
        package.evaluate_variant(pd.DataFrame(),pd.DataFrame(),['form'],'cached',{'catboost_mae':{}},1,3,True,tmp_path,0,forecast_provider=forbidden)
    assert path.read_bytes()==b'preserved'


def test_manifest_bound_before_selection_and_mutation_between_loads_fails(provenance_package,monkeypatch):
    state=provenance_package
    original=package.load_research_features
    loaded=[]
    def load(max_gw,path):
        loaded.append(max_gw)
        if max_gw==5:
            assert (state['output']/'selection_lock.json').exists()
            registration=json.loads((state['output']/'registration.json').read_text())
            assert registration['manifest_path']==str(state['manifest'].resolve())
            assert registration['manifest_sha256']==package.file_hash(state['manifest'])
            state['manifest'].write_text('{"max_gw_global":999}')
        return original(max_gw,path)
    monkeypatch.setattr(package,'load_research_features',load)
    with pytest.raises(ValueError,match='frozen manifest changed after registration'):
        package.run_package(state['dataset'],state['output'])
    assert loaded==[3,5]
    saved=json.loads((state['output']/'summary.json').read_text())
    assert saved['status']=='incomplete' and saved['evaluation']=={}
    assert not (state['output']/'evaluation').exists()


def test_completion_logger_cannot_change_final_summary(provenance_package,monkeypatch):
    state=provenance_package
    def log(name,*args):
        state['logs'].append((name,*args))
        if name=='research_package_completed':
            path=state['output']/'summary.json'
            path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(package.experiment,'log_result',log)
    with pytest.raises(ValueError,match='completed summary changed after registration'):
        package.run_package(state['dataset'],state['output'])
    saved=json.loads((state['output']/'summary.json').read_text())
    assert saved['status']=='incomplete'
    assert state['logs'][-1][0]=='research_package_incomplete'


def test_reference_schema_matches_registered_selection_schema(provenance_package,monkeypatch):
    state=provenance_package;original=package.load_research_features
    def load(max_gw,path):
        raw,frame,columns=original(max_gw,path)
        if max_gw==5:
            assert (state['output']/'selection_lock.json').exists()
            columns=['changed']
        return raw,frame,columns
    monkeypatch.setattr(package,'load_research_features',load)
    with pytest.raises(ValueError,match='reference ordered feature columns differ'):
        package.run_package(state['dataset'],state['output'])
    assert all(phase=='selection' for phase,_,_ in state['evaluations'])


def test_reference_dataset_provenance_must_match_registered_selection(provenance_package,monkeypatch):
    state=provenance_package;original=package.load_research_features
    def load(max_gw,path):
        raw,frame,columns=original(max_gw,path)
        if max_gw==5:raw.attrs['frozen_dataset_sha256']='0'*64
        return raw,frame,columns
    monkeypatch.setattr(package,'load_research_features',load)
    with pytest.raises(ValueError,match='reference frozen dataset provenance differs'):
        package.run_package(state['dataset'],state['output'])
    assert all(phase=='selection' for phase,_,_ in state['evaluations'])
