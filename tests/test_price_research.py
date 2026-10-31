"""Mocked cache-only price proxy registration and immutable scoring inputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl import config
from fpl.model import price_research as runner


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _forecasts(start, end, prediction):
    rows = []
    for origin in range(start, end + 1):
        for gw in range(origin, min(origin + 3, end + 1)):
            for player in (1, 2):
                rows.append({"player_id": player, "position": "MID", "origin_gw": origin,
                             "GW": gw, "fixture": gw * 10 + player, "value": 50., "minutes": 90.,
                             "actual_total_points": 2., "predicted_total_points": float(prediction)})
    return pd.DataFrame(rows)


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    source = tmp_path / "fpl" / "model" / "research_package.py"
    source.parent.mkdir(parents=True)
    source.write_text("new registered code\n")
    dataset = tmp_path / "frozen.csv"
    dataset.write_text("frozen complete bytes\n")
    digest = runner.research_package.file_hash(dataset)
    source_dir, output_dir = tmp_path / "old_run", tmp_path / "new_run"
    cutoffs = {"discovery_max_gw": 1, "selection_min_gw": 2, "selection_max_gw": 3,
               "backtest_min_gw": 4, "backtest_max_gw": 6}
    monkeypatch.setattr(runner, "protocol_cutoffs", lambda: cutoffs)
    old_source = source_dir / "evaluated_sources" / "fpl" / "model" / "research_package.py"
    old_source.parent.mkdir(parents=True)
    old_source.write_text("different preserved old code\n")
    code_sha = {"fpl/model/research_package.py": runner.research_package.file_hash(old_source)}
    params = {name: {position: {name: {"depth": 3, "random_seed": 0}}
                    for position in config.ONFIELD_POSITIONS}
              for name in ("catboost_mae", "catboost_rmse")}
    registration = {"seed": 0, "cutoffs": cutoffs, "retrain_every": 4, "horizon": 3,
                    "scoring_protocol": "autosubs_v2", "points_accounting": "net_after_transfer_hits",
                    "dataset_sha256": digest, "parameters": params,
                    "feature_variants": {"mae_full": ["base"], "rmse_full": ["base"]},
                    "gap_rel": 0, "chips": "disabled", "code_sha256": code_sha}
    _write(source_dir / "registration.json", registration)
    _write(source_dir / "source_bundle.json", {"status": "verified", "source_sha256": code_sha})
    selection, evaluation, cached_paths = {}, {}, []
    for phase, start, end in (("selection", 2, 3), ("evaluation", 4, 6)):
        for variant, prediction in (("mae_full", 1), ("rmse_full", 3)):
            path = source_dir / phase / f"{variant}.csv"
            path.parent.mkdir(exist_ok=True)
            _forecasts(start, end, prediction).to_csv(path, index=False)
            cached_paths.append(path)
            result = {"predictions": path.name, "prediction_sha256": runner.research_package.file_hash(path)}
            if phase == "selection":
                selection[variant] = result
            else:
                evaluation[variant] = {"origin": result}
    _write(source_dir / "summary.json", {"status": "complete", "selection": selection, "evaluation": evaluation})
    _write(source_dir / "selection_summary.json", {"status": "running", "selection": selection})
    _write(source_dir / "selection_lock.json", {
        "selection_summary_sha256": runner.research_package.file_hash(source_dir / "selection_summary.json")})
    _write(source_dir / "verification.json", {
        "status": "PASS", "summary_file_sha256": runner.research_package.file_hash(source_dir / "summary.json"),
        "selection_summary_sha256": runner.research_package.file_hash(source_dir / "selection_summary.json"),
        "evaluated_runner_sha256": runner.research_package.file_hash(old_source)})
    raw = pd.DataFrame({"player_id": [1] * 7 + [2] * 4,
                        "GW_global": list(range(1, 8)) + list(range(3, 7)),
                        "value": [50.] * 7 + [70.] * 4, "total_points": 2., "position": "MID"})
    raw.attrs["frozen_dataset_sha256"] = digest
    loaded = []
    def load(path):
        loaded.append(path)
        return raw.copy()
    monkeypatch.setattr(runner, "load_frozen_research_dataset", load)
    records, evaluated, compares, catalog_inputs = [], [], [], []
    monkeypatch.setattr(runner.experiment, "_library_versions", lambda: {"fake": "1"})
    monkeypatch.setattr(runner.experiment, "log_result", lambda *args: records.append(args))
    build_catalog = runner.build_origin_price_catalog
    def catalog(frame, player_ids, start, end):
        assert frame.GW_global.max() <= end
        catalog_inputs.append((start, end, int(frame.GW_global.max()), player_ids))
        return build_catalog(frame, player_ids, start, end)
    monkeypatch.setattr(runner, "build_origin_price_catalog", catalog)

    def evaluate(raw_frame, feature_frame, columns, arm, parameters, start, end, origin, directory, seed, **kwargs):
        assert records[0][0] == "price_research_registration"
        assert (output_dir / "selection_origin_prices.csv").is_file()
        assert (output_dir / "reference_origin_prices.csv").is_file()
        assert (output_dir / "catalog_lock.json").is_file()
        assert len(catalog_inputs) == 2  # Both phases prepared before any solve.
        assert columns == ["base"] and seed == 0 and origin is True
        assert kwargs["retrain_every"] == 4
        variant, model, policy = runner.ARMS[arm]
        assert kwargs["model_name"] == model
        assert kwargs["origin_price_policy"] == policy
        assert kwargs["origin_price_catalog"] == output_dir / f"{directory.name}_origin_prices.csv"
        assert parameters == params
        assert raw_frame.GW_global.max() <= end
        forecasts = kwargs["forecast_provider"]()
        assert set(forecasts.predicted_total_points) == ({1.} if variant == "mae_full" else {3.})
        predictions, squads = f"{arm}_origin.predictions.csv", f"{arm}_origin.squads.csv"
        forecasts.to_csv(directory / predictions, index=False)
        pd.DataFrame({"gameweek": range(start, end + 1), "actual_total_points": 2.}).to_csv(directory / squads, index=False)
        evaluated.append((directory.name, arm, start, end))
        result = {"total_points": 100., "predictions": predictions, "squads": squads,
                  "prediction_sha256": runner.research_package.file_hash(directory / predictions),
                  "squad_sha256": runner.research_package.file_hash(directory / squads)}
        _write(directory / f"{arm}_origin.result.json", result)
        _write(directory / f"{arm}_origin.admission.json", {"seed": seed, "price_policy": policy})
        (directory / f"{arm}_origin.log").write_text("mocked evaluation log\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", evaluate)
    def compare(candidate, control, **kwargs):
        assert Path(candidate).is_file() and Path(control).is_file()
        assert kwargs == {"n_boot": 10000, "block_len": 3, "seed": 0}
        compares.append((candidate, control))
        return {"total_diff": 0.}
    monkeypatch.setattr(runner.research_package.compare_backtests, "compare", compare)
    return SimpleNamespace(dataset=dataset, source_dir=source_dir, out=output_dir, source=source,
                           old_source=old_source, raw=raw, loaded=loaded, records=records,
                           evaluated=evaluated, compares=compares, catalog_inputs=catalog_inputs,
                           cached_paths=cached_paths)


def _run(study):
    return runner.run(study.dataset, study.out, source_dir=study.source_dir)


def test_both_catalogs_registered_before_eight_cached_arms(study):
    result = _run(study)
    assert result["status"] == "complete" and result["promotion_eligible"] is False
    assert study.loaded == [study.dataset]
    assert study.catalog_inputs == [(2, 3, 3, [1, 2]), (4, 6, 6, [1, 2])]
    assert len(study.evaluated) == 8 and len(study.compares) == 8
    for phase in ("selection", "reference"):
        assert [arm for directory, arm, _, _ in study.evaluated if directory == phase] == list(runner.ARMS)
        for model in ("mae", "rmse"):
            left = result["phases"][phase][f"{model}_replay"]
            right = result["phases"][phase][f"{model}_asof"]
            assert left["prediction_sha256"] == right["prediction_sha256"]
    assert study.records[-1][1]["summary_sha256"] == runner.research_package.file_hash(study.out / "summary.json")
    assert result["registration_sha256"] == runner.research_package.file_hash(study.out / "registration.json")
    for path, digest in result["immutable_output_sha256"].items():
        assert runner.research_package.file_hash(Path(path)) == digest
    for phase in ("selection", "reference"):
        assert str((study.out / f"{phase}_summary.json").resolve()) in result["immutable_output_sha256"]


def test_unknown_eligibility_and_price_age_are_descriptive(study):
    result = _run(study)
    earlier = result["catalog_diagnostics"]["selection"]
    assert [row["eligible"] for row in earlier] == [1, 1]
    assert [row["unknown"] for row in earlier] == [1, 1]
    assert [row["age_mean"] for row in earlier] == [1., 1.]
    later = result["catalog_diagnostics"]["reference"]
    assert [row["eligible"] for row in later] == [2, 2, 2]
    assert [row["unknown"] for row in later] == [0, 0, 0]
    registration = json.loads((study.out / "registration.json").read_text())
    assert registration["eligibility_basis"] == "phase_optimizer_union_with_known_lagged_prices"
    assert any("not actual verified historical deadline" in text for text in registration["limitations"])


def test_old_source_provenance_allows_current_code_changes(study):
    _run(study)
    assert study.source.read_bytes() != study.old_source.read_bytes()
    assert (study.out / "source_archive/fpl/model/research_package.py").read_bytes() == study.source.read_bytes()
    registration = json.loads((study.out / "registration.json").read_text())
    assert registration["source_registration"]["code_sha256"]["fpl/model/research_package.py"] == runner.research_package.file_hash(study.old_source)
    lock = json.loads((study.out / "catalog_lock.json").read_text())
    assert lock["registration_sha256"] == runner.research_package.file_hash(study.out / "registration.json")


@pytest.mark.parametrize("artifact", ["old_source", "cached", "bundle", "verification"])
def test_old_provenance_tampering_fails_before_scoring(study, artifact):
    path = {"old_source": study.old_source, "cached": study.cached_paths[0],
            "bundle": study.source_dir / "source_bundle.json", "verification": study.source_dir / "verification.json"}[artifact]
    if artifact in ("bundle", "verification"):
        data = json.loads(path.read_text())
        data["status"] = "tampered"
        _write(path, data)
    else:
        path.write_text("tampered\n")
    with pytest.raises(ValueError):
        _run(study)
    assert study.evaluated == [] and study.loaded == []


def test_dataset_mismatch_before_catalogs(study):
    study.raw.attrs["frozen_dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dataset differs"):
        _run(study)
    assert study.evaluated == [] and study.catalog_inputs == []


@pytest.mark.parametrize("artifact", ["catalog", "catalog_lock", "dataset", "current_source", "cached", "scored_output"])
def test_mutation_preserves_incomplete_and_prevents_next_score(study, monkeypatch, artifact):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        path = {"catalog": study.out / "reference_origin_prices.csv", "catalog_lock": study.out / "catalog_lock.json",
                "dataset": study.dataset, "current_source": study.source, "cached": study.cached_paths[-1],
                "scored_output": args[8] / result["squads"]}[artifact]
        path.write_text(path.read_text() + "\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError):
        _run(study)
    summary = json.loads((study.out / "summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["elapsed_seconds"] >= 0
    assert len(study.evaluated) == 1 and study.compares == []


def test_price_policies_cannot_change_predictions(study, monkeypatch):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[3] == "mae_asof":
            path = args[8] / result["predictions"]
            frame = pd.read_csv(path)
            frame.loc[0, "predicted_total_points"] += .1
            frame.to_csv(path, index=False)
            result["prediction_sha256"] = runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="identical cached forecast bytes"):
        _run(study)
    assert len(study.evaluated) == 2 and study.compares == []


def test_existing_study_never_overwritten(study):
    _run(study)
    saved = {path: path.read_bytes() for path in study.out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already exists"):
        _run(study)
    assert {path: path.read_bytes() for path in saved} == saved
    assert len(study.evaluated) == 8


def test_cli_portable_source_and_dataset(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda dataset, output, **kwargs: calls.append((dataset, output, kwargs)))
    runner.main(["--dataset", "frozen.csv", "--output-dir", "new", "--source-dir", "old"])
    assert calls == [(Path("frozen.csv"), Path("new"), {"source_dir": Path("old")})]


def test_source_verification_is_mandatory_before_any_solve(study):
    (study.source_dir / "verification.json").unlink()
    with pytest.raises(ValueError, match="verification.json is required"):
        _run(study)
    assert study.evaluated == [] and study.loaded == []


@pytest.mark.parametrize("setting,value", [("gap_rel", .01), ("chips", "enabled"), ("features", ["different"])])
def test_original_protocol_requires_exact_solver_chips_and_features(study, setting, value):
    path = study.source_dir / "registration.json"
    original = json.loads(path.read_text())
    if setting == "features":
        original["feature_variants"]["rmse_full"] = value
    else:
        original[setting] = value
    _write(path, original)
    with pytest.raises(ValueError):
        _run(study)
    assert not study.evaluated


@pytest.mark.parametrize("suffix", ["admission.json", "result.json", "log"])
def test_completed_arm_metadata_and_log_bytes_remain_sealed(study, monkeypatch, suffix):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[3] == "mae_asof":
            path = args[8] / f"mae_replay_origin.{suffix}"
            path.write_text(path.read_text() + "\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="sealed price research artifact changed"):
        _run(study)
    assert len(study.evaluated) == 2 and not study.compares
    summary = json.loads((study.out / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert str((study.out / "selection" / f"mae_replay_origin.{suffix}").resolve()) in summary["immutable_output_sha256"]

@pytest.fixture(params=[1, 2])
def modern(study, request, monkeypatch):
    seed = request.param
    registration_path = study.source_dir / 'registration.json'
    reg = json.loads(registration_path.read_text())
    reg['seed'] = seed
    reg['source_archive'] = 'evaluated_sources'
    for model, positions in reg['parameters'].items():
        for position in positions:
            positions[position][model]['random_seed'] = seed
    _write(registration_path, reg)
    (study.source_dir / 'source_bundle.json').unlink()
    summary_path = study.source_dir / 'summary.json'
    summary = json.loads(summary_path.read_text())
    immutable = {}
    for path in study.cached_paths:
        prefix = path.with_suffix('')
        for suffix in ('.squads.csv', '.admission.json', '.result.json', '.log'):
            artifact = prefix.with_suffix(suffix)
            artifact.write_text('preserved source artifact\n')
            immutable[str(artifact.relative_to(study.source_dir))] = runner.research_package.file_hash(artifact)
        immutable[str(path.relative_to(study.source_dir))] = runner.research_package.file_hash(path)
    for name in ('selection_summary.json', 'selection_lock.json'):
        immutable[name] = runner.research_package.file_hash(study.source_dir / name)
    summary['immutable_output_sha256'] = immutable
    summary['registration_sha256'] = runner.research_package.file_hash(registration_path)
    _write(summary_path, summary)
    _write(study.source_dir / 'verification.json', {
        'status': 'verified', 'summary_sha256': runner.research_package.file_hash(summary_path),
        'registration_sha256': runner.research_package.file_hash(registration_path)})
    study.seed = seed
    study.reg = reg
    def evaluate(raw, frame, columns, arm, params, start, end, origin, directory, actual_seed, **kwargs):
        assert actual_seed == seed and params == reg['parameters']
        assert (start, end) == (4, 6) and directory.name == 'reference'
        assert study.catalog_inputs == [(4, 6, 6, [1, 2])]
        assert not (study.out / 'selection_origin_prices.csv').exists()
        assert study.records[0][0] == 'price_research_registration'
        registered = json.loads((study.out / 'registration.json').read_text())
        assert registered['seed'] == seed and registered['reference_only']
        assert registered['include_current_proxy']
        assert len(registered['arms']) == (4 if registered['skip_lagged_proxy'] else 6)
        assert kwargs['model_name'] == ('catboost_mae' if arm.startswith('mae') else 'catboost_rmse')
        assert kwargs['origin_price_policy'] == ('origin-current-proxy' if arm.endswith('current') else runner.ARMS[arm][2])
        assert kwargs['retrain_every'] == 4 and origin
        forecast = kwargs['forecast_provider']()
        assert raw is frame and raw.GW_global.max() == 6 and columns == ['base']
        predictions, squads = f'{arm}_origin.predictions.csv', f'{arm}_origin.squads.csv'
        forecast.to_csv(directory / predictions, index=False)
        pd.DataFrame({'gameweek': [4,5,6], 'actual_total_points': [2.,2.,2.]}).to_csv(directory / squads, index=False)
        result = {'predictions': predictions, 'squads': squads, 'total_points': 6.,
                  'prediction_sha256': runner.research_package.file_hash(directory / predictions),
                  'squad_sha256': runner.research_package.file_hash(directory / squads)}
        for suffix in ('admission.json','result.json'):
            _write(directory / f'{arm}_origin.{suffix}', result)
        (directory / f'{arm}_origin.log').write_text('cached only\n')
        study.evaluated.append((directory.name, arm, start, end))
        return result
    monkeypatch.setattr(runner.research_package, 'evaluate_variant', evaluate)
    def compare(candidate, control, **kwargs):
        assert kwargs == {'n_boot': 10000, 'block_len': 3, 'seed': seed}
        study.compares.append((candidate, control))
        return {'total_diff': 0.}
    monkeypatch.setattr(runner.research_package.compare_backtests, 'compare', compare)
    return study


def _replicate(study):
    return runner.run(study.dataset, study.out, source_dir=study.source_dir,
                      seed=study.seed, reference_only=True, include_current_proxy=True)


def test_modern_six_reference_arms_nine_comparisons_no_fits(modern):
    result = _replicate(modern)
    assert result['status'] == 'complete'
    assert list(result['phases']) == ['reference']
    assert [arm for _,arm,_,_ in modern.evaluated] == [*runner.ARMS,'mae_current','rmse_current']
    assert len(modern.compares) == 9 and len(runner.ARMS) == 4
    assert modern.loaded == [modern.dataset]
    registration = json.loads((modern.out / 'registration.json').read_text())
    assert registration['comparison']['seed'] == modern.seed
    for model in ('mae','rmse'):
        hashes = {result['phases']['reference'][f'{model}_{policy}']['prediction_sha256']
                  for policy in ('replay','asof','current')}
        assert len(hashes) == 1
    for path, digest in result['immutable_output_sha256'].items():
        assert runner.research_package.file_hash(Path(path)) == digest


@pytest.mark.parametrize('artifact', ['verification.json','evaluated_sources/fpl/model/research_package.py'])
def test_modern_missing_required_source_no_fallback(modern, artifact):
    (modern.source_dir / artifact).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _replicate(modern)
    assert not modern.evaluated and not modern.loaded


@pytest.mark.parametrize('artifact', ['selection_lock.json','selection_summary.json',
                                      'selection/mae_full.admission.json',
                                      'evaluation/rmse_full.result.json',
                                      'evaluation/mae_full.log'])
def test_modern_bound_metadata_tamper_fails_before_solve(modern, artifact):
    path = modern.source_dir / artifact
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        _replicate(modern)
    assert not modern.evaluated


def test_modern_runtime_source_metadata_rechecked(modern, monkeypatch):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        path = modern.source_dir / 'evaluation/mae_full.log'
        path.write_text(path.read_text() + '\n')
        return result
    monkeypatch.setattr(runner.research_package, 'evaluate_variant', changed)
    with pytest.raises(ValueError, match='sealed price research artifact changed'):
        _replicate(modern)
    assert len(modern.evaluated) == 1 and not modern.compares
    assert json.loads((modern.out / 'summary.json').read_text())['status'] == 'incomplete'


def test_modern_parameter_seed_must_match(modern):
    reg = modern.reg
    reg['parameters']['catboost_mae']['MID']['catboost_mae']['random_seed'] = 0
    _write(modern.source_dir / 'registration.json', reg)
    reg_hash = runner.research_package.file_hash(modern.source_dir / 'registration.json')
    summary_path = modern.source_dir / 'summary.json'
    summary = json.loads(summary_path.read_text()); summary['registration_sha256'] = reg_hash
    _write(summary_path, summary)
    _write(modern.source_dir / 'verification.json', {'status': 'verified', 'registration_sha256': reg_hash,
                                                   'summary_sha256': runner.research_package.file_hash(summary_path)})
    with pytest.raises(ValueError, match='parameter seed'):
        _replicate(modern)
    assert not modern.evaluated


@pytest.mark.parametrize('seed', [True, False, -1, 3, 1.0, '1'])
def test_invalid_seed_refused_before_source_loading(study, seed):
    with pytest.raises(ValueError, match='seed must'):
        runner.run(study.dataset, study.out, source_dir=study.source_dir, seed=seed)
    assert not study.evaluated and not study.loaded


def test_cli_replication_flags(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, 'run', lambda *args, **kwargs: calls.append((args,kwargs)))
    runner.main(['--output-dir','new','--seed','2','--reference-only','--include-current-proxy'])
    assert calls[0][1] == {'source_dir': None, 'seed': 2, 'reference_only': True, 'include_current_proxy': True}


def test_exclude_lagged_proxy_four_reference_arms_four_cis(modern):
    result = runner.run(modern.dataset, modern.out, source_dir=modern.source_dir,
                        seed=modern.seed, reference_only=True, include_current_proxy=True,
                        skip_lagged_proxy=True)
    assert [arm for _,arm,_,_ in modern.evaluated] == ['mae_replay','rmse_replay','mae_current','rmse_current']
    assert len(modern.compares) == 4
    registration = json.loads((modern.out / 'registration.json').read_text())
    assert registration['skip_lagged_proxy'] and registration['evaluation_jobs'] == 4
    assert registration['expected_optimal_solves'] == 12  # Four arms x three synthetic rounds.
    assert set(registration['comparison_labels']) == {'rmse_vs_mae_replay', 'mae_current_vs_replay',
                                                     'rmse_current_vs_replay', 'rmse_vs_mae_current'}
    assert all('asof' not in label for label in result['comparisons']['reference'])
    assert len(runner.ARMS) == 4  # Legacy global protocol preserved.


def test_cli_skip_lagged_proxy(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, 'run', lambda *args, **kwargs: calls.append(kwargs))
    runner.main(['--output-dir','new','--seed','1','--reference-only','--include-current-proxy','--skip-lagged-proxy'])
    assert calls[0]['skip_lagged_proxy'] is True and calls[0]['seed'] == 1


def test_modern_declared_seed_must_match(modern):
    with pytest.raises(ValueError, match='matching-seed'):
        runner._source_inputs(modern.source_dir, runner.protocol_cutoffs(), seed=3-modern.seed)
    assert not modern.evaluated


def test_modern_status_cannot_use_legacy_verification(modern):
    path = modern.source_dir / 'verification.json'
    verification=json.loads(path.read_text()); verification['status']='PASS'
    _write(path, verification)
    with pytest.raises(ValueError, match='verified status'):
        _replicate(modern)
    assert not modern.evaluated


def test_modern_current_proxy_cannot_change_predictions(modern, monkeypatch):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if args[3] == 'mae_current':
            path=args[8] / result['predictions']
            frame=pd.read_csv(path); frame.loc[0,'predicted_total_points'] += 1.
            frame.to_csv(path,index=False)
            result['prediction_sha256']=runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package,'evaluate_variant',changed)
    with pytest.raises(ValueError, match='identical cached forecast bytes'):
        _replicate(modern)
    assert not modern.compares


def test_default_source_directory_matches_seed(study, monkeypatch):
    calls=[]
    def source(path, cutoffs, seed=0):
        calls.append((path,seed))
        raise RuntimeError('stop before data')
    monkeypatch.setattr(runner,'_source_inputs',source)
    with pytest.raises(RuntimeError,match='stop before data'):
        runner.run(study.dataset,study.out,seed=2)
    assert calls == [(config.ROOT / 'experiments/research_priorities_2026-10-03/selection_seed2',2)]
