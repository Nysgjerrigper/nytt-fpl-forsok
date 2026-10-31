"""Mocked verified-control reuse and conditional horizon-price registration."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl.model import horizon_price_research as runner
from test_price_research import study as source_study


@pytest.fixture
def study(source_study, monkeypatch):
    p = source_study
    # Build a completed price-study fixture using its fully mocked evaluator; no
    # real source forecasts, model fitting or MILP execution occurs here.
    result = runner.price_research.run(p.dataset, p.out, source_dir=p.source_dir)
    reg = json.loads((p.out / "registration.json").read_text())
    verification = {"status": "verified", "registration_sha256": runner.research_package.file_hash(p.out / "registration.json"),
                    "summary_sha256": runner.research_package.file_hash(p.out / "summary.json")}
    (p.out / "verification.json").write_text(json.dumps(verification))
    monkeypatch.setattr(runner, "protocol_cutoffs", lambda: reg["source_registration"]["cutoffs"])
    monkeypatch.setattr(runner, "load_frozen_research_dataset", runner.price_research.load_frozen_research_dataset)
    out = p.out.parent / "horizon_run"
    records, evaluations, comparisons = [], [], []
    monkeypatch.setattr(runner.experiment, "log_result", lambda *args: records.append(args))

    def evaluate(raw, frame, columns, arm, params, start, end, origin, directory, seed, **kwargs):
        assert records[0][0] == "horizon_price_research_registration"
        assert arm in runner.ARMS and origin is True and seed == 0
        variant, model, prefix = runner.ARMS[arm]
        assert kwargs["model_name"] == model and kwargs["retrain_every"] == 4
        assert kwargs["origin_price_policy"] == "origin-current-proxy"
        assert kwargs["origin_price_catalog"] == p.out / f"{directory.name}_origin_prices.csv"
        assert params == reg["parameters"] and columns == reg["feature_columns"]
        assert raw.GW_global.max() <= end
        forecasts = kwargs["forecast_provider"]()
        baseline = pd.read_csv(p.out / directory.name / result["phases"][directory.name][f"{prefix}_replay"]["predictions"])
        pd.testing.assert_frame_equal(forecasts.reset_index(drop=True), baseline)
        predictions, squads = f"{arm}_origin.predictions.csv", f"{arm}_origin.squads.csv"
        forecasts.to_csv(directory / predictions, index=False)
        pd.DataFrame({"gameweek": range(start, end + 1), "actual_total_points": 3.}).to_csv(directory / squads, index=False)
        output = {"total_points": 100., "predictions": predictions, "squads": squads,
                  "prediction_sha256": runner.research_package.file_hash(directory / predictions),
                  "squad_sha256": runner.research_package.file_hash(directory / squads)}
        (directory / f"{arm}_origin.result.json").write_text(json.dumps(output))
        (directory / f"{arm}_origin.admission.json").write_text(json.dumps({"origin_price_policy": "origin-current-proxy"}))
        (directory / f"{arm}_origin.log").write_text("mock scoring log\n")
        evaluations.append((directory.name, arm, start, end))
        return output
    monkeypatch.setattr(runner.research_package, "evaluate_variant", evaluate)

    def compare(candidate, control, **kwargs):
        assert Path(candidate).is_file() and Path(control).is_file()
        assert kwargs == {"n_boot": 10000, "block_len": 3, "seed": 0}
        comparisons.append((candidate, control))
        return {"total_diff": 0.}
    monkeypatch.setattr(runner.research_package.compare_backtests, "compare", compare)
    return SimpleNamespace(dataset=p.dataset, source_dir=p.out, out=out, source=p.source,
                           raw=p.raw, old_result=result, registration=reg, records=records,
                           evaluations=evaluations, comparisons=comparisons,
                           original_source=p.source_dir, original_cached_paths=p.cached_paths)


def _run(study):
    return runner.run(study.dataset, study.out, source_dir=study.source_dir)


def test_only_four_new_score_calls_reuse_old_controls(study):
    result = _run(study)
    assert result["status"] == "complete" and result["promotion_eligible"] is False
    assert len(study.evaluations) == 4 and len(study.comparisons) == 10
    assert study.evaluations == [
        ("selection", "mae_horizon", 2, 3), ("selection", "rmse_horizon", 2, 3),
        ("reference", "mae_horizon", 4, 6), ("reference", "rmse_horizon", 4, 6)]
    for phase in ("selection", "reference"):
        assert set(result["phases"][phase]) == {"mae_horizon", "rmse_horizon"}
        assert [label for label in result["comparisons"][phase]] == [row[0] for row in runner.COMPARISONS]
        for arm, (_, _, prefix) in runner.ARMS.items():
            expected = study.old_result["phases"][phase][f"{prefix}_replay"]["prediction_sha256"]
            assert result["phases"][phase][arm]["prediction_sha256"] == expected
    assert result["reused_controls"] == study.old_result["phases"]
    assert result["catalog_diagnostics"] == study.old_result["catalog_diagnostics"]
    assert result["registration_sha256"] == runner.research_package.file_hash(study.out / "registration.json")
    for path, digest in result["immutable_output_sha256"].items():
        assert runner.research_package.file_hash(Path(path)) == digest
    assert study.records[-1][1]["summary_sha256"] == runner.research_package.file_hash(study.out / "summary.json")


def test_exact_registration_and_old_current_archives_can_differ(study):
    # Changing current Python bytes does not invalidate the independently pinned
    # older evaluated archive. The new bytes are archived before scoring.
    study.source.write_text("new horizon policy source\n")
    _run(study)
    reg = json.loads((study.out / "registration.json").read_text())
    assert reg["new_score_calls"] == 4 and reg["expected_optimal_solves"] == 10
    assert reg["initial_budget"] == 1000 and reg["origin_price_policy"] == "origin-current-proxy"
    assert reg["parameters"] == study.registration["parameters"]
    assert reg["origin_price_catalog_sha256"] == study.registration["origin_price_catalog_sha256"]
    assert (study.out / "source_archive/fpl/model/research_package.py").read_bytes() == study.source.read_bytes()
    assert any("legacy forward/back filling" in text for text in reg["limitations"])


@pytest.mark.parametrize("tamper", ["missing_verification", "summary", "old_archive", "old_catalog", "old_result", "original_cache"])
def test_missing_or_tampered_old_artifacts_fail_before_new_scores(study, tamper):
    if tamper == "missing_verification":
        (study.source_dir / "verification.json").unlink()
    else:
        path = {"summary": study.source_dir / "summary.json",
                "old_archive": study.source_dir / "source_archive/fpl/model/research_package.py",
                "old_catalog": study.source_dir / "reference_origin_prices.csv",
                "old_result": study.source_dir / "selection/mae_replay_origin.result.json",
                "original_cache": study.original_cached_paths[0]}[tamper]
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        _run(study)
    assert study.evaluations == []


def test_no_missing_archive_fallback_to_current_sources(study):
    (study.source_dir / "source_archive/fpl/model/research_package.py").unlink()
    with pytest.raises(FileNotFoundError):
        _run(study)
    assert study.evaluations == []


@pytest.mark.parametrize("artifact", ["catalog", "old_control", "original_cache", "new_log", "current_source", "dataset"])
def test_phase_mutation_preserves_incomplete_checkpoint(study, monkeypatch, artifact):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        path = {"catalog": study.source_dir / "reference_origin_prices.csv",
                "old_control": study.source_dir / "selection/mae_replay_origin.squads.csv",
                "original_cache": study.original_cached_paths[-1],
                "new_log": args[8] / f"{args[3]}_origin.log",
                "current_source": study.source, "dataset": study.dataset}[artifact]
        # A new log has not been sealed yet in the current evaluator return; mutate
        # it during the next callback to test an already-bound output instead.
        if artifact != "new_log" or len(study.evaluations) == 2:
            if artifact == "new_log":
                path = args[8] / "mae_horizon_origin.log"
            path.write_text(path.read_text() + "\n")
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError):
        _run(study)
    summary = json.loads((study.out / "summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["elapsed_seconds"] >= 0
    assert len(study.evaluations) == (2 if artifact == "new_log" else 1)
    assert not study.comparisons


def test_new_policy_must_preserve_old_model_prediction_bytes(study, monkeypatch):
    evaluate = runner.research_package.evaluate_variant
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        path = args[8] / result["predictions"]
        frame = pd.read_csv(path)
        frame.loc[0, "predicted_total_points"] += .1
        frame.to_csv(path, index=False)
        result["prediction_sha256"] = runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package, "evaluate_variant", changed)
    with pytest.raises(ValueError, match="prediction bytes"):
        _run(study)
    assert len(study.evaluations) == 1 and not study.comparisons


def test_existing_outputs_never_overwritten(study):
    _run(study)
    saved = {path: path.read_bytes() for path in study.out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already exists"):
        _run(study)
    assert {path: path.read_bytes() for path in saved} == saved
    assert len(study.evaluations) == 4


def test_cli_portable_price_source(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda dataset, out, **kwargs: calls.append((dataset, out, kwargs)))
    runner.main(["--dataset", "frozen.csv", "--output-dir", "new", "--source-dir", "old-price"])
    assert calls == [(Path("frozen.csv"), Path("new"), {"source_dir": Path("old-price")})]
