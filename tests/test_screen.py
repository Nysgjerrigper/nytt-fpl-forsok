"""Leakage and fair-comparison contracts for exploratory screening."""
import json

import numpy as np
import pandas as pd
import pytest

from fpl.model import screen


def synthetic_admission(path):
    """Stamp synthetic CLI input after replacing real frozen admission in tests."""
    frame = pd.read_csv(path)
    frame.attrs["frozen_dataset_sha256"] = screen._file_hash(path)
    return frame


def panel():
    cutoffs = screen.protocol_cutoffs()
    rows = []
    for gw in range(cutoffs["selection_min_gw"], cutoffs["selection_max_gw"] + 1):
        for pid in range(4):
            for expert, offset in (("lightgbm_l2", 0.2), ("catboost_rmse", -0.3)):
                rows.append(dict(player_id=pid, position="MID", GW_global=gw, expert=expert,
                                 prediction=pid + offset, actual_total_points=pid + gw % 2,
                                 train_max_gw=gw - 1, seed=0, params_hash=expert,
                                 stage="exploratory_screen", mase_scale=2.0))
    return pd.DataFrame(rows)


def test_weights_never_see_current_or_future_outcomes():
    original = panel()
    report, predictions, weights, _ = screen.compare_ensembles(original, positions=("MID",), warmup_weeks=4)
    gw = int(predictions.GW_global.min())
    changed = original.copy()
    changed.loc[changed.GW_global >= gw, "actual_total_points"] += 1000
    _, new_predictions, new_weights, _ = screen.compare_ensembles(changed, positions=("MID",), warmup_weeks=4)
    np.testing.assert_array_equal(weights[weights.GW_global == gw].weight, new_weights[new_weights.GW_global == gw].weight)
    np.testing.assert_array_equal(predictions[predictions.GW_global == gw].prediction,
                                  new_predictions[new_predictions.GW_global == gw].prediction)
    assert (weights.weight_train_max_gw < weights.GW_global).all()
    assert report.rows.nunique() == 1 and report.weeks.nunique() == 1
    assert set(report.strategy) == {"single:catboost_rmse", "single:lightgbm_l2", "equal", "nnls", "ridge", "top_k"}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nonfinite", "future_train", "missing_week"])
def test_bad_evidence_cannot_produce_a_report(fault):
    oof = panel()
    if fault == "missing":
        oof = oof.iloc[1:]
    elif fault == "duplicate":
        oof = pd.concat([oof, oof.iloc[:1]])
    elif fault == "nonfinite":
        oof.loc[0, "prediction"] = np.inf
    elif fault == "future_train":
        oof["train_max_gw"] = oof.GW_global
    else:
        oof = oof[oof.GW_global != oof.GW_global.min()]
    with pytest.raises(ValueError):
        screen.compare_ensembles(oof, positions=("MID",), warmup_weeks=4)


def test_generator_uses_past_labels_and_explicit_builder(monkeypatch):
    oof = panel()
    frame = oof.drop_duplicates(["player_id", "GW_global"])[["player_id", "position", "GW_global", "actual_total_points"]]
    frame = frame.rename(columns={"actual_total_points": "total_points"})
    first = int(frame.GW_global.min())
    history = frame[frame.GW_global <= first + 1].copy()
    history.GW_global -= 2
    frame = pd.concat([history, frame], ignore_index=True)
    frame["x"] = frame.player_id
    fits = []

    class Dummy:
        def fit(self, X, y):
            self.mean = float(y.mean())
            fits.append(X.index.copy())
            return self

        def predict(self, X):
            return np.full(len(X), self.mean)

    calls = []
    def build(name, **kwargs):
        calls.append((name, kwargs))
        return Dummy()
    monkeypatch.setattr(screen.models, "build_registered_model", build)
    monkeypatch.setattr(screen.models, "_tuned_params", lambda *a, **k: pytest.fail("ambient parameters read"))
    result = screen.generate_oof(frame, ["x"], {"lightgbm_l2": {"n_estimators": 2}},
                                 positions=("MID",), seed=7, retrain_every=4)
    assert (result.train_max_gw < result.GW_global).all()
    assert len(fits) == 4
    assert all(kwargs["params"] == {"n_estimators": 2} and kwargs["seed"] == 7 for _, kwargs in calls)
    for idx, train_indices in enumerate(fits):
        assert frame.loc[train_indices, "GW_global"].max() < first + idx * 4
    changed = frame.copy()
    changed.loc[changed.GW_global >= first, "total_points"] += 100
    second = screen.generate_oof(changed, ["x"], {"lightgbm_l2": {}}, positions=("MID",), seed=7, retrain_every=4)
    np.testing.assert_array_equal(result.loc[result.GW_global < first + 4, "prediction"],
                                  second.loc[second.GW_global < first + 4, "prediction"])


def test_cli_writes_hashes_and_refuses_overwrite(tmp_path, monkeypatch):
    dataset = tmp_path / "raw.csv"
    pd.DataFrame({"GW_global": [1, 137, 153, 191]}).to_csv(dataset, index=False)
    seen = {}
    def build(raw):
        seen["raw"] = raw
        return raw
    monkeypatch.setattr(screen, "load_frozen_research_dataset", synthetic_admission)
    monkeypatch.setattr(screen.features, "build_feature_frame", build)
    monkeypatch.setattr(screen.features, "feature_columns", lambda frame: ["x"])
    monkeypatch.setattr(screen, "generate_oof", lambda *a, **k: panel())
    output = tmp_path / "run"
    argv = ["--models", "lightgbm_l2,catboost_rmse", "--positions", "MID", "--dataset", str(dataset), "--output-dir", str(output)]
    screen.main(argv)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["promotion_eligible"] is False
    assert manifest["dataset_sha256"] == screen._file_hash(dataset)
    assert manifest["semantics"]["training"] == "per-fixture feature rows and targets"
    assert manifest["semantics"]["mase_scale"] == "training-only summed player-round lag-one scale"
    assert manifest["outputs"]["member_oof.csv"] == screen._file_hash(output / "member_oof.csv")
    assert seen["raw"].GW_global.max() == screen.protocol_cutoffs()["selection_min_gw"]
    with pytest.raises(SystemExit):
        screen.main(argv)


def test_failed_run_cannot_look_complete(tmp_path, monkeypatch):
    dataset = tmp_path / "raw.csv"
    pd.DataFrame({"GW_global": [1]}).to_csv(dataset, index=False)
    monkeypatch.setattr(screen, "load_frozen_research_dataset", synthetic_admission)
    monkeypatch.setattr(screen.features, "build_feature_frame", lambda raw: raw)
    monkeypatch.setattr(screen.features, "feature_columns", lambda frame: ["x"])
    def fail(*args, **kwargs):
        raise ValueError("bad model output")
    monkeypatch.setattr(screen, "generate_oof", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="bad model output"):
        screen.main(["--models", "lightgbm_l2", "--dataset", str(dataset), "--output-dir", str(output)])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and not manifest["promotion_eligible"]


def fixture_frame():
    """Build an identifiable two-fixture round with nonzero round-level history."""
    cutoffs = screen.protocol_cutoffs()
    first = cutoffs["selection_min_gw"]
    rows = []
    for gw in range(first - 2, cutoffs["selection_max_gw"] + 1):
        for pid in range(4):
            for fixture in (1, 2):
                points = fixture + pid + (2 if gw == first - 1 else gw % 2)
                rows.append(dict(player_id=pid, position="MID", GW_global=gw,
                                 fixture=gw * 10 + fixture, total_points=points,
                                 x=fixture))
    return pd.DataFrame(rows)


def install_fixture_model(monkeypatch, fits):
    """Use an observable fixture-level predictor without fitting real estimators."""
    class FixtureModel:
        def fit(self, X, y):
            fits.append((X.copy(), y.copy()))
            return self

        def predict(self, X):
            return X["x"].to_numpy(dtype=float)

    monkeypatch.setattr(screen.models, "build_registered_model", lambda *a, **k: FixtureModel())


def test_double_gameweeks_fit_fixtures_and_score_summed_rounds(monkeypatch):
    frame = fixture_frame()
    fits = []
    install_fixture_model(monkeypatch, fits)
    parameters = {"lightgbm_l2": {}, "catboost_rmse": {}}
    result = screen.generate_oof(frame, ["x"], parameters,
                                 positions=("MID",), seed=0, retrain_every=4)
    first = screen.protocol_cutoffs()["selection_min_gw"]
    history = frame[frame.GW_global < first]
    assert len(fits[0][0]) == len(history) == 16
    pd.testing.assert_series_equal(fits[0][1], history.total_points)
    assert set(fits[0][0].x) == {1, 2}
    assert not result.duplicated(["player_id", "position", "GW_global", "expert"]).any()
    assert len(result) == 4 * 16 * 2
    assert (result.fixture_count == 2).all()
    assert (result.prediction == 3).all()
    expected = frame.groupby(["player_id", "position", "GW_global"]).total_points.sum()
    for expert in parameters:
        actual = result[result.expert == expert].set_index(["player_id", "position", "GW_global"]).actual_total_points
        pd.testing.assert_series_equal(actual.sort_index(), expected.reindex(actual.index).sort_index(), check_names=False)
    _, predictions, _, _ = screen.compare_ensembles(result, positions=("MID",), warmup_weeks=4)
    assert (predictions.fixture_count == 2).all()
    singles = predictions[predictions.strategy.str.startswith("single:")]
    np.testing.assert_allclose(singles.prediction, 3)


@pytest.mark.parametrize("fault", ["duplicate", "missing_column", "missing_value"])
def test_ambiguous_fixture_rows_fail_before_fit(monkeypatch, fault):
    frame = fixture_frame()
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        message = "duplicate fixture identities"
    elif fault == "missing_column":
        frame = frame.drop(columns="fixture")
        message = "nonmissing fixture identities"
    else:
        frame.loc[0, "fixture"] = np.nan
        message = "nonmissing fixture identities"
    monkeypatch.setattr(screen.models, "build_registered_model", lambda *a, **k: pytest.fail("fitted ambiguous data"))
    with pytest.raises(ValueError, match=message):
        screen.generate_oof(frame, ["x"], {"lightgbm_l2": {}},
                            positions=("MID",), seed=0, retrain_every=4)


def test_scale_uses_training_round_totals_only(monkeypatch):
    frame = fixture_frame()
    install_fixture_model(monkeypatch, [])
    first = screen.protocol_cutoffs()["selection_min_gw"]
    history = frame[frame.GW_global < first]
    round_history = history.groupby(["player_id", "position", "GW_global"], as_index=False).total_points.sum()
    expected = screen.metrics.naive_lag1_scale(round_history)
    assert expected != screen.metrics.naive_lag1_scale(history)
    result = screen.generate_oof(frame, ["x"], {"lightgbm_l2": {}},
                                 positions=("MID",), seed=0, retrain_every=4)
    assert (result.loc[result.GW_global < first + 4, "mase_scale"] == expected).all()
    changed = frame.copy()
    changed.loc[changed.GW_global >= first, "total_points"] += 1000
    second = screen.generate_oof(changed, ["x"], {"lightgbm_l2": {}},
                                 positions=("MID",), seed=0, retrain_every=4)
    np.testing.assert_array_equal(result.loc[result.GW_global < first + 4, "mase_scale"],
                                  second.loc[second.GW_global < first + 4, "mase_scale"])


@pytest.mark.parametrize("history_rows", ["constant", "single_round"])
def test_invalid_training_round_scale_fails_before_fit(monkeypatch, history_rows):
    frame = fixture_frame()
    first = screen.protocol_cutoffs()["selection_min_gw"]
    if history_rows == "constant":
        frame.loc[frame.GW_global < first, "total_points"] = 1
    else:
        frame = frame[frame.GW_global >= first - 1]
    monkeypatch.setattr(screen.models, "build_registered_model", lambda *a, **k: pytest.fail("fitted invalid scale"))
    with pytest.raises(ValueError, match="MASE scale must be finite and positive"):
        screen.generate_oof(frame, ["x"], {"lightgbm_l2": {}},
                            positions=("MID",), seed=0, retrain_every=4)


@pytest.mark.parametrize("count", [0, 1.5, np.nan, np.inf, 2])
def test_invalid_or_inconsistent_fixture_counts_are_rejected(count):
    oof = panel()
    oof["fixture_count"] = 1.0
    oof.loc[0, "fixture_count"] = count
    with pytest.raises(ValueError):
        screen.compare_ensembles(oof, positions=("MID",), warmup_weeks=4)


def test_screen_rejects_gw232_before_feature_construction(tmp_path, monkeypatch):
    from fpl.data.provenance import FrozenDatasetProvenanceError
    dataset = tmp_path / "future.csv"
    pd.DataFrame({"season": ["2026-27"], "GW": [4], "GW_global": [232]}).to_csv(dataset, index=False)
    monkeypatch.setattr(screen.features, "build_feature_frame", lambda raw: pytest.fail("future data reached features"))
    output = tmp_path / "rejected"
    with pytest.raises(FrozenDatasetProvenanceError, match="cutoff 232"):
        screen.main(["--models", "lightgbm_l2", "--dataset", str(dataset), "--output-dir", str(output)])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (output / "member_oof.csv").exists()
