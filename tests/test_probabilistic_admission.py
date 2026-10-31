"""Frozen admission, temporal bounds, and fixture identity for uncertainty research."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl.data import research
from fpl.data.provenance import FrozenDatasetProvenanceError
from fpl.model import probabilistic as quantiles
from fpl.model import probabilistic_buckets as buckets


def test_features_trim_only_after_complete_admission(monkeypatch):
    raw = pd.DataFrame({"GW_global": [0, 1, 152, 153, 231], "feature": [0, 1, 2, 3, 4]})
    raw.attrs.update(frozen_dataset_sha256="a" * 64, frozen_dataframe_fingerprint="whole")
    events = []

    def load(path):
        events.append(("load", path))
        return raw

    def build(subset):
        events.append(("build", subset.GW_global.tolist()))
        assert "frozen_dataframe_fingerprint" not in subset.attrs
        return subset.copy()

    monkeypatch.setattr(research, "load_frozen_research_dataset", load)
    monkeypatch.setattr(research, "load_frozen_dataset_manifest", lambda: {"max_gw_global": 231})
    monkeypatch.setattr(research.features, "build_feature_frame", build)
    monkeypatch.setattr(research.features, "feature_columns", lambda frame: ["feature"])
    subset, frame, columns = research.load_research_features(152, Path("frozen.csv"))
    assert events == [("load", Path("frozen.csv")), ("build", [1, 152])]
    assert subset.attrs == frame.attrs == {"frozen_dataset_sha256": "a" * 64}
    assert columns == ["feature"]
    assert raw.attrs["frozen_dataframe_fingerprint"] == "whole"


@pytest.mark.parametrize("cutoff", [0, -1, True, 1.5, 232])
def test_invalid_feature_cutoff_rejected_before_build(monkeypatch, cutoff):
    monkeypatch.setattr(research, "load_frozen_research_dataset", lambda path: pd.DataFrame())
    monkeypatch.setattr(research, "load_frozen_dataset_manifest", lambda: {"max_gw_global": 231})
    monkeypatch.setattr(research.features, "build_feature_frame", lambda raw: pytest.fail("features built"))
    with pytest.raises(ValueError):
        research.load_research_features(cutoff)


def test_admission_failure_propagates_before_features(monkeypatch):
    def reject(path):
        raise FrozenDatasetProvenanceError("hash mismatch")
    monkeypatch.setattr(research, "load_frozen_research_dataset", reject)
    monkeypatch.setattr(research.features, "build_feature_frame", lambda raw: pytest.fail("features built"))
    with pytest.raises(FrozenDatasetProvenanceError, match="hash mismatch"):
        research.load_research_features(152)


@pytest.mark.parametrize("module", [quantiles, buckets])
@pytest.mark.parametrize("window", [(153, 154, 183), (152, 152, 183), (152, 154, 153), (0, 153, 183)])
def test_static_windows_rejected_before_loading(monkeypatch, module, window):
    monkeypatch.setattr(module, "load_research_features", lambda *args: pytest.fail("input loaded"))
    with pytest.raises(ValueError):
        module.evaluate_static_split(*window)


@pytest.mark.parametrize("function", [buckets.evaluate_walk_forward, buckets.walk_forward_predictions_csv])
@pytest.mark.parametrize("kwargs", [{"start_gw": 154, "end_gw": 153}, {"retrain_every": 0}])
def test_walk_forward_invalid_bounds(monkeypatch, function, kwargs):
    monkeypatch.setattr(buckets, "load_research_features", lambda *args: pytest.fail("input loaded"))
    with pytest.raises(ValueError):
        function(**kwargs)


def test_tuning_cap_before_optuna_or_data_load(monkeypatch):
    monkeypatch.setattr(buckets, "load_research_features", lambda *args: pytest.fail("input loaded"))
    with pytest.raises(ValueError):
        buckets.run_bucket_tuning(["coarse5"], train_max_gw=153)
    with pytest.raises(ValueError):
        buckets.tune_bucket_position(None, [], "MID", None, train_max_gw=153)


@pytest.mark.parametrize("module,function,kwargs,cutoff", [
    (quantiles, quantiles.evaluate_static_split, {}, 183),
    (buckets, buckets.evaluate_static_split, {}, 183),
    (buckets, buckets.evaluate_walk_forward, {}, 183),
    (buckets, buckets.walk_forward_predictions_csv, {}, 183),
    (buckets, buckets.run_bucket_tuning, {"scheme_names": ["coarse5"]}, 152),
])
def test_entrypoints_forward_path_and_cutoff(monkeypatch, module, function, kwargs, cutoff):
    def stop(max_gw, dataset_path):
        assert max_gw == cutoff
        assert dataset_path == Path("verified.csv")
        raise FrozenDatasetProvenanceError("test admission rejection")
    monkeypatch.setattr(module, "load_research_features", stop)
    with pytest.raises(FrozenDatasetProvenanceError, match="test admission rejection"):
        function(dataset_path=Path("verified.csv"), **kwargs)


def test_export_retains_dgw_fixture_identity_and_minutes(monkeypatch):
    frame = pd.DataFrame({"player_id": [1, 1], "GW_global": [153, 153], "name": ["Player"] * 2,
                          "position": ["MID"] * 2, "team": [2, 2], "value": [50, 50],
                          "total_points": [3, 4], "minutes": [90, 45], "fixture": [10, 11],
                          "opponent_team": [3, 4], "was_home": [True, False], "feature": [1, 2]})
    monkeypatch.setattr(buckets, "load_research_features", lambda *args: (frame, frame, ["feature"]))

    class Model:
        bucket_values_ = np.array([2., 4.])
        def fit(self, train, features):
            return self
        def predict_distribution(self, test):
            return np.tile([.5, .5], (len(test), 1))

    monkeypatch.setattr(buckets, "candidate_models", lambda **kwargs: [Model()])
    def steps(df, start, end, retrain, fit):
        yield 153, fit(df), df
    monkeypatch.setattr(buckets, "walk_forward_steps", steps)
    result = buckets.walk_forward_predictions_csv()
    assert result.fixture.tolist() == [10, 11]
    assert result.minutes.tolist() == [90, 45]
    assert result.opponent_team.tolist() == [3, 4]
    assert result.was_home.tolist() == [True, False]
    assert result.GW.tolist() == [153, 153]
    assert result.actual_total_points.tolist() == [3, 4]
    assert result.predicted_total_points.tolist() == [3., 3.]


@pytest.mark.parametrize("module", [quantiles, buckets])
def test_cli_accepts_dataset(module):
    assert module.parse_args(["--dataset", "frozen.csv"]).dataset == Path("frozen.csv")


@pytest.mark.parametrize("module,arguments,target", [
    (quantiles, [], "evaluate_static_split"),
    (buckets, [], "evaluate_static_split"),
    (buckets, ["--walk-forward"], "evaluate_walk_forward"),
    (buckets, ["--tune"], "run_bucket_tuning"),
    (buckets, ["--export-predictions"], "walk_forward_predictions_csv"),
])
def test_cli_forwards_dataset_to_selected_entrypoint(monkeypatch, module, arguments, target):
    def stop(**kwargs):
        assert kwargs["dataset_path"] == Path("verified.csv")
        raise FrozenDatasetProvenanceError("test CLI admission rejection")
    monkeypatch.setattr(module, target, stop)
    with pytest.raises(FrozenDatasetProvenanceError, match="test CLI admission rejection"):
        module.main(arguments + ["--dataset", "verified.csv"])


def test_tuning_ceiling_follows_runtime_season_ordinals(monkeypatch):
    """A changed history origin shifts the registered selection cutoff immediately."""
    from fpl import config
    from fpl.model.moe_tournament import protocol_cutoffs

    monkeypatch.setattr(config, "DEFAULT_START_SEASON", "2021-22")
    ceiling = protocol_cutoffs()["selection_max_gw"]
    assert ceiling == 114
    research.validate_tuning_cutoff(ceiling)
    with pytest.raises(ValueError):
        research.validate_tuning_cutoff(ceiling + 1)
