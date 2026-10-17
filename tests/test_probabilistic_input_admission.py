"""All probabilistic file consumers admit frozen bytes before windowed features."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fpl import config, features
from fpl.data.provenance import FrozenDatasetProvenanceError, load_frozen_research_dataset
from fpl.model import probabilistic as quantiles
from fpl.model import probabilistic_buckets as buckets
from fpl.model import research_input


CONSUMERS = (
    (lambda: quantiles.evaluate_static_split(train_max_gw=136, test_min_gw=137, test_max_gw=152), 152),
    (lambda: buckets.evaluate_static_split(train_max_gw=136, test_min_gw=137, test_max_gw=152), 152),
    (lambda: buckets.evaluate_walk_forward(start_gw=137, end_gw=152), 152),
    (lambda: buckets.run_bucket_tuning(["coarse5"], train_max_gw=136), 136),
    (lambda: buckets.walk_forward_predictions_csv(start_gw=137, end_gw=152), 152),
)


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    rows = [
        {"season": "2023-24", "GW": 22, "GW_global": 136},
        {"season": "2023-24", "GW": 23, "GW_global": 137},
        {"season": "2026-27", "GW": 3, "GW_global": 231},
    ]
    dataset = tmp_path / "frozen.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "data_rows": 3,
        "max_gw_global": 231, "current_season": "2026-27", "current_season_max_gw": 3}))
    calls = []

    def admitted():
        calls.append("admission")
        return load_frozen_research_dataset(dataset, manifest)

    monkeypatch.setattr(research_input, "load_frozen_research_dataset", admitted)
    monkeypatch.setattr(config, "MASTER_DATASET_PATH", tmp_path / "mutable-input-must-not-be-read.csv")
    return dataset, calls


@pytest.mark.parametrize("consume,maximum", CONSUMERS)
def test_each_consumer_admits_full_snapshot_then_trims_before_features(snapshot, monkeypatch, consume, maximum):
    dataset, calls = snapshot
    original = dataset.read_bytes()

    class ReachedFeatures(Exception):
        pass

    def inspect(raw):
        assert calls == ["admission"]
        assert raw.GW_global.tolist() == ([136] if maximum == 136 else [136, 137])
        assert raw.attrs["frozen_dataset_sha256"] == hashlib.sha256(original).hexdigest()
        assert "frozen_dataframe_fingerprint" not in raw.attrs
        raise ReachedFeatures

    monkeypatch.setattr(features, "build_feature_frame", inspect)
    with pytest.raises(ReachedFeatures):
        consume()
    assert dataset.read_bytes() == original


@pytest.mark.parametrize("consume,maximum", CONSUMERS)
def test_changed_frozen_bytes_stop_every_consumer_before_features(snapshot, monkeypatch, consume, maximum):
    dataset, calls = snapshot
    dataset.write_bytes(dataset.read_bytes() + b"\n")
    monkeypatch.setattr(features, "build_feature_frame", lambda *a, **k: pytest.fail("unadmitted features"))
    with pytest.raises(FrozenDatasetProvenanceError, match="SHA-256"):
        consume()
    assert calls == ["admission"]


@pytest.mark.parametrize("end,start,train", [
    (232, 153, None), (0, 1, None), (152.0, 137, None), (True, 1, None),
    (152, False, None), (152, 153, None), (152, 137.0, None),
    (152, 137, 137), (152, 137, 138), (152, 137, 0), (152, 137, True),
])
def test_invalid_window_rejected_before_loading_or_features(monkeypatch, end, start, train):
    monkeypatch.setattr(research_input, "load_frozen_research_dataset", lambda: pytest.fail("invalid window loaded data"))
    with pytest.raises(ValueError):
        research_input.load_research_raw(end, start_gw=start, train_max_gw=train)


def test_tuning_cannot_enter_evaluation_or_spent_rounds(monkeypatch):
    monkeypatch.setattr(research_input, "load_frozen_research_dataset", lambda: pytest.fail("invalid tuning loaded data"))
    with pytest.raises(ValueError):
        buckets.run_bucket_tuning(["coarse5"], train_max_gw=config.TUNING_TRAIN_MAX_GW + 1)


def test_numpy_integer_window_retains_training_history(snapshot):
    raw = research_input.load_research_raw(np.int64(152), start_gw=np.int64(137), train_max_gw=np.int64(136))
    assert raw.GW_global.tolist() == [136, 137]


def test_empty_admitted_window_fails_before_features(snapshot):
    with pytest.raises(ValueError, match="empty|rows"):
        research_input.load_research_raw(100)
