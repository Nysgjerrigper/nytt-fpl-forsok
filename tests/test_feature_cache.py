import json

import pandas as pd
import pytest

from fpl.data import feature_cache
from fpl.data.provenance import FrozenDatasetProvenanceError, dataframe_fingerprint


def test_feature_cache_hits_without_rebuilding(tmp_path, monkeypatch):
    raw = pd.DataFrame({"value": [1, 2]})
    source = tmp_path / "features.py"
    source.write_text("version one")
    calls = []

    def build(frame):
        calls.append(True)
        return frame.assign(feature=frame["value"] * 2)

    monkeypatch.setattr(feature_cache.features, "build_feature_frame", build)
    first = feature_cache.load_or_build_feature_frame(
        raw, "a" * 64, tmp_path / "cache", source
    )
    second = feature_cache.load_or_build_feature_frame(
        raw, "a" * 64, tmp_path / "cache", source
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1


def test_feature_cache_invalidates_on_feature_code_change(tmp_path, monkeypatch):
    raw = pd.DataFrame({"value": [1]})
    source = tmp_path / "features.py"
    source.write_text("version one")
    calls = []
    monkeypatch.setattr(
        feature_cache.features, "build_feature_frame",
        lambda frame: calls.append(True) or frame.assign(feature=len(calls)),
    )
    first = feature_cache.load_or_build_feature_frame(raw, "b" * 64, tmp_path / "cache", source)
    source.write_text("version two")
    second = feature_cache.load_or_build_feature_frame(raw, "b" * 64, tmp_path / "cache", source)
    assert first["feature"].iloc[0] == 1
    assert second["feature"].iloc[0] == 2


def test_feature_cache_rebuilds_when_artifact_is_tampered(tmp_path, monkeypatch):
    raw = pd.DataFrame({"value": [1]})
    source = tmp_path / "features.py"
    source.write_text("stable")
    calls = []
    monkeypatch.setattr(
        feature_cache.features, "build_feature_frame",
        lambda frame: calls.append(True) or frame.assign(feature=len(calls)),
    )
    feature_cache.load_or_build_feature_frame(raw, "c" * 64, tmp_path / "cache", source)
    metadata_path = next((tmp_path / "cache").glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    frame_path = metadata_path.with_suffix(".pkl")
    frame_path.write_bytes(b"corrupt")
    rebuilt = feature_cache.load_or_build_feature_frame(raw, "c" * 64, tmp_path / "cache", source)
    assert rebuilt["feature"].iloc[0] == 2


def test_frozen_cache_wrapper_rejects_partial_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(
        feature_cache,
        "load_frozen_dataset_manifest",
        lambda: {"dataset_sha256": "d" * 64, "data_rows": 2, "max_gw_global": 3},
    )
    partial = pd.DataFrame({"GW_global": [3]})
    with pytest.raises(FrozenDatasetProvenanceError, match="complete registered"):
        feature_cache.load_frozen_feature_frame(partial, cache_dir=tmp_path)


def test_frozen_cache_wrapper_rejects_unadmitted_same_shape_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(
        feature_cache,
        "load_frozen_dataset_manifest",
        lambda: {"dataset_sha256": "d" * 64, "data_rows": 1, "max_gw_global": 3},
    )
    unadmitted = pd.DataFrame({"GW_global": [3]})
    with pytest.raises(FrozenDatasetProvenanceError, match="complete registered"):
        feature_cache.load_frozen_feature_frame(unadmitted, cache_dir=tmp_path)


def _admitted_frame(monkeypatch):
    raw = pd.DataFrame({"GW_global": [1, 3], "total_points": [2, 4]})
    raw.attrs["frozen_dataset_sha256"] = "e" * 64
    raw.attrs["frozen_dataframe_fingerprint"] = dataframe_fingerprint(raw)
    monkeypatch.setattr(feature_cache, "load_frozen_dataset_manifest",
                        lambda: {"dataset_sha256": "e" * 64, "data_rows": 2, "max_gw_global": 3})
    return raw


@pytest.mark.parametrize("mutation", ["target", "index", "column", "dtype", "missing_fingerprint"])
def test_frozen_wrapper_rejects_mutated_admitted_frame(monkeypatch, tmp_path, mutation):
    raw = _admitted_frame(monkeypatch)
    if mutation == "target":
        raw.loc[0, "total_points"] = 999
    elif mutation == "index":
        raw.index = [10, 20]
    elif mutation == "column":
        raw.rename(columns={"total_points": "altered"}, inplace=True)
    elif mutation == "dtype":
        raw["total_points"] = raw.total_points.astype(float)
    else:
        del raw.attrs["frozen_dataframe_fingerprint"]
    monkeypatch.setattr(feature_cache, "load_or_build_feature_frame", lambda *a, **k: pytest.fail("mutation reached cache"))
    with pytest.raises(FrozenDatasetProvenanceError, match="complete registered"):
        feature_cache.load_frozen_feature_frame(raw, cache_dir=tmp_path)


def test_frozen_wrapper_allows_an_exact_admitted_copy(monkeypatch, tmp_path):
    raw = _admitted_frame(monkeypatch)
    seen = []
    monkeypatch.setattr(feature_cache, "load_or_build_feature_frame", lambda frame, *a, **k: seen.append(frame) or frame)
    copied = raw.copy()
    result = feature_cache.load_frozen_feature_frame(copied, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(result, raw)
    assert seen[0] is copied


def test_cache_identity_includes_raw_frame_content(tmp_path, monkeypatch):
    raw = pd.DataFrame({"value": [1]})
    source = tmp_path / "features.py"
    source.write_text("stable")
    calls = []
    monkeypatch.setattr(feature_cache.features, "build_feature_frame",
                        lambda frame: calls.append(True) or frame.assign(feature=frame.value * 2))
    first = feature_cache.load_or_build_feature_frame(raw, "f" * 64, tmp_path / "cache", source)
    changed = raw.copy()
    changed.loc[0, "value"] = 8
    second = feature_cache.load_or_build_feature_frame(changed, "f" * 64, tmp_path / "cache", source)
    assert first.feature.tolist() == [2]
    assert second.feature.tolist() == [16]
    assert len(calls) == 2


def test_frozen_features_retain_verified_dataset_bytes_when_builder_drops_attrs(monkeypatch, tmp_path):
    raw = _admitted_frame(monkeypatch)
    rebuilt = pd.DataFrame({"feature": [1, 2]})
    monkeypatch.setattr(feature_cache, "load_or_build_feature_frame", lambda *a, **k: rebuilt)
    result = feature_cache.load_frozen_feature_frame(raw, cache_dir=tmp_path)
    assert result.attrs["frozen_dataset_sha256"] == "e" * 64
