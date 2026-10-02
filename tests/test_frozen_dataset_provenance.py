"""Admission control for the immutable GW231 research snapshot."""
import hashlib
import json

import pandas as pd
import pytest

from fpl.data.provenance import FrozenDatasetProvenanceError, dataframe_fingerprint, load_frozen_research_dataset


def _write_snapshot(tmp_path, rows, **overrides):
    dataset = tmp_path / "master_dataset.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    manifest = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "data_rows": len(rows),
        "max_gw_global": 231,
        "current_season": "2026-27",
        "current_season_max_gw": 3,
    }
    manifest.update(overrides)
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(manifest))
    return dataset, manifest_path


def _rows():
    return [
        {"season": "2025-26", "GW": 38, "GW_global": 228, "total_points": 1},
        {"season": "2026-27", "GW": 1, "GW_global": 229, "total_points": 2},
        {"season": "2026-27", "GW": 2, "GW_global": 230, "total_points": 3},
        {"season": "2026-27", "GW": 3, "GW_global": 231, "total_points": 4},
    ]


def test_exact_manifest_bound_gw231_snapshot_loads(tmp_path):
    dataset, manifest = _write_snapshot(tmp_path, _rows())

    loaded = load_frozen_research_dataset(dataset, manifest)

    assert len(loaded) == 4
    assert int(loaded["GW_global"].max()) == 231


def test_gw232_is_rejected_before_hash_or_feature_construction(tmp_path):
    dataset, manifest = _write_snapshot(tmp_path, _rows())
    rows = _rows() + [{"season": "2026-27", "GW": 4, "GW_global": 232, "total_points": 5}]
    pd.DataFrame(rows).to_csv(dataset, index=False)

    with pytest.raises(FrozenDatasetProvenanceError, match="cutoff 232"):
        load_frozen_research_dataset(dataset, manifest)


def test_incomplete_gw229_snapshot_is_rejected(tmp_path):
    dataset, manifest = _write_snapshot(tmp_path, _rows()[:2])

    with pytest.raises(FrozenDatasetProvenanceError, match="cutoff 229"):
        load_frozen_research_dataset(dataset, manifest)


def test_byte_change_is_rejected_by_manifest_hash(tmp_path):
    dataset, manifest = _write_snapshot(tmp_path, _rows())
    dataset.write_bytes(dataset.read_bytes() + b"\n")

    with pytest.raises(FrozenDatasetProvenanceError, match="SHA-256"):
        load_frozen_research_dataset(dataset, manifest)


def test_current_season_provenance_is_checked_after_hash(tmp_path):
    rows = _rows()
    rows[-1]["GW"] = 2
    dataset, manifest = _write_snapshot(tmp_path, rows)

    with pytest.raises(FrozenDatasetProvenanceError, match="current-season"):
        load_frozen_research_dataset(dataset, manifest)


def test_loader_reads_once_and_parses_the_hashed_bytes(tmp_path, monkeypatch):
    from io import BytesIO
    from pathlib import Path
    from fpl.data import provenance
    dataset, manifest = _write_snapshot(tmp_path, _rows())
    expected_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    original_read = Path.read_bytes
    original_parse = provenance.pd.read_csv
    reads = []

    def read_once(path):
        if path == dataset:
            reads.append(path)
        return original_read(path)

    def mutate_during_parse(source, **kwargs):
        assert isinstance(source, BytesIO)
        dataset.write_text("season,GW,GW_global\n2026-27,4,232\n")
        return original_parse(source, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    monkeypatch.setattr(provenance.pd, "read_csv", mutate_during_parse)
    admitted = load_frozen_research_dataset(dataset, manifest)
    assert reads == [dataset]
    assert admitted.GW_global.max() == 231
    assert admitted.attrs["frozen_dataset_sha256"] == expected_digest
    assert admitted.attrs["frozen_dataframe_fingerprint"] == dataframe_fingerprint(admitted)


@pytest.mark.parametrize("mutation", ["value", "index", "column", "dtype"])
def test_content_fingerprint_detects_values_index_and_schema(mutation):
    frame = pd.DataFrame({"value": [1, 2]}, index=[10, 20])
    digest = dataframe_fingerprint(frame)
    assert dataframe_fingerprint(frame.copy()) == digest
    if mutation == "value":
        frame.loc[10, "value"] = 9
    elif mutation == "index":
        frame.index = [11, 20]
    elif mutation == "column":
        frame.columns = ["different"]
    else:
        frame["value"] = frame.value.astype(float)
    assert dataframe_fingerprint(frame) != digest
