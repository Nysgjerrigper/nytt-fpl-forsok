"""Synthetic admission contracts; timestamp assertions do not audit real sources."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fpl import config
from fpl.data.external import load_external_records, map_external_players, latest_available_records


def records():
    return pd.DataFrame({"season": ["2023-24"] * 3, "external_player_id": ["007"] * 3,
                         "event_id": ["001", "001", "002"],
                         "event_time": ["2023-09-01T12:00:00Z"] * 2 + ["2023-09-03T12:00:00Z"],
                         "available_at": ["2023-09-02T12:00:00Z", "2023-09-04T12:00:00Z", "2023-09-05T12:00:00Z"],
                         "stat": [np.nan, 0., 2.]})


def files(tmp_path, frame=None, **changes):
    csv = tmp_path / "source.csv"
    manifest_path = tmp_path / "manifest.json"
    (records() if frame is None else frame).to_csv(csv, index=False)
    manifest = {"schema": config.EXTERNAL_SOURCE_SCHEMA, "source": "synthetic",
                "revision": "a" * 40, "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "snapshot_available_at": "2023-09-06T12:00:00Z", "timing_basis": "archived_capture",
                "publication_evidence": "https://example.org/archive/snapshot"}
    manifest.update(changes)
    manifest_path.write_text(json.dumps(manifest))
    return csv, manifest_path


def load(tmp_path, frame=None, **changes):
    return load_external_records(*files(tmp_path, frame, **changes), max_available_at="2023-09-07T12:00:00Z")


def mapping():
    return pd.DataFrame({"season": ["2023-24"], "external_player_id": ["007"], "fpl_element_id": [1]})


def roster():
    return pd.DataFrame({"season": ["2023-24"], "fpl_element_id": [1], "player_id": [1001], "position": ["MID"]})


def mapped(frame=None, links=None, players=None, **kwargs):
    return map_external_players(records() if frame is None else frame, mapping() if links is None else links,
                                roster() if players is None else players,
                                mapping_available_at="2023-09-01T10:00:00Z",
                                roster_available_at="2023-09-01T11:00:00Z", **kwargs)


def test_verified_bytes_preserve_opaque_ids_missing_statistics_and_attrs(tmp_path):
    result = load(tmp_path)
    assert result.external_player_id.tolist() == ["007"] * 3
    assert result.event_id.tolist() == ["001", "001", "002"]
    assert pd.isna(result.stat.iloc[0]) and result.stat.iloc[1] == 0.
    assert str(result.available_at.dt.tz) == "UTC"
    assert result.attrs["source"] == "synthetic" and result.attrs["revision"] == "a" * 40
    assert len(result.attrs["input_sha256"]) == 64


def test_sha_rejected_before_csv_parsing(tmp_path):
    csv, manifest = files(tmp_path)
    csv.write_bytes(b"not even csv\xff")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_external_records(csv, manifest, max_available_at="2023-09-07T12:00:00Z")


@pytest.mark.parametrize("changes", [{"schema": "bad"}, {"source": ""}, {"revision": "x" * 40},
    {"sha256": "a" * 63}, {"timing_basis": "git_commit_time"}, {"timing_basis": "match_time"},
    {"timing_basis": "unknown"}, {"publication_evidence": "http://example.org"},
    {"snapshot_available_at": "2023-09-06"}, {"snapshot_available_at": "2023-09-06T12:00:00"},
    {"snapshot_available_at": "2023-09-07T12:00:00Z"}])
def test_invalid_manifest(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


@pytest.mark.parametrize("column,value", [("event_time", "2023-09-01"), ("available_at", "2023-09-02T12:00:00"),
    ("available_at", "NaT"), ("event_time", "2023-02-30T12:00:00Z"), ("available_at", "2023-09-07T12:00:00Z"),
    ("event_time", "2023-09-05T12:00:00Z"), ("external_player_id", ""), ("season", " ")])
def test_invalid_record_columns(tmp_path, column, value):
    frame = records()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        load(tmp_path, frame)


def test_duplicate_versions_rejected_but_distinct_revisions_allowed(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load(tmp_path, pd.concat([records(), records().iloc[[0]]]))
    result = load(tmp_path)
    assert len(result) == 3
    selected = latest_available_records(result, "2023-09-04T13:00:00Z")
    assert selected.event_id.tolist() == ["001"] and selected.stat.tolist() == [0.]
    assert latest_available_records(result, "2023-09-02T12:00:00Z").empty
    assert latest_available_records(result, "2023-09-07T12:00:00Z", excluded_event_ids=("001",)).event_id.tolist() == ["002"]


def test_late_publication_not_early_match_controls_admission():
    assert latest_available_records(records(), "2023-09-02T10:00:00Z").empty
    assert len(latest_available_records(records(), "2023-09-03T12:00:00Z")) == 1


def test_mapping_composes_availability_and_preserves_order():
    frame = records().iloc[[2, 0]].copy()
    frame.index = [7, 2]
    result = map_external_players(frame, mapping(), roster(), mapping_available_at="2023-09-06T00:00:00+02:00",
                                  roster_available_at="2023-09-06T01:00:00Z")
    assert result.index.tolist() == [7, 2] and result.event_id.tolist() == ["002", "001"]
    assert result.player_id.tolist() == [1001, 1001]
    assert (result.available_at == pd.Timestamp("2023-09-06T01:00:00Z")).all()
    assert latest_available_records(result, "2023-09-06T01:00:00Z").empty


def test_season_scopes_element_and_external_id_resets():
    frame = records().iloc[[0, 1]].copy()
    frame.loc[1, "season"] = "2024-25"
    links = pd.concat([mapping(), mapping().assign(season="2024-25")])
    players = pd.concat([roster(), roster().assign(season="2024-25", player_id=2002)])
    assert mapped(frame, links, players).player_id.tolist() == [1001, 2002]


def test_unmatched_strict_and_optional_explicit():
    frame = records().assign(external_player_id="999")
    with pytest.raises(ValueError, match="unmatched"):
        mapped(frame)
    result = mapped(frame, allow_unmapped=True)
    assert result.player_id.isna().all() and result.position.isna().all()
    assert result.mapping_status.tolist() == ["unmapped"] * 3


@pytest.mark.parametrize("table,key", [("mapping", "external_player_id"), ("mapping", "fpl_element_id"),
                                       ("roster", "fpl_element_id"), ("roster", "player_id")])
def test_one_to_one_mapping_required(table, key):
    links, players = mapping(), roster()
    if table == "mapping":
        other = links.assign(fpl_element_id=2) if key == "external_player_id" else links.assign(external_player_id="008")
        links = pd.concat([links, other])
    else:
        other = players.assign(player_id=1002) if key == "fpl_element_id" else players.assign(fpl_element_id=2)
        players = pd.concat([players, other])
    with pytest.raises(ValueError, match="one-to-one"):
        mapped(links=links, players=players)


@pytest.mark.parametrize("key,value", [("player_id", 0), ("player_id", 1.5), ("fpl_element_id", np.nan), ("position", "wing")])
def test_invalid_roster_identity_and_positions(key, value):
    players = roster()
    players[key] = value
    with pytest.raises(ValueError):
        mapped(players=players)


def test_identity_collisions_and_numeric_opaque_ids_fail():
    with pytest.raises(ValueError, match="already contain"):
        mapped(records().assign(player_id=1001))
    with pytest.raises(ValueError, match="string identifiers"):
        mapped(records().assign(external_player_id=7))
    frame = records()
    frame["external_player_id"] = frame["external_player_id"].astype(object)
    frame.loc[0, "external_player_id"] = 7
    with pytest.raises(ValueError, match="string identifiers"):
        mapped(frame)


def test_latest_empty_and_ambiguous_versions():
    empty = records().iloc[:0]
    assert latest_available_records(empty, "2023-09-10T00:00:00Z").empty
    with pytest.raises(ValueError, match="duplicate"):
        latest_available_records(pd.concat([records(), records().iloc[[0]]]), "2023-09-10T00:00:00Z")
    with pytest.raises(ValueError, match="timezone-aware"):
        latest_available_records(records(), "2023-09-10")


def test_provider_publication_and_timezone_equivalence(tmp_path):
    frame = records()
    frame.loc[0, "event_time"] = "2023-09-01T14:00:00+02:00"
    result = load(tmp_path, frame, timing_basis="provider_publication")
    assert result.event_time.iloc[0] == result.event_time.iloc[1]


def test_mapping_naive_snapshot_and_missing_roster_match_fail():
    with pytest.raises(ValueError, match="timezone-aware"):
        map_external_players(records(), mapping(), roster(), mapping_available_at="2023-09-01T00:00:00",
                             roster_available_at="2023-09-01T00:00:00Z")
    with pytest.raises(ValueError, match="unmatched"):
        mapped(players=roster().assign(fpl_element_id=2))


def test_mapping_attrs_and_blank_ids_preserved_or_rejected(tmp_path):
    source = load(tmp_path)
    result = mapped(source)
    assert result.attrs == source.attrs
    with pytest.raises(ValueError, match="string identifiers"):
        mapped(links=mapping().assign(external_player_id=" "))


def test_later_mapping_keeps_source_revision_order_independent_of_file_order():
    versions = records().iloc[:2]
    for frame in (versions, versions.iloc[::-1]):
        result = map_external_players(frame, mapping(), roster(),
                                      mapping_available_at="2023-09-06T00:00:00Z",
                                      roster_available_at="2023-09-05T00:00:00Z")
        assert result.available_at.nunique() == 1
        assert result.source_available_at.nunique() == 2
        assert latest_available_records(result, "2023-09-06T00:00:00Z").empty
        latest = latest_available_records(result, "2023-09-06T00:00:01Z")
        assert len(latest) == 1 and latest.stat.iloc[0] == 0.
        assert latest.source_available_at.iloc[0] == pd.Timestamp("2023-09-04T12:00:00Z")


def test_identical_source_revision_rejected_after_mapping():
    result = mapped(pd.concat([records().iloc[[0]], records().iloc[[0]]]))
    with pytest.raises(ValueError, match="duplicate"):
        latest_available_records(result, "2023-09-07T00:00:00Z")


@pytest.mark.parametrize("source_time", ["2023-09-01T00:00:00", "2023-09-07T00:00:00Z"])
def test_source_version_metadata_requires_timezone_and_no_backdating(source_time):
    result = mapped(records().iloc[[0]])
    result["source_available_at"] = source_time
    with pytest.raises(ValueError):
        latest_available_records(result, "2023-09-08T00:00:00Z")


def test_source_metadata_collision_rejected():
    with pytest.raises(ValueError, match="already contain"):
        mapped(records().assign(source_available_at="2023-09-01T00:00:00Z"))


@pytest.mark.parametrize("column", ["player_id", "fpl_element_id"])
def test_mixed_boolean_integer_roster_ids_rejected(column):
    players = pd.concat([roster(), roster().assign(fpl_element_id=2, player_id=1002)], ignore_index=True)
    players[column] = pd.Series([True, 2], dtype=object)
    with pytest.raises(ValueError, match="positive integer"):
        mapped(players=players)


def test_mixed_boolean_integer_mapping_ids_rejected():
    links = pd.concat([mapping(), mapping().assign(external_player_id="008", fpl_element_id=2)], ignore_index=True)
    links["fpl_element_id"] = pd.Series([True, 2], dtype=object)
    with pytest.raises(ValueError, match="positive integer"):
        mapped(links=links)
