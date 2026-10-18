"""Pinned origin-capture admission and future-metadata-independent preparation."""

from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fpl.milp.origin_prices import load_origin_captures, prepare_origin_frame, selling_prices


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    folder = tmp_path / "gw153"
    folder.mkdir()
    bootstrap = {"teams": [{"id": i, "name": f"club{i}"} for i in range(1, 21)],
        "elements": [{"id": i, "code": 1000 + i, "team": i, "element_type": i,
                      "now_cost": 54 if i == 1 else 47, "web_name": f"player{i}"} for i in (1, 2)],
        "events": [{"id": 1, "deadline_time": "2024-08-16T17:30:00Z"},
                   {"id": 2, "deadline_time": "2024-08-23T17:30:00Z"}]}
    payload = json.dumps(bootstrap).encode()
    compressed = gzip.compress(payload)
    manifest = {"event": 1, "event_deadline": "2024-08-16T17:30:00Z", "captured_at": "2024-08-16T16:30:00Z",
        "backfill": {"season": "2024-25"}, "files": [{"name": "bootstrap-static.json.gz", "endpoint": "/bootstrap-static/", "sha256": _sha(payload)}]}
    (folder / "bootstrap-static.json.gz").write_bytes(compressed)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    descriptor = {"schema_version": 1, "season": "2024-25", "captures": {"153": {"folder": "gw153", "input_sha256": {
        "manifest.json": _sha((folder / "manifest.json").read_bytes()), "bootstrap-static.json.gz": _sha(compressed)}}}}
    path = tmp_path / "captures.json"
    path.write_text(json.dumps(descriptor))
    return path


def _edit_json(path, change):
    value = json.loads(path.read_bytes())
    change(value)
    path.write_text(json.dumps(value))


def _repin(path):
    _edit_json(path, lambda descriptor: descriptor["captures"]["153"].__setitem__("input_sha256", {
        name: _sha((path.parent / "gw153" / name).read_bytes()) for name in ("manifest.json", "bootstrap-static.json.gz")}))


def _edit_register(path, change):
    target = path.parent / "gw153" / "bootstrap-static.json.gz"
    bootstrap = json.loads(gzip.decompress(target.read_bytes()))
    change(bootstrap)
    payload = json.dumps(bootstrap).encode()
    target.write_bytes(gzip.compress(payload))
    _edit_json(path.parent / "gw153" / "manifest.json", lambda value: value["files"][0].__setitem__("sha256", _sha(payload)))
    _repin(path)


def _load(path):
    return load_origin_captures(path, season="2024-25", origins=[153])[153]


def _forecasts():
    return pd.DataFrame({"player_id": [1001, 1001, 9999], "GW": [153, 154, 153], "origin_gw": [153] * 3,
        "predicted_total_points": [3., 4., 10.], "fixture": [1, 2, 3],
        "name": ["future_name"] * 3, "team": ["future_club"] * 3, "position": ["FWD"] * 3,
        "value": [200, 300, 400], "actual_total_points": [100, 200, 300], "minutes": [90, 90, 90]})


def test_admitted_capture_binds_register_metadata_hashes_and_aware_timing(tmp_path):
    path = _fixture(tmp_path)
    capture = _load(path)
    metadata = capture["metadata"]
    assert metadata.index.tolist() == [1001, 1002]
    assert metadata.loc[1001].to_dict() == {"name": "player1", "team": "club1", "position": "GK", "value": 54}
    assert capture["receipt"]["origin_gw"] == 153
    assert capture["receipt"]["event"] == 1
    assert pd.Timestamp(capture["receipt"]["captured_at"]).tzinfo is not None
    assert capture["receipt"]["descriptor_sha256"] == _sha(path.read_bytes())
    assert len(capture["receipt"]["payload_sha256"]) == 64


@pytest.mark.parametrize("change", ["extra_field", "wrong_schema", "wrong_season", "noncanonical_origin", "missing_origin"])
def test_descriptor_requires_exact_schema_and_requested_coverage(tmp_path, change):
    path = _fixture(tmp_path)
    def mutate(value):
        if change == "extra_field": value["execute"] = "arbitrary"
        elif change == "wrong_schema": value["schema_version"] = True
        elif change == "wrong_season": value["season"] = "2025-26"
        elif change == "noncanonical_origin": value["captures"]["0153"] = value["captures"].pop("153")
        else: value["captures"] = {}
    _edit_json(path, mutate)
    with pytest.raises(ValueError): _load(path)


@pytest.mark.parametrize("name", ["manifest.json", "bootstrap-static.json.gz"])
def test_changed_pinned_capture_input_is_rejected(tmp_path, name):
    path = _fixture(tmp_path)
    target = path.parent / "gw153" / name
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        _load(path)


@pytest.mark.parametrize("change", ["late", "stale", "naive", "wrong_deadline", "wrong_event", "wrong_backfill", "wrong_payload", "wrong_endpoint"])
def test_manifest_payload_event_and_capture_timing_are_checked(tmp_path, change):
    path = _fixture(tmp_path)
    def mutate(value):
        if change == "late": value["captured_at"] = "2024-08-16T17:30:01Z"
        elif change == "stale": value["captured_at"] = "2024-08-15T16:00:00Z"
        elif change == "naive": value["captured_at"] = "2024-08-16T16:30:00"
        elif change == "wrong_deadline": value["event_deadline"] = "2024-08-17T17:30:00Z"
        elif change == "wrong_event": value["event"] = 2
        elif change == "wrong_backfill": value["backfill"]["season"] = "2025-26"
        elif change == "wrong_payload": value["files"][0]["sha256"] = "a" * 64
        else: value["files"][0]["endpoint"] = "/fixtures/"
    _edit_json(path.parent / "gw153" / "manifest.json", mutate)
    _repin(path)
    with pytest.raises(ValueError): _load(path)


@pytest.mark.parametrize("change", ["duplicate_code", "duplicate_element", "unknown_team", "bad_position", "bad_price",
                                    "blank_name", "wrong_first_year", "duplicate_team_name"])
def test_register_identities_and_season_are_strict(tmp_path, change):
    path = _fixture(tmp_path)
    def mutate(value):
        if change == "duplicate_code": value["elements"][1]["code"] = 1001
        elif change == "duplicate_element": value["elements"][1]["id"] = 1
        elif change == "unknown_team": value["elements"][0]["team"] = 99
        elif change == "bad_position": value["elements"][0]["element_type"] = True
        elif change == "bad_price": value["elements"][0]["now_cost"] = 54.0
        elif change == "blank_name": value["elements"][0]["web_name"] = " "
        elif change == "wrong_first_year": value["events"][0]["deadline_time"] = "2025-08-16T17:30:00Z"
        else: value["teams"][1]["name"] = value["teams"][0]["name"]
    _edit_register(path, mutate)
    with pytest.raises(ValueError): _load(path)


@pytest.mark.parametrize("escape", ["traversal", "symlink"])
def test_descriptor_cannot_escape_capture_root(tmp_path, escape):
    base = tmp_path / "base"
    base.mkdir()
    path = _fixture(base)
    if escape == "traversal":
        _edit_json(path, lambda value: value["captures"]["153"].__setitem__("folder", "../outside"))
    else:
        outside = tmp_path / "outside"
        (base / "gw153").rename(outside)
        (base / "gw153").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        _load(path)


def test_preparation_overlays_origin_identity_filters_pool_and_never_invents_missing_player(tmp_path):
    capture = _load(_fixture(tmp_path))
    forecasts = _forecasts()
    original = forecasts.copy(deep=True)
    prepared, receipt = prepare_origin_frame(forecasts, capture, origin_gw=153, owned=[1001])
    assert prepared.player_id.tolist() == [1001, 1001]
    assert prepared.GW.tolist() == [153, 154]
    assert prepared.value.tolist() == [54, 54]
    assert prepared.name.tolist() == ["player1", "player1"]
    assert prepared.team.tolist() == ["club1", "club1"]
    assert prepared.position.tolist() == ["GK", "GK"]
    assert prepared.fixture.tolist() == [1, 2]
    assert "actual_total_points" not in prepared and "minutes" not in prepared
    assert receipt["excluded_player_ids"] == [9999]
    assert receipt["excluded_forecast_rows"] == 1
    assert receipt["registered_without_forecasts"] == 1
    assert 1002 not in prepared.player_id.tolist()
    assert receipt["price_policy"] == "origin_capture_half_profit_static_horizon_v1"
    assert receipt["pool_policy"] == "registered_forecast_rows"
    pd.testing.assert_frame_equal(forecasts, original)


def test_future_metadata_prices_and_outcomes_cannot_change_prepared_forecasts(tmp_path):
    capture = _load(_fixture(tmp_path))
    first = _forecasts()
    changed = first.copy()
    for column in ("name", "team", "position", "value", "actual_total_points", "minutes"):
        changed[column] = "deliberately invalid future metadata"
    a, receipt_a = prepare_origin_frame(first, capture, origin_gw=153)
    b, receipt_b = prepare_origin_frame(changed, capture, origin_gw=153)
    pd.testing.assert_frame_equal(a, b)
    assert receipt_a == receipt_b


@pytest.mark.parametrize("owned", [[1002], [9999], [1001, 1001]])
def test_owned_players_require_registered_explicit_forecast_coverage(tmp_path, owned):
    capture = _load(_fixture(tmp_path))
    with pytest.raises(ValueError, match="owned players"):
        prepare_origin_frame(_forecasts(), capture, origin_gw=153, owned=owned)


@pytest.mark.parametrize("column,value", [("origin_gw", 154), ("origin_gw", True), ("GW", 152),
                                         ("player_id", "1001"), ("predicted_total_points", np.nan),
                                         ("predicted_total_points", True)])
def test_forecast_origin_identity_and_numeric_admission_are_strict(tmp_path, column, value):
    capture = _load(_fixture(tmp_path))
    frame = _forecasts()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError): prepare_origin_frame(frame, capture, origin_gw=153)


@pytest.mark.parametrize("change", ["bootstrap", "metadata"])
def test_mutated_admitted_capture_rejected_by_preparation_and_sales(tmp_path, change):
    capture = _load(_fixture(tmp_path))
    if change == "bootstrap": capture["bootstrap"]["elements"][0]["now_cost"] = 100
    else: capture["metadata"].loc[1001, "value"] = 100
    with pytest.raises(ValueError, match="changed"):
        prepare_origin_frame(_forecasts(), capture, origin_gw=153)
    with pytest.raises(ValueError, match="changed"):
        selling_prices(capture, [1001], {1001: 50})


def test_half_profit_and_loss_use_explicit_basis_and_rebuy_basis_resets(tmp_path):
    capture = _load(_fixture(tmp_path))
    before = deepcopy(capture["bootstrap"])
    assert selling_prices(capture, [1001, 1002], {1001: 50, 1002: 50}) == {1001: 52, 1002: 47}
    assert selling_prices(capture, [1001], {1001: 54}) == {1001: 54}
    assert capture["bootstrap"] == before


@pytest.mark.parametrize("basis", [{}, {1001: 0}, {1001: True}, {"1001": 50}, {1001: 50, 1002: 50}])
def test_selling_basis_must_exactly_cover_owned_numeric_codes(tmp_path, basis):
    capture = _load(_fixture(tmp_path))
    with pytest.raises(ValueError): selling_prices(capture, [1001], basis)
