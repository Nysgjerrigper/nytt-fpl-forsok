"""Pinned event collections preserve strict settlement and remain offline-readable."""
from copy import deepcopy
import gzip
import hashlib
import json
import shutil

import pandas as pd
import pytest

from fpl.milp import replay


def sha(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "_source_hashes", lambda: {"test": "fixed"})
    monkeypatch.setattr(replay, "_runtime_versions", lambda: {"test": "fixed"})
    root = tmp_path / "source"
    root.mkdir()
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    team_ids = {code: 1 if code == 12 else 2 + (code - 1) // 3 for code in range(1, 16)}
    teams = pd.DataFrame({"id": range(1, 21), "name": [f"club{i}" for i in range(1, 21)]})
    wheel, fixtures = list(range(1, 21)), []
    for event in range(1, 20):
        for index in range(10):
            home, away = wheel[index], wheel[-index-1]
            fixtures.extend([{"event": event, "team_h": home, "team_a": away},
                             {"event": event+19, "team_h": away, "team_a": home}])
        wheel = [wheel[0], wheel[-1], *wheel[1:-1]]
    fixtures = pd.DataFrame(fixtures)
    for event in (1, 3):
        fixtures.loc[(fixtures.event == event) & ((fixtures.team_h == 1) | (fixtures.team_a == 1)), "event"] = event+1
    fixtures = fixtures.assign(id=range(1, 381), finished=True, team_h_score=0, team_a_score=0)
    fixtures["kickoff_time"] = [(pd.Timestamp("2024-08-17T14:00Z") + pd.Timedelta(days=7*(event-1))).isoformat()
                                 for event in fixtures.event]
    deadlines = {event: (pd.Timestamp("2024-08-16T17:30Z") + pd.Timedelta(days=7*(event-1))).isoformat()
                 for event in (1, 3)}
    register = {"teams": teams.to_dict("records"),
                "elements": [{"code": code, "team": team_ids[code],
                              "element_type": {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[positions[code-1]]}
                             for code in range(1, 16)],
                "events": [{"id": event, "deadline_time": deadline} for event, deadline in deadlines.items()]}
    payload = json.dumps(register).encode()
    entries = []
    for event in (1, 3):
        folder = root / f"event-{event}"
        folder.mkdir()
        manifest = {"event": event, "event_deadline": deadlines[event],
                    "captured_at": (pd.Timestamp(deadlines[event])-pd.Timedelta(hours=1)).isoformat(),
                    "backfill": {"season": "2024-25"},
                    "files": [{"name": "bootstrap-static.json.gz", "endpoint": "/bootstrap-static/", "sha256": sha(payload)}]}
        assets = {"fixtures.csv": fixtures.to_csv(index=False).encode(),
                  "teams.csv": teams.to_csv(index=False).encode(),
                  "bootstrap-static.json.gz": gzip.compress(payload), "manifest.json": json.dumps(manifest).encode()}
        descriptor = {"schema_version": 1, "season": "2024-25", "event": event, "first_global_gw": 153,
                      "fixture_source_commit": "a"*40, "input_sha256": {name: sha(data) for name, data in assets.items()}}
        assets["blank-evidence.json"] = json.dumps(descriptor).encode()
        for name, data in assets.items():
            (folder / name).write_bytes(data)
        entries.append({"path": f"event-{event}/blank-evidence.json", "sha256": sha(assets["blank-evidence.json"])})
    collection = root / "collection.json"
    write_json(collection, {"schema_version": 1, "events": entries})
    row = {"gameweek": 153, "squad": list(range(1, 16)),
           "lineup": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15], "bench_order": [2, 6, 7, 12],
           "captain": [8], "vice_captain": [9], "chip_played": None, "alpha": 0}
    state = root / "snapshot.json"
    write_json(state, {"schema_version": 1, "settlement_completed": False,
                       "decisions": [row, dict(row, gameweek=155)]})
    frame = pd.DataFrame([{"player_id": code, "GW": gw, "position": positions[code-1],
                           "team": f"club{team_ids[code]}", "fixture": gw,
                           "actual_total_points": 1., "minutes": 90.}
                          for gw in (153, 155) for code in range(1, 16) if code != 12])
    predictions = root / "predictions.csv"
    frame.to_csv(predictions, index=False)
    return root, collection, state, predictions


def run(inputs, tmp_path):
    _, collection, state, predictions = inputs
    return replay.run_replay(state, predictions, tmp_path / "replayed", blank_evidence_path=collection)


def test_two_event_complete_ledger_settlement_and_offline_read(inputs, tmp_path):
    root, collection, _, predictions = inputs
    original = predictions.read_bytes()
    output = run(inputs, tmp_path)
    receipt = replay.read_replay(output)
    assert receipt["gross_points"] == receipt["net_points"] == 24
    assert receipt["explicit_blank_outcomes"] == 2
    audit = pd.read_csv(output / "explicit-blank-outcomes.csv")
    assert list(zip(audit.player_id, audit.GW)) == [(12, 153), (12, 155)]
    assert (output / "predictions.csv").read_bytes() == original
    assert receipt["registration_identity"] == "CAPTURE_TIME_ONLY"
    assert receipt["native_optimality"] == receipt["independent_financial_rule_legality"] == "NOT ASSESSED"
    assert receipt["promotion_eligible"] is False
    assert len([name for name in receipt["input_sha256"] if name.startswith("blank-evidence-events/")]) == 10
    shutil.rmtree(root)
    assert replay.read_replay(output) == receipt


@pytest.mark.parametrize("change", ["empty", "extra_key", "entry_key", "absolute", "traversal", "noncanonical", "hash", "duplicate_event", "changed_asset"])
def test_invalid_collection_rejected_before_output(inputs, tmp_path, change):
    root, collection, _, _ = inputs
    data = json.loads(collection.read_bytes())
    if change == "empty": data["events"] = []
    elif change == "extra_key": data["extra"] = 1
    elif change == "entry_key": data["events"][0]["extra"] = 1
    elif change == "absolute": data["events"][0]["path"] = str(root / data["events"][0]["path"])
    elif change == "traversal": data["events"][0]["path"] = "../source/event-1/blank-evidence.json"
    elif change == "noncanonical": data["events"][0]["path"] = "event-1//blank-evidence.json"
    elif change == "hash": data["events"][0]["sha256"] = "0"*64
    elif change == "duplicate_event": data["events"][1] = deepcopy(data["events"][0])
    else: (root / "event-3/fixtures.csv").write_text("changed")
    write_json(collection, data)
    with pytest.raises(ValueError): run(inputs, tmp_path)
    assert not (tmp_path / "replayed").exists()


@pytest.mark.parametrize("asset", [False, True])
def test_symlink_escape_rejected(inputs, tmp_path, asset):
    root, collection, _, _ = inputs
    source = root / "event-1" / ("fixtures.csv" if asset else "blank-evidence.json")
    external = tmp_path / source.name
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"): run(inputs, tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra", "changed", "rebound_extra", "rebound_missing", "rebound_descriptor", "output_only_extra"])
def test_read_rejects_incomplete_extra_or_modified_captured_evidence(inputs, tmp_path, change):
    output = run(inputs, tmp_path)
    receipt_path = output / "complete.json"
    receipt = json.loads(receipt_path.read_bytes())
    name = "blank-evidence-events/0002/manifest.json"
    if change in ("missing", "rebound_missing"):
        (output / name).unlink()
        if change == "rebound_missing":
            receipt["input_sha256"].pop(name)
            receipt["output_sha256"].pop(name)
    elif change in ("extra", "rebound_extra", "output_only_extra"):
        name = "blank-evidence-events/0003/extra.json"
        (output / name).parent.mkdir()
        (output / name).write_text("{}")
        if change in ("rebound_extra", "output_only_extra"):
            receipt["output_sha256"][name] = sha(b"{}")
        if change == "rebound_extra":
            receipt["input_sha256"][name] = sha(b"{}")
    elif change == "rebound_descriptor":
        name = "blank-evidence-events/0002/blank-evidence.json"
        data = json.loads((output / name).read_bytes())
        data["fixture_source_commit"] = "b"*40
        write_json(output / name, data)
        receipt["input_sha256"][name] = receipt["output_sha256"][name] = sha((output / name).read_bytes())
    else: (output / name).write_text("changed")
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError): replay.read_replay(output)


def test_per_event_strict_failure_preserves_failure_receipt(inputs, tmp_path):
    root, collection, _, _ = inputs
    descriptor_path = root / "event-3/blank-evidence.json"
    descriptor = json.loads(descriptor_path.read_bytes())
    manifest_path = root / "event-3/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["captured_at"] = "2024-09-01T00:00:00Z"  # After target deadline.
    write_json(manifest_path, manifest)
    descriptor["input_sha256"]["manifest.json"] = sha(manifest_path.read_bytes())
    write_json(descriptor_path, descriptor)
    data = json.loads(collection.read_bytes())
    data["events"][1]["sha256"] = sha(descriptor_path.read_bytes())
    write_json(collection, data)
    with pytest.raises(ValueError, match="predeadline"): run(inputs, tmp_path)
    assert (tmp_path / "replayed/failed.json").exists()
    assert not (tmp_path / "replayed/complete.json").exists()


def test_duplicate_blank_keys_rejected_before_scoring(inputs, monkeypatch):
    captured = replay._capture_blank_inputs(inputs[1])
    duplicate = pd.DataFrame({"player_id": [12], "GW": [153]})
    monkeypatch.setattr(replay, "_admit_blank_evidence", lambda *args: duplicate.copy())
    with pytest.raises(ValueError, match="duplicate blank player/gameweek"):
        replay._admit_blank_collection({}, pd.DataFrame(), captured)
