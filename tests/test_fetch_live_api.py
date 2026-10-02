"""Offline active-season ingestion, fixture identity, and atomic-save contracts."""
import copy

import numpy as np
import pandas as pd
import pytest
import requests

from fpl import config
from fpl.data import fetch


def payloads():
    bootstrap = {
        "events": [{"id": 3, "data_checked": True}, {"id": 2, "data_checked": False},
                   {"id": 1, "data_checked": True, "deadline_time": "2026-08-21T18:30:00Z"},
                   {"id": 4, "data_checked": False}],
        "teams": [{"id": 1, "name": "Home"}, {"id": 2, "name": "Away"}, {"id": 3, "name": "Current"}],
        "element_types": [{"id": 1, "singular_name_short": "GKP"}],
        "elements": [{"id": pid, "code": 1000 + pid, "first_name": "Test", "second_name": str(pid),
                      "element_type": 1, "team": 3, "ep_this": "99"} for pid in (20, 10)],
    }
    fixtures = [{"id": fid, "event": gw, "team_h": 1, "team_a": 2,
                 "team_h_difficulty": 2, "team_a_difficulty": 4}
                for fid, gw in ((101, 1), (102, 2), (103, 3), (104, 3))]
    histories = {
        pid: {"history": [{"element": pid, "fixture": fid, "round": gw, "was_home": home,
                           "opponent_team": 2 if home else 1, "minutes": 90, "total_points": 6,
                           "value": 50}
                          for fid, gw, home in ((103, 3, False), (101, 1, True), (102, 2, True), (104, 3, True))]}
        for pid in (10, 20)
    }
    return bootstrap, fixtures, histories


@pytest.fixture
def api(monkeypatch):
    bootstrap, fixtures, histories = payloads()
    calls = []

    def get(url):
        calls.append(url)
        if url.endswith("bootstrap-static/"):
            return copy.deepcopy(bootstrap)
        if url.endswith("fixtures/"):
            return copy.deepcopy(fixtures)
        if "element-summary" in url:
            return copy.deepcopy(histories[int(url.rstrip("/").split("/")[-1])])
        raise AssertionError(url)

    monkeypatch.setattr(fetch, "_get_json", get)
    return bootstrap, fixtures, histories, calls


def test_checked_membership_fixture_time_club_and_dgws_are_preserved(api):
    rows, teams, codes, fdr = fetch.fetch_live_season_gws("2026-27")
    assert rows.GW.tolist() == [1, 3, 3, 1, 3, 3]
    assert rows.fixture.tolist() == [101, 103, 104, 101, 103, 104]
    assert rows.team.tolist() == ["Home", "Away", "Home"] * 2
    assert rows.position.tolist() == ["GK"] * 6
    assert rows.xP.isna().all()
    assert codes == {20: 1020, 10: 1010}
    assert set(fdr.GW) == {1, 2, 3}
    assert len([url for url in api[3] if url.endswith("bootstrap-static/")]) == 1
    cleaned = fetch.clean_season(rows, "2026-27", teams_map=teams, codes_map=codes, fixture_diff=fdr)
    assert cleaned.player_code.tolist() == [1010] * 3 + [1020] * 3
    assert cleaned.opponent_team.tolist() == ["Away", "Home", "Away"] * 2


def test_injected_bootstrap_does_not_refetch_and_wrong_season_fails_before_summaries(api):
    bootstrap = api[0]
    fetch.fetch_live_season_gws("2026-27", bootstrap=bootstrap)
    assert not any(url.endswith("bootstrap-static/") for url in api[3])
    api[3].clear()
    with pytest.raises(ValueError, match="does not match API-active"):
        fetch.fetch_live_season_gws("2025-26", bootstrap=bootstrap)
    assert api[3] == []


@pytest.mark.parametrize("fault", ["duplicate_fixture", "missing_fixture", "foreign_player", "duplicate_history",
                                   "missing_identity", "opponent", "event", "stable_code", "position", "fdr"])
def test_bad_api_identity_or_metadata_is_rejected(api, fault):
    bootstrap, fixtures, histories, _ = api
    row = histories[10]["history"][1]
    if fault == "duplicate_fixture":
        fixtures.append(copy.deepcopy(fixtures[0]))
    elif fault == "missing_fixture":
        row["fixture"] = 999
    elif fault == "foreign_player":
        row["element"] = 999
    elif fault == "duplicate_history":
        histories[10]["history"].append(copy.deepcopy(row))
    elif fault == "missing_identity":
        row["fixture"] = None
    elif fault == "opponent":
        row["opponent_team"] = 1
    elif fault == "event":
        fixtures[0]["event"] = 3
    elif fault == "stable_code":
        bootstrap["elements"][0]["code"] = None
    elif fault == "position":
        bootstrap["elements"][0]["element_type"] = 999
    else:
        fixtures[0]["team_h_difficulty"] = np.nan
    with pytest.raises(ValueError):
        fetch.fetch_live_season_gws("2026-27")


def test_any_summary_worker_failure_aborts_without_archive_fallback(api, monkeypatch):
    original = fetch._get_json

    def failing(url):
        if url.endswith("element-summary/20/"):
            raise requests.Timeout("summary failed")
        return original(url)

    monkeypatch.setattr(fetch, "_get_json", failing)
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda *a: pytest.fail("stale archive fallback"))
    with pytest.raises(requests.Timeout, match="summary failed"):
        fetch.build_master_dataset(["2026-27"], save=False)


def test_checked_events_with_empty_summaries_are_not_an_unstarted_skip(api):
    for pid in api[2]:
        api[2][pid]["history"] = []
    with pytest.raises(RuntimeError, match="checked events have no player history"):
        fetch.build_master_dataset(["2026-27"], save=False)


def test_discovery_includes_api_active_missing_from_archive_and_bootstrap_fetched_once(api, monkeypatch):
    monkeypatch.setattr(fetch, "discover_seasons", lambda: ["2024-25"])
    archive = pd.DataFrame({"player_code": [999], "GW": [1], "name": ["Archive"], "team": ["Home"], "position": ["MID"]})
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda season: archive)
    original_clean = fetch.clean_season
    monkeypatch.setattr(fetch, "clean_season", lambda frame, season, **kw: original_clean(frame, season, **kw) if kw else frame.assign(season=season))
    built = fetch.build_master_dataset(save=False)
    assert set(built.season) == {"2024-25", "2026-27"}
    assert built.loc[built.season == "2024-25", "GW_global"].tolist() == [153]
    assert set(built.loc[built.season == "2026-27", "GW_global"]) == {229, 231}
    assert len([url for url in api[3] if url.endswith("bootstrap-static/")]) == 1


def test_archives_only_sort_deduplicate_and_preserve_canonical_season_gaps(monkeypatch):
    monkeypatch.setattr(fetch, "_get_json", lambda *a: pytest.fail("archive-only official API call"))
    calls = []
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda season: calls.append(season) or pd.DataFrame({"GW": [1], "player_code": [1]}))
    monkeypatch.setattr(fetch, "clean_season", lambda frame, season: frame.assign(season=season))
    built = fetch.build_master_dataset(["2024-25", "2020-21", "2024-25"], save=False, include_live=False)
    assert calls == ["2020-21", "2024-25"]
    assert built.GW_global.tolist() == [1, 153]


def test_explicit_request_order_does_not_select_wrong_live_season(api, monkeypatch):
    archived = []
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda season: archived.append(season) or pd.DataFrame({"GW": [1], "player_code": [1]}))
    real_clean = fetch.clean_season
    monkeypatch.setattr(fetch, "clean_season", lambda frame, season, **kw: real_clean(frame, season, **kw) if kw else frame.assign(season=season))
    built = fetch.build_master_dataset(["2026-27", "2024-25"], save=False)
    assert archived == ["2024-25"]
    assert set(built.season) == {"2024-25", "2026-27"}


def test_bootstrap_failure_never_uses_active_archive(monkeypatch):
    monkeypatch.setattr(fetch, "_get_json", lambda *a: (_ for _ in ()).throw(requests.Timeout("bootstrap failed")))
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda *a: pytest.fail("stale fallback"))
    with pytest.raises(requests.Timeout, match="bootstrap failed"):
        fetch.build_master_dataset(["2026-27"], save=False)


def test_frozen_path_protection_runs_before_network(monkeypatch, tmp_path):
    path = tmp_path / "frozen.csv"
    path.write_text("immutable bytes")
    monkeypatch.setattr(config, "MASTER_DATASET_PATH", path)
    monkeypatch.setattr(config, "FROZEN_RESEARCH_DATASET_PATH", path)
    monkeypatch.setattr(fetch, "_get_json", lambda *a: pytest.fail("network before frozen protection"))
    with pytest.raises(ValueError, match="equals frozen"):
        fetch.build_master_dataset(["2026-27"])
    assert path.read_text() == "immutable bytes"


def test_api_failure_preserves_existing_master_file(api, monkeypatch, tmp_path):
    path = tmp_path / "master.csv"
    path.write_text("existing master bytes")
    monkeypatch.setattr(config, "MASTER_DATASET_PATH", path)
    api[2][10]["history"][1]["fixture"] = 999
    with pytest.raises(ValueError):
        fetch.build_master_dataset(["2026-27"])
    assert path.read_text() == "existing master bytes"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["master.csv"]


def test_successful_save_uses_atomic_replace(api, monkeypatch, tmp_path):
    destination = tmp_path / "master.csv"
    destination.write_text("old bytes")
    monkeypatch.setattr(config, "MASTER_DATASET_PATH", destination)
    replacements = []
    real_replace = fetch.os.replace

    def replace(source, target):
        assert destination.read_text() == "old bytes"
        assert source.parent == destination.parent
        replacements.append((source, target))
        return real_replace(source, target)

    monkeypatch.setattr(fetch.os, "replace", replace)
    built = fetch.build_master_dataset(["2026-27"])
    assert len(replacements) == 1
    pd.testing.assert_frame_equal(pd.read_csv(destination), built, check_dtype=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["master.csv"]


def test_empty_injected_maps_do_not_trigger_archive_network(monkeypatch):
    raw = pd.DataFrame({"element": [10], "GW": [1], "opponent_team": [2], "team": ["Home"],
                        "name": ["Player"], "position": ["MID"], "was_home": [1]})
    for name in ("fetch_teams_map", "fetch_player_codes", "fetch_fixture_difficulty"):
        monkeypatch.setattr(fetch, name, lambda *a: pytest.fail("empty injected map triggered fallback"))
    fdr = pd.DataFrame({"team": ["Home"], "GW": [1], "fixture_difficulty": [2], "fixture_difficulty_next3": [2]})
    with pytest.raises(ValueError, match="join coverage"):
        fetch.clean_season(raw, "2026-27", teams_map={}, codes_map={}, fixture_diff=fdr)
