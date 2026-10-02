"""Only a typed unstarted API-active season may be skipped during master ingestion."""
import pandas as pd
import pytest

from fpl.data import fetch


def test_master_build_skips_only_explicitly_unstarted_active_season(monkeypatch):
    raw = pd.DataFrame({"GW": [1], "name": ["Player"], "team": ["A"], "position": ["MID"]})
    monkeypatch.setattr(fetch, "_get_json", lambda url: {"events": [{"id": 1, "deadline_time": "2026-08-21T18:30:00Z"}]})
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda season: raw)
    monkeypatch.setattr(fetch, "fetch_live_season_gws", lambda season, **kw: (_ for _ in ()).throw(fetch.ActiveSeasonNotStartedError("unstarted")))
    monkeypatch.setattr(fetch, "clean_season", lambda frame, season: frame.copy())
    monkeypatch.setattr(fetch, "assign_player_ids", lambda frame: frame.assign(player_id=1))
    built = fetch.build_master_dataset(["2024-25", "2026-27"], save=False)
    assert built.GW_global.tolist() == [153]


def test_missing_completed_historical_season_is_not_skipped(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda season: (_ for _ in ()).throw(RuntimeError(f"No gameweek data found yet for season {season}.")))
    with pytest.raises(RuntimeError, match="No gameweek data"):
        fetch.build_master_dataset(["2024-25"], save=False, include_live=False)
