"""Live discovery must not let an empty upcoming season discard valid history."""
import pandas as pd

from fpl.data import fetch


def test_master_build_skips_only_explicitly_empty_upcoming_season(monkeypatch):
    raw = pd.DataFrame({"GW": [1], "name": ["Player"], "team": ["A"], "position": ["MID"]})
    monkeypatch.setattr(fetch, "fetch_season_gws", lambda season: raw if season == "2024-25" else (
        (_ for _ in ()).throw(RuntimeError(f"No gameweek data found yet for season {season}."))
    ))
    monkeypatch.setattr(fetch, "clean_season", lambda frame, season: frame.copy())
    monkeypatch.setattr(fetch, "assign_player_ids", lambda frame: frame.assign(player_id=1))
    built = fetch.build_master_dataset(["2024-25", "2026-27"], save=False)
    assert built["GW_global"].tolist() == [1]
