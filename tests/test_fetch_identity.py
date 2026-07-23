"""
Tests for the stable player-identity assignment in fpl.data.fetch (TODO 4.8):
player_id must be the FPL `code` where available, with name-based fallback ids
offset far above the real code range so the two can never collide.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl import config
from fpl.data.fetch import assign_player_ids


def _master(rows):
    return pd.DataFrame(rows, columns=["name", "player_code", "GW_global"])


def test_player_id_is_the_code_and_stable_across_seasons():
    # Same code in two different seasons (different per-season element ids upstream)
    # must yield ONE player_id; two players sharing a name must stay separate.
    master = assign_player_ids(_master([
        ("Ben Davies", 55605.0, 1),     # Spurs Ben Davies, season 1
        ("Ben Davies", 55605.0, 39),    # same person, next season
        ("Ben Davies", 199249.0, 39),   # the OTHER Ben Davies
    ]))
    assert master["player_id"].tolist() == [55605, 55605, 199249]


def test_missing_code_falls_back_to_offset_name_ids():
    master = assign_player_ids(_master([
        ("Known Player", 12345.0, 1),
        ("Ghost One", None, 1),
        ("Ghost One", None, 2),         # same missing-code name -> same fallback id
        ("Ghost Two", None, 1),
    ]))
    known = master.loc[master["name"] == "Known Player", "player_id"].iloc[0]
    ghosts = master.loc[master["name"].str.startswith("Ghost"), "player_id"]
    assert known == 12345
    # Fallback ids sit above the offset, are per-name consistent, and never collide
    # with real codes.
    assert (ghosts >= config.FALLBACK_PLAYER_ID_OFFSET).all()
    assert ghosts.nunique() == 2
    assert master.loc[master["name"] == "Ghost One", "player_id"].nunique() == 1
