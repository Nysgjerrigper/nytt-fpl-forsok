"""
Guard for fpl.model.predict.origin_based_predictions (audit finding B2).

The property under test is the whole point of the export: a forecast issued at origin
GW t may only use information available at t's deadline. Tampering with outcomes AT or
AFTER t must therefore leave every origin-t prediction unchanged - including the
predictions for t+1/t+2, which is exactly where the standard walk-forward leaks future
form into the MILP's lookahead. Uses a tiny synthetic frame and the untuned LightGBM
member so the test stays fast and deterministic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl import features
from fpl.model import predict as predict_mod

N_GWS = 12
ORIGIN = 10


def _synthetic_raw(seed=0):
    """Two clubs' worth of MID players over 12 GWs with noisy but learnable scoring."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(8):
        team, opp = ("AAA", "BBB") if pid % 2 == 0 else ("BBB", "AAA")
        skill = 2.0 + pid * 0.5
        for gw in range(1, N_GWS + 1):
            rows.append({
                "player_id": pid, "name": f"p{pid}", "position": "MID",
                "team": team, "opponent_team": opp,
                "GW_global": gw, "was_home": gw % 2,
                "minutes": 90.0, "value": 50.0 + pid,
                "total_points": float(max(0, rng.normal(skill, 1.0))),
                "goals_scored": float(rng.integers(0, 2)),
                "goals_conceded": float(rng.integers(0, 3)),
                "xP": skill,
            })
    df = pd.DataFrame(rows)
    for col in features.FORM_STATS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _origin_preds(raw):
    df = features.build_feature_frame(raw)
    cols = features.feature_columns(df)
    return predict_mod.origin_based_predictions(
        df, raw, cols, start_gw=ORIGIN, end_gw=N_GWS, horizon=3,
        retrain_every=1, weight_window=4, weight_strategy="single:lightgbm",
    )


def test_origin_predictions_do_not_see_at_or_after_origin_outcomes():
    raw = _synthetic_raw()
    base = _origin_preds(raw)

    # Explode every outcome from the origin GW onward: deadline-time forecasts made AT
    # the origin cannot know any of this.
    tampered = raw.copy()
    at_or_after = tampered["GW_global"] >= ORIGIN
    tampered.loc[at_or_after, "total_points"] = 999.0
    after = _origin_preds(tampered)

    key = ["origin_gw", "GW", "player_id"]
    b = base[base["origin_gw"] == ORIGIN].sort_values(key).reset_index(drop=True)
    a = after[after["origin_gw"] == ORIGIN].sort_values(key).reset_index(drop=True)
    # Same players covered at the origin, for the whole horizon (t, t+1, t+2)...
    pd.testing.assert_frame_equal(a[key], b[key])
    # ...with identical forecasts: nothing at or after the origin reached them.
    np.testing.assert_allclose(a["predicted_total_points"], b["predicted_total_points"])
    # The actuals column, by contrast, must reflect the tampering (it is the scoring
    # ground truth, not a model input).
    assert (a["actual_total_points"] == 999.0).all()

    # Sanity on shape: every origin covers itself plus up to horizon-1 later GWs.
    for origin, grp in base.groupby("origin_gw"):
        assert grp["GW"].min() == origin
        assert grp["GW"].max() <= min(origin + 2, N_GWS)


def test_later_origins_do_use_newer_information():
    """Complement to the freeze test: the SAME target GW forecast from a LATER origin must
    move when pre-origin outcomes change - otherwise the export is frozen everywhere and
    measures nothing."""
    raw = _synthetic_raw()
    base = _origin_preds(raw)

    tampered = raw.copy()
    gw10 = tampered["GW_global"] == ORIGIN
    tampered.loc[gw10, "total_points"] = 999.0
    after = _origin_preds(tampered)

    key = ["origin_gw", "GW", "player_id"]
    b = base[base["origin_gw"] == ORIGIN + 1].sort_values(key).reset_index(drop=True)
    a = after[after["origin_gw"] == ORIGIN + 1].sort_values(key).reset_index(drop=True)
    assert not np.allclose(a["predicted_total_points"], b["predicted_total_points"])
