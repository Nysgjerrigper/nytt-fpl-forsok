"""
Tests for the 3-class minutes hurdle (minutes-model v2, fpl.model.models.ThreeClassHurdle):
E[points] = P(cameo) x E[pts | cameo] + P(full) x E[pts | full].

Synthetic frames with unambiguous minutes structure - the contract under test is the
class decomposition plumbing (three minute regimes learned from the label, per-class
regressors fit on their own rows only, degenerate fallbacks), not forecast accuracy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import models


def _frame(n_gws=40, seed=0):
    """Three archetypes, one per minutes class: starters (90', score ~6), supersubs
    (20' cameos, score ~1), benchwarmers (0', score 0). A feature encodes the archetype
    so both stages have something learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(12):
        kind = pid % 3  # 0=starter, 1=supersub, 2=benchwarmer
        for gw in range(1, n_gws + 1):
            minutes = {0: 90.0, 1: 20.0, 2: 0.0}[kind]
            points = {0: max(0.0, rng.normal(6, 1)), 1: max(0.0, rng.normal(1, 0.2)), 2: 0.0}[kind]
            rows.append({
                "player_id": pid, "GW_global": gw, "position": "MID",
                "kind_feat": float(kind) + rng.normal(0, 0.01),
                "noise_feat": rng.normal(),
                "minutes": minutes, "total_points": points,
            })
    return pd.DataFrame(rows)


FEATS = ["kind_feat", "noise_feat"]


def test_minute_class_boundaries():
    h = models.ThreeClassHurdle()
    assert h._minute_class([0, 1, 59, 60, 90]).tolist() == [0, 1, 1, 2, 2]


def test_three_regimes_learned_and_ordered():
    df = _frame()
    model = models.fit_model("catboost_hurdle3", df[FEATS], df["total_points"],
                             position="MID", minutes=df["minutes"])
    preds = {kind: model.predict(df[df["player_id"] % 3 == kind][FEATS].head(5)).mean()
             for kind in (0, 1, 2)}
    # Starters >> supersubs >> benchwarmers, and each near its regime's mean.
    assert preds[0] > 4.0
    assert 0.3 < preds[1] < 2.5
    assert preds[2] < 0.5


def test_missing_minutes_label_raises():
    df = _frame()
    try:
        models.fit_model("catboost_hurdle3", df[FEATS], df["total_points"], position="MID")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "minutes" in str(e)


def test_degenerate_single_class_falls_back():
    # Everyone always plays full games: classifier degenerates to a constant, the
    # full-game regressor still learns, and predictions stay finite/positive.
    df = _frame()
    df = df[df["player_id"] % 3 == 0]
    model = models.fit_model("catboost_hurdle3", df[FEATS], df["total_points"],
                             position="MID", minutes=df["minutes"])
    preds = model.predict(df[FEATS].head(5))
    assert np.isfinite(preds).all() and (preds > 3.0).all()


def test_nobody_ever_plays_predicts_zero():
    df = _frame()
    df = df[df["player_id"] % 3 == 2]
    model = models.fit_model("catboost_hurdle3", df[FEATS], df["total_points"],
                             position="MID", minutes=df["minutes"])
    assert (model.predict(df[FEATS].head(5)) == 0.0).all()
