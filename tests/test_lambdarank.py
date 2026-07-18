"""
Tests for the LambdaRank member (audit C3, fpl.model.models.LambdaRankScorer):
within-GW ranking, mapped isotonically back to the points scale.

Synthetic frames with an unambiguous within-round ordering - the contract under test
is the plumbing (GW group labels threaded through fit_model, group-contiguous sorting,
monotone score-to-points mapping), not NDCG accuracy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import models


def _frame(n_gws=40, n_players=12, seed=0):
    """Points are a noisy increasing function of one feature, so the learnable
    within-round ranking is simply 'higher quality_feat ranks higher'."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        quality = pid / n_players
        for gw in range(1, n_gws + 1):
            rows.append({
                "player_id": pid, "GW_global": gw,
                "quality_feat": quality + rng.normal(0, 0.05),
                "noise_feat": rng.normal(),
                "total_points": float(max(0, rng.normal(8 * quality, 1))),
            })
    return pd.DataFrame(rows)


FEATS = ["quality_feat", "noise_feat"]


def test_fit_model_requires_gw_label():
    df = _frame()
    with pytest.raises(ValueError, match="gw"):
        models.fit_model("lgbm_rank", df[FEATS], df["total_points"])


def test_learns_within_round_ranking():
    df = _frame()
    model = models.fit_model("lgbm_rank", df[FEATS], df["total_points"],
                             position="MID", gw=df["GW_global"])
    test = _frame(n_gws=1, seed=99)
    preds = pd.Series(model.predict(test[FEATS]), index=test.index)
    # Best-quality player must outrank worst-quality within the round.
    assert preds[test["player_id"].idxmax()] > preds[test["player_id"].idxmin()]


def test_predictions_are_on_the_points_scale():
    """The isotonic map must restore an absolute scale the MILP can consume:
    predictions live in the observed points range, not raw ranker-score space."""
    df = _frame()
    model = models.fit_model("lgbm_rank", df[FEATS], df["total_points"],
                             position="MID", gw=df["GW_global"])
    preds = model.predict(df[FEATS])
    assert preds.min() >= df["total_points"].min() - 1e-9
    assert preds.max() <= df["total_points"].max() + 1e-9
    # And the total level is broadly calibrated (isotonic fit on the same rows).
    assert 0.5 < preds.sum() / df["total_points"].sum() < 1.5


def test_unsorted_gw_input_is_handled():
    """Rows arriving in arbitrary order must not corrupt the group structure."""
    df = _frame().sample(frac=1.0, random_state=1).reset_index(drop=True)
    model = models.fit_model("lgbm_rank", df[FEATS], df["total_points"],
                             position="MID", gw=df["GW_global"])
    assert len(model.predict(df[FEATS])) == len(df)


def test_registered_in_model_names():
    assert "lgbm_rank" in models.MODEL_NAMES
