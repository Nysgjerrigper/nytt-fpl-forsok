"""
Tests for the two-stage hurdle member (audit C1, fpl.model.models.TwoStageHurdle):
E[points] = P(minutes > 0) x E[points | played].

Synthetic frames with unambiguous participation structure - the contract under test is
the decomposition plumbing (participation learned from the minutes label, regression
learned from played rows only, degenerate one-class fallbacks), not forecast accuracy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import models
from fpl.model.train import fit_position_ensembles


def _frame(n_gws=30, seed=0):
    """Two archetypes: 'nailed' players who always play and score ~5, and 'benchwarmers'
    who never play and score exactly 0. A feature encodes the archetype so both stages
    have something learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(10):
        nailed = pid < 5
        for gw in range(1, n_gws + 1):
            rows.append({
                "player_id": pid, "GW_global": gw, "position": "MID",
                "is_nailed_feat": float(nailed) + rng.normal(0, 0.01),
                "noise_feat": rng.normal(),
                "minutes": 90.0 if nailed else 0.0,
                "total_points": float(max(0, rng.normal(5, 1))) if nailed else 0.0,
            })
    return pd.DataFrame(rows)


FEATS = ["is_nailed_feat", "noise_feat"]


def test_fit_model_requires_minutes_label():
    df = _frame()
    with pytest.raises(ValueError, match="minutes"):
        models.fit_model("catboost_hurdle", df[FEATS], df["total_points"])


def test_hurdle_separates_participation_from_scoring():
    df = _frame()
    model = models.fit_model("catboost_hurdle", df[FEATS], df["total_points"],
                             minutes=df["minutes"])
    nailed = df[df["player_id"] < 5].head(5)
    bench = df[df["player_id"] >= 5].head(5)
    pred_nailed = model.predict(nailed[FEATS])
    pred_bench = model.predict(bench[FEATS])
    # Benchwarmers' expectation is crushed by P(played) ~ 0; nailed players sit near
    # their conditional mean of ~5. Loose bounds - this is plumbing, not accuracy.
    assert pred_bench.max() < 1.0
    assert pred_nailed.min() > 2.0


def test_hurdle_regression_head_learns_from_played_rows_only():
    # If the regressor saw the 0-minute zeros it would be dragged toward the mixed
    # median; trained on played rows only, its standalone output for a nailed player
    # must sit near the CONDITIONAL mean (~5), well above the mixed mean (~2.5).
    df = _frame()
    model = models.fit_model("catboost_hurdle", df[FEATS], df["total_points"],
                             minutes=df["minutes"])
    nailed = df[df["player_id"] < 5].head(20)
    reg_only = model.regressor_.predict(nailed[FEATS])
    assert reg_only.mean() > 4.0


def test_degenerate_participation_falls_back_to_constant():
    df = _frame()
    all_played = df[df["player_id"] < 5]  # every row has minutes > 0
    model = models.fit_model("catboost_hurdle", all_played[FEATS], all_played["total_points"],
                             minutes=all_played["minutes"])
    assert model.classifier_ is None and model.p_played_const_ == 1.0
    assert model.predict(all_played[FEATS].head(3)).min() > 2.0

    none_played = df[df["player_id"] >= 5]
    model = models.fit_model("catboost_hurdle", none_played[FEATS], none_played["total_points"],
                             minutes=none_played["minutes"])
    assert model.regressor_ is None
    assert (model.predict(none_played[FEATS].head(3)) == 0.0).all()


def test_hurdle_is_a_registry_member_and_fits_through_the_production_path():
    assert "catboost_hurdle" in models.MODEL_NAMES
    df = _frame()
    ensembles = fit_position_ensembles(df, FEATS, {"MID": {"catboost_hurdle": 1.0}})
    preds = ensembles["MID"].predict(df[FEATS].head(4))
    assert len(preds) == 4 and np.isfinite(preds).all()
