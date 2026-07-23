"""
Tests for the shared walk-forward skeleton (fpl.model.walk_forward, TODO 4.1).

The four backtest loops that used to copy this control flow all depend on two
invariants the skeleton must guarantee: models are refit exactly every
`retrain_every` played gameweeks, and each refit trains on rows STRICTLY BEFORE
the current gameweek (the leakage guarantee). These pin both with a tiny frame so
the expected retrain gameweeks and training-window maxima are hand-checkable.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model.walk_forward import walk_forward_steps


def _frame(gws):
    # one row per gameweek is enough to exercise the loop's bookkeeping
    return pd.DataFrame({"GW_global": gws, "val": gws})


def test_retrains_on_cadence_and_trains_on_strictly_earlier_rows():
    df = _frame(list(range(1, 11)))
    fit_calls = []  # (gw_at_fit, max_gw_in_train_df)

    def fit_fn(train_df):
        return train_df["GW_global"].max()

    steps = []
    for gw, cache, test_df in walk_forward_steps(df, start_gw=5, end_gw=10,
                                                 retrain_every=2, fit_fn=fit_fn):
        steps.append((gw, cache))
        # test_df holds exactly the current gameweek's rows
        assert list(test_df["GW_global"].unique()) == [gw]

    # Refit at GW5 (first), then every 2 GWs: 5, 7, 9. Cache carries forward between.
    assert [gw for gw, _ in steps] == [5, 6, 7, 8, 9, 10]
    assert [cache for _, cache in steps] == [4, 4, 6, 6, 8, 8]
    # Each cache is max(train GW) = current GW - 1, i.e. strictly earlier (no leakage).


def test_skips_gameweeks_absent_from_the_frame():
    df = _frame([1, 2, 3, 4, 5, 6, 9, 10])  # GW7, GW8 missing (e.g. a blank)
    seen = [gw for gw, _, _ in walk_forward_steps(df, 5, 10, retrain_every=1,
                                                  fit_fn=lambda t: None)]
    assert seen == [5, 6, 9, 10]


def test_empty_training_frame_yields_nothing():
    df = _frame([1, 2, 3])
    # start_gw=1: the only candidate refit is at GW1, whose train_df (GW<1) is empty,
    # so the loop skips it and never yields.
    steps = list(walk_forward_steps(df, start_gw=1, end_gw=1, retrain_every=1,
                                    fit_fn=lambda t: t))
    assert steps == []


def test_window_bounds_are_respected():
    df = _frame(list(range(1, 21)))
    seen = [gw for gw, _, _ in walk_forward_steps(df, start_gw=8, end_gw=12,
                                                  retrain_every=3, fit_fn=lambda t: None)]
    assert seen == [8, 9, 10, 11, 12]
