"""
Shared walk-forward retrain-and-step skeleton.

Four backtest loops in this codebase were near-identical copies of the same
control flow - "iterate the played gameweeks in a window, refit models on
strictly-earlier data every N gameweeks, predict the current gameweek":
`fpl.model.predict.walk_forward_predictions`, `.origin_based_predictions`,
and `fpl.model.probabilistic_buckets.evaluate_walk_forward` /
`.walk_forward_predictions_csv`. Duplicating this loop is exactly the class of
mistake that produced audit finding A1 (a live/backtest path silently drifting
from its twin), so the skeleton lives here once and each caller supplies only
what differs: how to fit (`fit_fn`) and how to predict (the loop body consuming
the yielded cache).

The generator yields per gameweek rather than taking a `predict_fn`, because the
four callers predict very differently - a single point-forecast frame, a stack of
probabilistic frames, or an inner horizon loop over frozen-form snapshots. A
generator lets each keep its own prediction body while sharing the retrain
bookkeeping that actually caused the bug.
"""
from typing import Callable, Iterator, Tuple

import pandas as pd


def walk_forward_steps(
    df: pd.DataFrame,
    start_gw: int,
    end_gw: int,
    retrain_every: int,
    fit_fn: Callable[[pd.DataFrame], object],
    gw_col: str = "GW_global",
    gw_word: str = "GW",
    label: str = "models",
) -> Iterator[Tuple[int, object, pd.DataFrame]]:
    """Yield ``(gw, cache, test_df)`` for each played gameweek in ``[start_gw, end_gw]``.

    For each gameweek ``gw`` present in ``df[gw_col]`` and within the window, in
    ascending order:

    - if at least ``retrain_every`` gameweeks have elapsed since the last fit (always
      on the first iteration), refit ``cache = fit_fn(train_df)`` where ``train_df`` is
      the rows STRICTLY BEFORE ``gw`` (the leakage guarantee every backtest here
      depends on), and log ``"Retrained {label} for {gw_word} {gw}"``;
    - yield the current ``gw``, the live ``cache``, and ``test_df`` (rows AT ``gw``).

    A gameweek whose training frame is empty (``gw`` at or before the first available
    gameweek) is skipped without yielding - there is nothing to predict from. The
    caller's ``fit_fn`` returns whatever the prediction body needs (a per-position
    model dict, a bucket-model cache, a snapshot-plus-models tuple); this module never
    inspects it.
    """
    cache = None
    last_trained_gw = None
    available_gws = sorted(set(df[gw_col].unique()) & set(range(start_gw, end_gw + 1)))
    for gw in available_gws:
        if last_trained_gw is None or gw - last_trained_gw >= retrain_every:
            train_df = df[df[gw_col] < gw]
            if train_df.empty:
                continue
            cache = fit_fn(train_df)
            last_trained_gw = gw
            print(f"Retrained {label} for {gw_word} {gw}")
        yield gw, cache, df[df[gw_col] == gw]
