"""
Hyperparameter tuning for the gradient-boosted members of the model registry.

Nothing in fpl.model.models is actually tuned - LGB_PARAMS/XGB_PARAMS/CATBOOST_PARAMS
are hand-picked defaults. That's a problem for the whole comparison table this project
turns on: when the registry ranks lightgbm vs. xgboost vs. catboost (and against the OLS
index), part of what's being measured is which algorithm's DEFAULTS happened to suit FPL's
data, not which algorithm is best once each is given a fair shot. This module gives each
GBM a fair shot by searching its own hyperparameters, so a later ranking reflects the models
rather than default-luck.

Two design choices worth stating up front:

- Optuna is a heavy, optional dependency (it is NOT in requirements.txt). Every function that
  needs it imports it lazily, so this module imports fine without optuna installed - importing
  it must never break the rest of the pipeline just because tuning wasn't set up. The tests
  skip themselves when it's absent.

- The cross-validation is EXPANDING-WINDOW over GW_global, never a random KFold. FPL data is a
  time series (one row per player-gameweek); shuffling rows would let a fold train on gameweek
  t+5 and validate on gameweek t, which leaks the future into the past and would reward
  hyperparameters that overfit that leakage rather than ones that actually forecast. Each fold
  therefore trains only on gameweeks strictly earlier than the block it validates on - the same
  discipline the rest of the pipeline enforces on features.

The objective is mean validation MASE (fpl.model.metrics), not MAE, for the reason spelled out
in that module: FPL points are intermittent, so a raw MAE isn't comparable across positions or
folds, whereas MASE (< 1 beats the naive last-gameweek forecast) is. The MASE scale is refit
per fold on that fold's TRAINING rows only, so the denominator can't leak information from the
validation block it is used to score.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config
from fpl.model.metrics import mase, naive_lag1_scale

# The GBMs this module knows how to tune. Kept deliberately narrow: the linear/distance
# members of the registry either have almost nothing to tune or are cheap enough that
# defaults are fine - the payoff from an Optuna search is concentrated in the tree
# ensembles, whose many interacting knobs (depth, leaves, regularisation, sampling) are
# exactly where hand-set defaults leave the most on the table.
SUPPORTED_MODELS = ("lightgbm", "xgboost", "catboost")

TARGET_COL = "total_points"


def _suggest_params(trial, model_name):
    """Map an Optuna trial to a hyperparameter dict for `model_name`.

    Search ranges are chosen around each library's sensible operating region for
    tabular data of this size (tens of thousands of rows per position), covering the
    knobs that actually trade bias against variance here - tree size/depth, learning
    rate paired with the number of trees, L1/L2 regularisation, and row/column
    subsampling - rather than every exposed parameter, which would only make the
    search space too large to explore in a practical number of trials.
    """
    if model_name == "lightgbm":
        return dict(
            objective="regression",
            metric="mae",
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 100),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            verbosity=-1,
        )
    if model_name == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            random_state=0,
            n_jobs=-1,
        )
    if model_name == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 100, 800),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-1, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            loss_function="MAE",
            random_seed=0,
            verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}")


def _build_model(model_name, params):
    """Instantiate a fresh regressor of `model_name` with `params`.

    Imported lazily and only for the requested library so tuning one model never
    forces every GBM to be installed - and so this module still imports when none is.
    """
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params)
    if model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    if model_name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    raise ValueError(f"unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}")


def _expanding_window_folds(gws, n_splits):
    """Yield (train_gws, val_gws) as an expanding-window split of the sorted unique
    gameweeks into n_splits successive validation blocks.

    The gameweeks are cut into n_splits+1 roughly equal contiguous chunks: chunk 0 seeds
    the first training window, then each later chunk is validated in turn while everything
    before it accumulates into the training window. This keeps every fold strictly causal
    (train always precedes validation in time) and, unlike a fixed-size sliding window,
    lets later folds benefit from all earlier history - matching how the production models
    are actually retrained on an ever-growing dataset week to week.
    """
    gws = np.asarray(sorted(gws))
    # n_splits validation blocks need n_splits+1 chunks (the first is training-only seed).
    chunks = np.array_split(gws, n_splits + 1)
    train_gws = chunks[0]
    for i in range(1, len(chunks)):
        val_gws = chunks[i]
        if len(train_gws) and len(val_gws):
            yield set(train_gws.tolist()), set(val_gws.tolist())
        train_gws = np.concatenate([train_gws, val_gws])


def tune_position(df, feature_cols, position, model_name, n_trials=50, n_splits=4):
    """Search hyperparameters for one GBM on one position, returning the best params dict.

    Runs an Optuna study whose objective refits `model_name` on each expanding-window fold
    and returns the mean validation MASE across folds (the study minimizes it). Only rows of
    the given `position` are used, since the pipeline trains a separate model per position and
    a mix of positions would blur the very scale differences that motivate that separation.

    Returned dict is ready to splat straight into the corresponding LGBMRegressor/XGBRegressor/
    CatBoostRegressor constructor - it includes the fixed objective/seed/verbosity settings, not
    only the searched knobs - so a caller can persist it and reconstruct the exact tuned model.
    """
    import optuna

    pos_df = df[df["position"] == position].sort_values("GW_global")
    gws = pos_df["GW_global"].unique()
    folds = list(_expanding_window_folds(gws, n_splits))
    if not folds:
        raise ValueError(
            f"not enough distinct gameweeks for {position} to build {n_splits} CV folds "
            f"(found {len(gws)})"
        )

    def objective(trial):
        params = _suggest_params(trial, model_name)
        fold_scores = []
        for train_gws, val_gws in folds:
            train = pos_df[pos_df["GW_global"].isin(train_gws)]
            val = pos_df[pos_df["GW_global"].isin(val_gws)]
            if train.empty or val.empty:
                continue
            model = _build_model(model_name, params)
            model.fit(train[feature_cols], train[TARGET_COL])
            preds = model.predict(val[feature_cols])
            # Scale from the fold's TRAINING rows only, so the MASE denominator never
            # sees the validation block it is used to judge (same discipline as train.py).
            scale = naive_lag1_scale(train)
            fold_scores.append(mase(val[TARGET_COL].to_numpy(), preds, scale))
        if not fold_scores:
            # No usable fold for these params - tell Optuna to discard the trial rather
            # than return a misleadingly-good 0.
            raise optuna.TrialPruned()
        return float(np.nanmean(fold_scores))

    # Fixed sampler seed so a rerun with the same data reproduces the same search path -
    # tuning should be a repeatable experiment, not a different answer every invocation.
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Re-derive the full param dict (searched values + the fixed settings) from the winning
    # trial, rather than returning study.best_params, which holds only the suggested knobs.
    best = _suggest_params(_FrozenTrial(study.best_trial.params), model_name)
    return best


class _FrozenTrial:
    """Replays a completed trial's chosen values through _suggest_params so the fixed
    (non-searched) settings get re-attached to the winning hyperparameters. suggest_* just
    returns the already-decided value for each name and ignores the range arguments."""

    def __init__(self, params):
        self._params = params

    def suggest_int(self, name, *args, **kwargs):
        return self._params[name]

    def suggest_float(self, name, *args, **kwargs):
        return self._params[name]

    def suggest_categorical(self, name, *args, **kwargs):
        return self._params[name]


def save_best_params(position, model_name, params):
    """Persist tuned params to fpl/models/tuned_params_<position>_<model>.json, returning
    the path. Lives alongside the saved ensembles (config.MODELS_DIR, gitignored) since it is
    a regenerable training artifact, not source - re-run the tuner to recreate it."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"tuned_params_{position}_{model_name}.json"
    path.write_text(json.dumps(params, indent=2, sort_keys=True))
    return path


def _load_features():
    """Local loader so this module doesn't import fpl.model.train (which pulls in the whole
    baselines/statsmodels stack just to reach build_feature_frame)."""
    import pandas as pd
    from fpl import features

    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    return features.build_feature_frame(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune a GBM's hyperparameters for one position via Optuna.")
    parser.add_argument("--position", required=True, choices=config.ONFIELD_POSITIONS)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=4)
    args = parser.parse_args(argv)

    from fpl import features

    df = _load_features()
    feature_cols = features.feature_columns(df)
    print(f"Tuning {args.model} for {args.position} over {args.n_trials} trials "
          f"({args.n_splits}-fold expanding-window CV)...")
    best = tune_position(df, feature_cols, args.position, args.model, args.n_trials, args.n_splits)
    path = save_best_params(args.position, args.model, best)
    print(f"Best params: {json.dumps(best, indent=2, sort_keys=True)}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
