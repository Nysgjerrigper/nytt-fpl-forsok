"""
Hyperparameter tuning for explicit model-tournament experts.

Nothing in fpl.model.models is actually tuned - LGB_PARAMS/XGB_PARAMS/CATBOOST_PARAMS
are hand-picked defaults. That's a problem for the whole comparison table this project
turns on: when the registry ranks lightgbm vs. xgboost vs. catboost (and against the OLS
index), part of what's being measured is which algorithm's DEFAULTS happened to suit FPL's
data, not which algorithm is best once each is given a fair shot. This module gives each
tunable expert a fair shot by searching its declared hyperparameters, so a later ranking
reflects the models rather than default-luck.

Two design choices worth stating up front:

- Optuna is a heavy dependency only tuning needs (listed in requirements.txt, but an
  environment without it should still run the rest of the pipeline). Every function that
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
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config
from fpl.model.metrics import mase, naive_lag1_scale
from fpl.model import models

# All tunable experts are declared by the registry. Production training continues to use
# models.MODEL_NAMES, which deliberately excludes research-only candidates.
SUPPORTED_MODELS = models.REGISTERED_MODEL_NAMES

TARGET_COL = "total_points"


def _suggest_params(trial, model_name, seed=0):
    """Map an Optuna trial to a hyperparameter dict for `model_name`.

    Search ranges are chosen around each library's sensible operating region for
    tabular data of this size (tens of thousands of rows per position), covering the
    knobs that actually trade bias against variance here - tree size/depth, learning
    rate paired with the number of trees, L1/L2 regularisation, and row/column
    subsampling - rather than every exposed parameter, which would only make the
    search space too large to explore in a practical number of trials.
    """
    try:
        spec = models.EXPERT_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}") from exc
    if spec.search_space is None:
        raise ValueError(f"expert {model_name!r} has no tuning search space")
    return spec.search_space(trial, seed)


def _build_model(model_name, params, *, seed=0, position=None):
    """Instantiate a fresh regressor of `model_name` with `params`.

    Imported lazily and only for the requested library so tuning one model never
    forces every GBM to be installed - and so this module still imports when none is.
    """
    return models.build_registered_model(model_name, params=params, seed=seed, position=position)


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


def tune_position(df, feature_cols, position, model_name, n_trials=50, n_splits=4,
                  train_max_gw=config.TUNING_TRAIN_MAX_GW, seed=0,
                  time_budget_seconds=None):
    """Search hyperparameters for one registered expert and position.

    Runs an Optuna study whose objective refits `model_name` on each expanding-window fold
    and returns the mean validation MASE across folds (the study minimizes it). Only rows of
    the given `position` are used, since the pipeline trains a separate model per position and
    a mix of positions would blur the very scale differences that motivate that separation.

    `train_max_gw` caps BOTH training and validation folds at that global gameweek
    (default config.TUNING_TRAIN_MAX_GW = the last GW before the standing MILP backtest
    window). Without the cap, hyperparameters get validated on the very gameweeks the
    headline realized-points number is later computed on, which quietly turns that number
    into in-sample performance (audit finding A2). Pass None only for experiments that will
    never be judged on the standing backtest window.

    Returned dict includes fixed objective/seed/verbosity settings as well as searched
    knobs. It can be passed to ``models.build_registered_model`` to reconstruct the exact
    candidate.
    """
    import optuna

    pos_df = df[df["position"] == position].sort_values("GW_global")
    if train_max_gw is not None:
        pos_df = pos_df[pos_df["GW_global"] <= train_max_gw]
    gws = pos_df["GW_global"].unique()
    folds = list(_expanding_window_folds(gws, n_splits))
    if not folds:
        raise ValueError(
            f"not enough distinct gameweeks for {position} to build {n_splits} CV folds "
            f"(found {len(gws)})"
        )

    def objective(trial):
        params = _suggest_params(trial, model_name, seed=seed)
        fold_scores = []
        for train_gws, val_gws in folds:
            train = pos_df[pos_df["GW_global"].isin(train_gws)]
            val = pos_df[pos_df["GW_global"].isin(val_gws)]
            if train.empty or val.empty:
                continue
            model = _build_model(model_name, params, seed=seed, position=position)
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
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=time_budget_seconds,
                   show_progress_bar=False)

    # Re-derive the full param dict (searched values + the fixed settings) from the winning
    # trial, rather than returning study.best_params, which holds only the suggested knobs.
    best = _suggest_params(_FrozenTrial(study.best_trial.params), model_name, seed=seed)
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


def save_best_params(position, model_name, params, train_max_gw=None, *, seed=0,
                     n_splits=None, time_budget_seconds=None, stage=None):
    """Persist tuned params to fpl/models/tuned_params_<position>_<model>.json, returning
    the path. Lives in config.MODELS_DIR (gitignored) since it is a regenerable training
    artifact, not source - re-run the tuner to recreate it.

    `train_max_gw` (the data cap the search actually ran under) is recorded in a "_meta"
    key so the file documents its own provenance - a params file with no recorded cap
    cannot prove the search didn't validate on the backtest window. Underscore-prefixed
    keys are stripped by models._tuned_params before the constructor splat."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"tuned_params_{position}_{model_name}.json"
    spec = models.EXPERT_SPECS[model_name]
    packages = {}
    for package in ("numpy", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "pytabkit"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    payload = {**params, "_meta": {
        "train_max_gw": train_max_gw,
        "position": position,
        "model": model_name,
        "stage": stage,
        "n_splits": n_splits,
        "time_budget_seconds": time_budget_seconds,
        "seed": seed,
        "preprocessing": spec.preprocessing,
        "auxiliary_labels": list(spec.auxiliary_labels),
        "provenance": spec.provenance,
        "python_version": sys.version,
        "packages": packages,
    }}
    payload["_meta"]["artifact_hash"] = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in payload.items() if k != "_meta"} | {"_meta": payload["_meta"]},
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_validated_params(path, *, position, model_name, seed, stage, max_train_gw):
    """Load a tuned artifact only when its complete causal provenance matches."""
    payload = json.loads(Path(path).read_text())
    meta = payload.get("_meta", {})
    required = {"position": position, "model": model_name, "seed": seed, "stage": stage}
    if any(meta.get(key) != value for key, value in required.items()):
        raise ValueError("tuned parameter provenance does not match requested model/position/seed/stage")
    if meta.get("train_max_gw") is None or int(meta["train_max_gw"]) > int(max_train_gw):
        raise ValueError("tuned parameter artifact exceeds the causal training cutoff")
    digest = meta.get("artifact_hash")
    if not digest:
        raise ValueError("tuned parameter artifact has no provenance hash")
    hashed_meta = {key: value for key, value in meta.items() if key != "artifact_hash"}
    actual = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in payload.items() if k != "_meta"} | {"_meta": hashed_meta},
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    if actual != digest:
        raise ValueError("tuned parameter artifact hash mismatch")
    return {key: value for key, value in payload.items() if not key.startswith("_")}


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=3600,
                        help="Wall-clock ceiling for this model-position study.")
    parser.add_argument("--stage", choices=("discovery", "selection", "finalist"), default="discovery")
    parser.add_argument("--train-max-gw", type=int, default=config.TUNING_TRAIN_MAX_GW,
                        help="Cap all CV folds at this global gameweek so the search never "
                             "validates on the standing backtest window (default: "
                             f"{config.TUNING_TRAIN_MAX_GW}, the GW before that window starts).")
    args = parser.parse_args(argv)

    from fpl import features

    df = _load_features()
    feature_cols = features.feature_columns(df)
    print(f"Tuning {args.model} for {args.position} over {args.n_trials} trials "
          f"({args.n_splits}-fold expanding-window CV, folds capped at GW{args.train_max_gw})...")
    best = tune_position(df, feature_cols, args.position, args.model, args.n_trials, args.n_splits,
                         train_max_gw=args.train_max_gw, seed=args.seed,
                         time_budget_seconds=args.time_budget_seconds)
    path = save_best_params(args.position, args.model, best, train_max_gw=args.train_max_gw,
                            seed=args.seed, n_splits=args.n_splits,
                            time_budget_seconds=args.time_budget_seconds, stage=args.stage)
    print(f"Best params: {json.dumps(best, indent=2, sort_keys=True)}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
