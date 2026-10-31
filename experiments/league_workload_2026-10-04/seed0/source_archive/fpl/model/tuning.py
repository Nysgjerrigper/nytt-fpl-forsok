"""
Hyperparameter tuning for explicit model-tournament experts.

Registry constructors supply hand-picked defaults; tracked tuned JSON artifacts preserve
completed position-specific studies. Tuning searches each expert's declared space under
causal expanding-window validation. Accuracy screening is diagnostic, and promotion
still requires realized MILP comparisons with confidence intervals.

Two design choices worth stating up front:

- Optuna is a heavy dependency only tuning needs (listed in requirements.txt, but an
  environment without it should still run the rest of the pipeline). Every function that
  needs it imports it lazily, so this module imports fine without optuna installed - importing
  it must never break the rest of the pipeline just because tuning wasn't set up. The tests
  skip themselves when it's absent.

- The cross-validation is EXPANDING-WINDOW over GW_global, never a random KFold. FPL data is a
  time series (fixture rows grouped by gameweek); shuffling rows would let a fold train on gameweek
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
from numbers import Real

import numpy as np

from fpl import config
from fpl.data.provenance import load_frozen_research_dataset
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
                  time_budget_seconds=None, storage_path=None, study_name=None, stage=None, checkpoint_report=None):
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

    Persistent resumes are opt-in via storage_path. They bind data/code/folds/runtime
    and use serial_trial_seed_v1, a separate sampler protocol from registered legacy
    studies. n_trials is the total terminal-trial target; timeout resets per invocation.
    RUNNING/WAITING trials and mismatched identities are preserved and refused.

    Returned dict includes fixed objective/seed/verbosity settings as well as searched
    knobs. It can be passed to ``models.build_registered_model`` to reconstruct the exact
    candidate.
    """
    import optuna

    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    if time_budget_seconds is not None and (not np.isfinite(time_budget_seconds)
                                           or time_budget_seconds <= 0):
        raise ValueError("time_budget_seconds must be finite and positive")
    if study_name is not None and storage_path is None:
        raise ValueError("study_name requires storage_path")
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
    if storage_path is None:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, timeout=time_budget_seconds,
                       show_progress_bar=False)
    else:
        from fpl.model.tuning_checkpoint import checkpoint_identity, run_checkpoint
        identity = checkpoint_identity(pos_df, list(feature_cols), position=position,
            model_name=model_name, seed=seed, stage=stage, train_max_gw=train_max_gw,
            n_splits=n_splits, folds=folds)
        study = run_checkpoint(objective, storage_path=storage_path,
            study_name=study_name or "identity_bound_tuning", identity=identity,
            seed=seed, n_trials=n_trials, timeout=time_budget_seconds)
        if checkpoint_report is not None:
            import hashlib
            checkpoint_report.update({
                "sampler_protocol": identity["protocol"],
                "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "study_name": study.study_name, "trial_target": n_trials,
                "terminal_trials": len(study.trials), "best_trial_number": study.best_trial.number,
                "trial_states": {state.name: sum(trial.state == state for trial in study.trials)
                                 for state in optuna.trial.TrialState},
                "invocation_timeout_seconds": time_budget_seconds,
            })

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
                     n_splits=None, time_budget_seconds=None, stage=None, checkpoint_metadata=None, output_dir=None):
    """Persist tuned params to fpl/models/tuned_params_<position>_<model>.json, returning
    the path. Tracked JSONs in config.MODELS_DIR preserve completed studies; do not
    overwrite or regenerate existing research artifacts merely to tidy the repository.

    `train_max_gw` (the data cap the search actually ran under) is recorded in a "_meta"
    key so the file documents its own provenance - a params file with no recorded cap
    cannot prove the search didn't validate on the backtest window. Underscore-prefixed
    keys are stripped by models._tuned_params before the constructor splat."""
    directory = Path(config.MODELS_DIR if output_dir is None else output_dir)
    if checkpoint_metadata is not None and directory.resolve() == config.MODELS_DIR.resolve():
        raise ValueError("Checkpoint artifacts require a separate output directory")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tuned_params_{position}_{model_name}.json"
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
    if checkpoint_metadata is not None:
        payload["_meta"]["checkpoint"] = dict(checkpoint_metadata)
    payload["_meta"]["artifact_hash"] = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in payload.items() if k != "_meta"} | {"_meta": payload["_meta"]},
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    content = json.dumps(payload, indent=2, sort_keys=True)
    if checkpoint_metadata is None:
        path.write_text(content)
    elif path.exists():
        if path.read_text() != content:
            raise ValueError("Checkpoint export already exists with different content; preserve it")
    else:
        with path.open("x") as destination:
            destination.write(content)
    return path


def load_validated_params(path, *, position, model_name, seed, stage, max_train_gw,
                          sampler_protocol=None):
    """Load a tuned artifact only when causal and sampler provenance match.

    None admits legacy studies only; new checkpoint sampler protocols require explicit
    caller registration. Checkpoint exports never enter legacy tournament validation.
    """
    payload = json.loads(Path(path).read_text())
    meta = payload.get("_meta", {})
    checkpoint = meta.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint sampler protocol metadata")
    actual_protocol = None if checkpoint is None else checkpoint.get("sampler_protocol")
    if (actual_protocol != sampler_protocol
            or (checkpoint is not None and not actual_protocol)):
        raise ValueError("Tuned artifact sampler protocol is not admitted by this study")
    required = {"position": position, "model": model_name, "seed": seed, "stage": stage}
    if any(meta.get(key) != value for key, value in required.items()):
        raise ValueError("tuned parameter provenance does not match requested model/position/seed/stage")
    cutoff = meta.get("train_max_gw")
    if (isinstance(cutoff, bool) or not isinstance(cutoff, Real)
            or not np.isfinite(cutoff) or cutoff < 0 or int(cutoff) != cutoff
            or cutoff > max_train_gw):
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
    from fpl.data.feature_cache import load_frozen_feature_frame

    raw = load_frozen_research_dataset()
    return load_frozen_feature_frame(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune a GBM's hyperparameters for one position via Optuna.")
    parser.add_argument("--position", required=True, choices=config.ONFIELD_POSITIONS)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=3600,
                        help="Wall-clock ceiling for this model-position study.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Opt-in identity-bound SQLite resume with serial_trial_seed_v1; "
                             "exports separately and is not legacy registered evidence.")
    parser.add_argument("--study-name", default=None)
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
    checkpoint_report = {}
    best = tune_position(df, feature_cols, args.position, args.model, args.n_trials, args.n_splits,
                         train_max_gw=args.train_max_gw, seed=args.seed,
                         time_budget_seconds=args.time_budget_seconds,
                         storage_path=args.checkpoint, study_name=args.study_name,
                         stage=args.stage, checkpoint_report=checkpoint_report)
    output_dir = None
    if args.checkpoint is not None:
        import hashlib
        namespace = hashlib.sha256(
            f"{args.checkpoint.resolve()}:{args.study_name or 'identity_bound_tuning'}".encode()
        ).hexdigest()[:24]
        output_dir = config.MODELS_DIR / ".optuna_exports" / namespace / f"trials-{args.n_trials}"
    path = save_best_params(args.position, args.model, best, train_max_gw=args.train_max_gw,
                            seed=args.seed, n_splits=args.n_splits,
                            time_budget_seconds=args.time_budget_seconds, stage=args.stage,
                            checkpoint_metadata=checkpoint_report or None, output_dir=output_dir)
    print(f"Best params: {json.dumps(best, indent=2, sort_keys=True)}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
