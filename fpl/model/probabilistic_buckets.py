"""
Probabilistic bucket forecasting for FPL points.

The production forecaster predicts one number per player-gameweek. This module
tests the alternative framing: predict a small probability distribution over
point outcomes, then derive expected points, blank risk, and haul upside from
that same distribution.

It is intentionally a parallel bake-off module. It reuses the normal feature
pipeline but does not feed the saved production ensembles or the MILP optimizer.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config, features
from fpl.model import models as point_models
from fpl.model.metrics import bias, mae, rmse, spearman_by_group, top1_capture, total_calibration
from fpl.model.train import POSITIONS


@dataclass(frozen=True)
class BucketScheme:
    """Ordered point buckets, inclusive integer bounds."""

    name: str
    labels: tuple[str, ...]
    ranges: tuple[tuple[float | None, float | None], ...]
    default_values: tuple[float, ...]

    @property
    def n_buckets(self):
        return len(self.labels)

    @property
    def loss_indices(self):
        return tuple(i for i, (_, hi) in enumerate(self.ranges) if hi is not None and hi <= 2)

    @property
    def haul_indices(self):
        return tuple(i for i, (lo, _) in enumerate(self.ranges) if lo is not None and lo >= 10)


BUCKET_SCHEMES = {
    "coarse5": BucketScheme(
        name="coarse5",
        labels=("zero_or_negative", "blank_1_2", "modest_3_5", "return_6_9", "haul_10_plus"),
        ranges=((None, 0), (1, 2), (3, 5), (6, 9), (10, None)),
        default_values=(0.0, 1.5, 4.0, 7.5, 12.0),
    ),
    "fine8": BucketScheme(
        name="fine8",
        labels=(
            "zero_or_negative",
            "one_point",
            "two_points",
            "small_return_3_4",
            "good_5_6",
            "strong_7_9",
            "small_haul_10_12",
            "mega_haul_13_plus",
        ),
        ranges=((None, 0), (1, 1), (2, 2), (3, 4), (5, 6), (7, 9), (10, 12), (13, None)),
        default_values=(0.0, 1.0, 2.0, 3.5, 5.5, 8.0, 11.0, 15.0),
    ),
    "fine10": BucketScheme(
        name="fine10",
        labels=(
            "zero_or_negative",
            "one_point",
            "two_points",
            "three_points",
            "four_five",
            "six_seven",
            "eight_nine",
            "ten_eleven",
            "twelve_fourteen",
            "fifteen_plus",
        ),
        ranges=((None, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 14), (15, None)),
        default_values=(0.0, 1.0, 2.0, 3.0, 4.5, 6.5, 8.5, 10.5, 13.0, 17.0),
    ),
}
DEFAULT_BUCKET_SCHEME = BUCKET_SCHEMES["coarse5"]

# Backwards-compatible aliases for tests and simple imports.
BUCKET_LABELS = list(DEFAULT_BUCKET_SCHEME.labels)
N_BUCKETS = DEFAULT_BUCKET_SCHEME.n_buckets
DEFAULT_BUCKET_VALUES = np.asarray(DEFAULT_BUCKET_SCHEME.default_values, dtype=float)
MODEL_NAMES = [
    "catboost_bucket",
    "lightgbm_bucket",
    "xgboost_bucket",
    "logistic_bucket",
    "catboost_hurdle_bucket",
    "lightgbm_hurdle_bucket",
]


def get_bucket_scheme(name):
    try:
        return BUCKET_SCHEMES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown bucket scheme {name!r}; choose one of {sorted(BUCKET_SCHEMES)}") from exc


def points_to_buckets(points, scheme=DEFAULT_BUCKET_SCHEME):
    """Map FPL points to ordered outcome buckets.

    Bucket 0 includes negative scores. For decision purposes a -1 and a 0 are
    both catastrophic outcomes; modelling them separately would waste scarce
    examples.
    """
    pts = np.asarray(points, dtype=float)
    buckets = np.full(len(pts), scheme.n_buckets - 1, dtype=int)
    for i, (lo, hi) in enumerate(scheme.ranges):
        mask = np.ones(len(pts), dtype=bool)
        if lo is not None:
            mask &= pts >= lo
        if hi is not None:
            mask &= pts <= hi
        buckets[mask] = i
    return buckets


def bucket_values_from_training(points, buckets, scheme=DEFAULT_BUCKET_SCHEME):
    """Representative point value per bucket, estimated on training data only."""
    pts = np.asarray(points, dtype=float)
    bucket_arr = np.asarray(buckets, dtype=int)
    values = np.asarray(scheme.default_values, dtype=float).copy()
    for bucket in range(scheme.n_buckets):
        mask = bucket_arr == bucket
        if np.any(mask):
            values[bucket] = float(np.mean(pts[mask]))
    return values


def expected_points_from_proba(proba, bucket_values):
    return np.asarray(proba, dtype=float) @ np.asarray(bucket_values, dtype=float)


def multiclass_brier(y_bucket, proba, n_buckets=None):
    y = np.asarray(y_bucket, dtype=int)
    p = np.asarray(proba, dtype=float)
    n = n_buckets or p.shape[1]
    one_hot = np.zeros((len(y), n), dtype=float)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def normalize_proba(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def aligned_multiclass_proba(model, X, scheme=DEFAULT_BUCKET_SCHEME):
    """Return predict_proba columns in the fixed 0..n bucket order."""
    raw = model.predict_proba(X)
    out = np.zeros((len(X), scheme.n_buckets), dtype=float)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, i]
    return normalize_proba(out)


def aligned_binary_positive_proba(model, X):
    """Return P(class=1), robust to degenerate one-class fits."""
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return np.asarray(raw[:, int(np.where(classes == 1)[0][0])], dtype=float)


def _lgb_params(quick, objective, metric):
    params = {k: v for k, v in point_models.LGB_PARAMS.items() if k not in ("objective", "metric")}
    if quick:
        params["n_estimators"] = 80
    return dict(objective=objective, metric=metric, **params)


def _xgb_params(quick, objective, eval_metric):
    params = point_models.XGB_PARAMS.copy()
    if quick:
        params["n_estimators"] = 80
    return dict(objective=objective, eval_metric=eval_metric, tree_method="hist", **params)


def _cat_params(quick, loss_function, position=None, use_tuned=False):
    """CatBoost params for a classifier, optionally adapted from the tuned
    per-position REGRESSION params saved by fpl.model.tuning.

    The tuned depth/learning-rate/iterations/l2/subsample transfer; the loss
    function obviously cannot (MAE is a regression loss), so it is swapped for
    the classification objective. CatBoost only allows `subsample` with a
    sampling bootstrap, so Bernoulli is pinned when subsample is present.
    """
    tuned = point_models._tuned_params("catboost", position) if use_tuned else None
    params = (tuned or point_models.CATBOOST_PARAMS).copy()
    params["loss_function"] = loss_function
    if "subsample" in params and "bootstrap_type" not in params:
        params["bootstrap_type"] = "Bernoulli"
    if quick:
        params["iterations"] = 80
    return params


def make_bucket_estimator(name, quick=False, position=None, use_tuned=False):
    if name == "catboost_bucket":
        return CatBoostClassifier(**_cat_params(quick, "MultiClass", position, use_tuned))
    if name == "lightgbm_bucket":
        return lgb.LGBMClassifier(**_lgb_params(quick, "multiclass", "multi_logloss"))
    if name == "xgboost_bucket":
        return xgb.XGBClassifier(**_xgb_params(quick, "multi:softprob", "mlogloss"))
    if name == "logistic_bucket":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ])
    raise KeyError(name)


def make_binary_estimator(family, quick=False, position=None, use_tuned=False):
    if family == "catboost":
        return CatBoostClassifier(**_cat_params(quick, "Logloss", position, use_tuned))
    if family == "lightgbm":
        return lgb.LGBMClassifier(**_lgb_params(quick, "binary", "binary_logloss"))
    raise KeyError(family)


@dataclass
class BucketModel:
    name: str
    scheme: BucketScheme = DEFAULT_BUCKET_SCHEME
    quick: bool = False
    position: str | None = None
    use_tuned: bool = False

    def fit(self, train_df, feature_cols):
        y_bucket = points_to_buckets(train_df[features.TARGET_COL], self.scheme)
        self.bucket_values_ = bucket_values_from_training(
            train_df[features.TARGET_COL], y_bucket, self.scheme
        )
        self.model_ = make_bucket_estimator(
            self.name, quick=self.quick, position=self.position, use_tuned=self.use_tuned
        )
        self.model_.fit(train_df[feature_cols], y_bucket)
        return self

    def predict_distribution(self, X):
        model = self.model_
        if isinstance(model, Pipeline):
            model = model.named_steps["model"]
            raw = self.model_.predict_proba(X)
            out = np.zeros((len(X), self.scheme.n_buckets), dtype=float)
            for i, cls in enumerate(model.classes_):
                out[:, int(cls)] = raw[:, i]
            return normalize_proba(out)
        return aligned_multiclass_proba(model, X, self.scheme)


@dataclass
class HurdleBucketModel:
    """P(plays) times a conditional bucket distribution for rows where he plays."""

    name: str
    family: str
    scheme: BucketScheme = DEFAULT_BUCKET_SCHEME
    quick: bool = False
    position: str | None = None
    use_tuned: bool = False

    def fit(self, train_df, feature_cols):
        y_bucket = points_to_buckets(train_df[features.TARGET_COL], self.scheme)
        self.bucket_values_ = bucket_values_from_training(train_df[features.TARGET_COL], y_bucket, self.scheme)
        self.play_model_ = make_binary_estimator(
            self.family, quick=self.quick, position=self.position, use_tuned=self.use_tuned
        )
        played = (train_df["minutes"] > 0).astype(int)
        self.play_model_.fit(train_df[feature_cols], played)

        played_df = train_df[played.astype(bool)]
        if played_df.empty:
            played_df = train_df
        self.bucket_model_ = make_bucket_estimator(
            f"{self.family}_bucket", quick=self.quick, position=self.position, use_tuned=self.use_tuned
        )
        self.bucket_model_.fit(played_df[feature_cols], points_to_buckets(played_df[features.TARGET_COL], self.scheme))
        return self

    def predict_distribution(self, X):
        p_play = aligned_binary_positive_proba(self.play_model_, X)
        cond = aligned_multiclass_proba(self.bucket_model_, X, self.scheme)
        proba = cond * p_play[:, None]
        proba[:, 0] += 1.0 - p_play
        return normalize_proba(proba)


def candidate_models(quick=False, scheme=DEFAULT_BUCKET_SCHEME, model_names=None, position=None, use_tuned=False):
    specs = {
        "catboost_bucket": lambda: BucketModel(
            "catboost_bucket", scheme=scheme, quick=quick, position=position, use_tuned=use_tuned
        ),
        "lightgbm_bucket": lambda: BucketModel("lightgbm_bucket", scheme=scheme, quick=quick),
        "xgboost_bucket": lambda: BucketModel("xgboost_bucket", scheme=scheme, quick=quick),
        "logistic_bucket": lambda: BucketModel("logistic_bucket", scheme=scheme, quick=quick),
        "catboost_hurdle_bucket": lambda: HurdleBucketModel(
            "catboost_hurdle_bucket", family="catboost", scheme=scheme, quick=quick,
            position=position, use_tuned=use_tuned,
        ),
        "lightgbm_hurdle_bucket": lambda: HurdleBucketModel(
            "lightgbm_hurdle_bucket", family="lightgbm", scheme=scheme, quick=quick
        ),
    }
    selected = model_names or MODEL_NAMES
    return [specs[name]() for name in selected]


def safe_auc(y_event, p_event):
    y = np.asarray(y_event, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p_event))


def evaluate_prediction_frame(pred_df):
    """Score one (model, subset) prediction frame.

    A pure point-forecast frame (the regression baseline) has NaN bucket
    probabilities; its distribution metrics are reported as NaN while the
    expected-points metrics stay comparable across all models.
    """
    scheme = get_bucket_scheme(pred_df["scheme"].iloc[0])
    y = pred_df["actual_points"].to_numpy()
    y_bucket = pred_df["bucket"].to_numpy(dtype=int)
    proba = pred_df[[f"p_bucket_{i}" for i in range(scheme.n_buckets)]].to_numpy()
    p_loss = pred_df["p_loss"].to_numpy()
    p_haul = pred_df["p_haul"].to_numpy()
    y_loss = (y <= 2).astype(int)
    y_haul = (y >= 10).astype(int)
    groups = pred_df["position"].astype(str) + "_" + pred_df["GW_global"].astype(str)
    has_distribution = bool(np.isfinite(proba).all())

    metrics = {
        "bucket_logloss": float("nan"),
        "bucket_brier": float("nan"),
        "ev_mae": mae(y, pred_df["expected_points"]),
        "ev_rmse": rmse(y, pred_df["expected_points"]),
        "bias": bias(y, pred_df["expected_points"]),
        "total_calibration": total_calibration(y, pred_df["expected_points"]),
        "loss_brier": float("nan"),
        "loss_auc": float("nan"),
        "haul_brier": float("nan"),
        "haul_auc": float("nan"),
        "spearman_pos_gw": spearman_by_group(y, pred_df["expected_points"], groups),
    }
    if has_distribution:
        metrics.update({
            "bucket_logloss": float(
                log_loss(y_bucket, normalize_proba(proba), labels=list(range(scheme.n_buckets)))
            ),
            "bucket_brier": multiclass_brier(y_bucket, proba, scheme.n_buckets),
            "loss_brier": float(np.mean((p_loss - y_loss) ** 2)),
            "loss_auc": safe_auc(y_loss, p_loss),
            "haul_brier": float(np.mean((p_haul - y_haul) ** 2)),
            "haul_auc": safe_auc(y_haul, p_haul),
        })
    return metrics


def captaincy_metrics(pred_df):
    pool = pred_df[pred_df["position"].isin(["MID", "FWD"])].copy()
    if pool.empty:
        return {"cap_ev": float("nan"), "cap_haul": float("nan"), "cap_tilt": float("nan")}
    out = {
        "cap_ev": top1_capture(pool, "GW_global", "actual_points", "expected_points"),
        "cap_haul": float("nan"),
        "cap_tilt": float("nan"),
    }
    if np.isfinite(pool["p_haul"].to_numpy()).all():
        pool["haul_tilt"] = pool["expected_points"] * (1.0 + pool["p_haul"])
        out["cap_haul"] = top1_capture(pool, "GW_global", "actual_points", "p_haul")
        out["cap_tilt"] = top1_capture(pool, "GW_global", "actual_points", "haul_tilt")
    return out


def prediction_frame_for_model(model, test_df, feature_cols, position):
    proba = model.predict_distribution(test_df[feature_cols])
    expected = expected_points_from_proba(proba, model.bucket_values_)
    out = pd.DataFrame({
        "scheme": model.scheme.name,
        "n_buckets": model.scheme.n_buckets,
        "model": model.name,
        "position": position,
        "GW_global": test_df["GW_global"].to_numpy(),
        "actual_points": test_df[features.TARGET_COL].to_numpy(dtype=float),
        "bucket": points_to_buckets(test_df[features.TARGET_COL], model.scheme),
        "expected_points": expected,
        "p_loss": proba[:, model.scheme.loss_indices].sum(axis=1),
        "p_haul": proba[:, model.scheme.haul_indices].sum(axis=1),
    })
    for i in range(model.scheme.n_buckets):
        out[f"p_bucket_{i}"] = proba[:, i]
    return out


REGRESSION_BASELINE = "catboost_regression"


def fit_regression_baseline(pos_train, feature_cols, position, quick=False):
    """The production point-forecast: tuned per-position CatBoost regression
    (fpl.model.models.fit_model auto-loads tuned_params_<POS>_catboost.json)."""
    if quick:
        params = point_models.CATBOOST_PARAMS.copy()
        params["iterations"] = 80
        model = CatBoostRegressor(**params)
        model.fit(pos_train[feature_cols], pos_train[features.TARGET_COL])
        return model
    return point_models.fit_model(
        "catboost", pos_train[feature_cols], pos_train[features.TARGET_COL], position=position
    )


def regression_prediction_frame(predictions, test_df, position, scheme):
    """Wrap a point forecast in the bucket prediction-frame format so it can be
    scored side-by-side. Distribution columns are NaN: a single number carries
    no P(blank)/P(haul), which is the whole comparison."""
    out = pd.DataFrame({
        "scheme": scheme.name,
        "n_buckets": scheme.n_buckets,
        "model": REGRESSION_BASELINE,
        "position": position,
        "GW_global": test_df["GW_global"].to_numpy(),
        "actual_points": test_df[features.TARGET_COL].to_numpy(dtype=float),
        "bucket": points_to_buckets(test_df[features.TARGET_COL], scheme),
        "expected_points": np.asarray(predictions, dtype=float),
        "p_loss": float("nan"),
        "p_haul": float("nan"),
    })
    for i in range(scheme.n_buckets):
        out[f"p_bucket_{i}"] = float("nan")
    return out


def format_metric(value, digits=4):
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_table(title, rows, columns):
    print(f"\n--- {title} ---")
    widths = {
        col: max(len(col), *(len(format_metric(row[col])) for row in rows))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    for row in rows:
        print("  ".join(format_metric(row[col]).ljust(widths[col]) for col in columns))


def report_predictions(predictions, n_schemes=1):
    """Group a stacked prediction frame by scheme/model/position, print the
    metric tables, and return (predictions, pooled_df, per_position_df)."""
    per_pos_rows = []
    pooled_rows = []
    for (scheme_name, model_name, position), gdf in predictions.groupby(["scheme", "model", "position"], sort=True):
        metrics = evaluate_prediction_frame(gdf)
        per_pos_rows.append({"scheme": scheme_name, "model": model_name, "position": position, **metrics})
    for (scheme_name, model_name), gdf in predictions.groupby(["scheme", "model"], sort=True):
        metrics = evaluate_prediction_frame(gdf)
        cap = captaincy_metrics(gdf)
        pooled_rows.append({"scheme": scheme_name, "model": model_name, **metrics, **cap})

    def _logloss_key(row):
        value = row["bucket_logloss"]
        return float("inf") if pd.isna(value) else value

    per_pos_rows.sort(key=lambda row: (row["scheme"], row["position"], _logloss_key(row)))
    pooled_rows.sort(key=lambda row: (row["scheme"], _logloss_key(row)))

    print_table(
        "Per-position probabilistic quality",
        per_pos_rows,
        [
            "scheme",
            "position",
            "model",
            "bucket_logloss",
            "bucket_brier",
            "ev_mae",
            "bias",
            "total_calibration",
            "loss_auc",
            "haul_auc",
            "spearman_pos_gw",
        ],
    )
    print_table(
        "Pooled bake-off",
        pooled_rows,
        [
            "scheme",
            "model",
            "bucket_logloss",
            "bucket_brier",
            "ev_mae",
            "ev_rmse",
            "bias",
            "total_calibration",
            "loss_brier",
            "loss_auc",
            "haul_brier",
            "haul_auc",
            "spearman_pos_gw",
            "cap_ev",
            "cap_haul",
            "cap_tilt",
        ],
    )
    if n_schemes > 1:
        comparable_rows = sorted(pooled_rows, key=lambda row: (row["model"], row["ev_mae"]))
        print_table(
            "Cross-scheme comparable decision metrics",
            comparable_rows,
            [
                "model",
                "scheme",
                "ev_mae",
                "ev_rmse",
                "loss_brier",
                "loss_auc",
                "haul_brier",
                "haul_auc",
                "spearman_pos_gw",
                "cap_ev",
                "cap_haul",
                "cap_tilt",
            ],
        )
    return predictions, pd.DataFrame(pooled_rows), pd.DataFrame(per_pos_rows)


def evaluate_walk_forward(
    start_gw=153,
    end_gw=183,
    retrain_every=4,
    quick=False,
    bucket_scheme_name="coarse5",
    model_names=None,
    include_regression=True,
    use_tuned=True,
):
    """Walk-forward head-to-head: bucket models vs the tuned production
    CatBoost regression, each GW predicted using only strictly-earlier data.

    This is the decision-grade version of evaluate_static_split: a static
    split fits once and can flatter a model; walk-forward retrains every
    `retrain_every` GWs, matching how fpl.model.predict backtests the
    production forecaster. `use_tuned=True` lets the bucket CatBoost reuse the
    tuned per-position regression hyperparameters so the comparison is not
    tuned-vs-defaults.
    """
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)
    scheme = get_bucket_scheme(bucket_scheme_name)
    names = model_names or ["catboost_bucket", "catboost_hurdle_bucket"]

    print(
        "\n=== Probabilistic bucket walk-forward "
        f"(GW {start_gw}-{end_gw}, retrain every {retrain_every}, scheme={scheme.name}, "
        f"tuned_params={use_tuned}) ==="
    )
    print("Models: " + ", ".join(names + ([REGRESSION_BASELINE] if include_regression else [])))
    if quick:
        print("Quick mode: GBM iterations reduced to 80 for fast exploration.")

    pred_frames = []
    bucket_cache = {}
    regression_cache = {}
    last_trained_gw = None
    available_gws = sorted(set(df["GW_global"].unique()) & set(range(start_gw, end_gw + 1)))
    for gw in available_gws:
        if last_trained_gw is None or gw - last_trained_gw >= retrain_every:
            train_df = df[df["GW_global"] < gw]
            if train_df.empty:
                continue
            for position in POSITIONS:
                pos_train = train_df[train_df["position"] == position]
                if pos_train.empty:
                    continue
                fitted = []
                for model in candidate_models(
                    quick=quick, scheme=scheme, model_names=names,
                    position=position, use_tuned=use_tuned,
                ):
                    fitted.append(model.fit(pos_train, feature_cols))
                bucket_cache[position] = fitted
                if include_regression:
                    regression_cache[position] = fit_regression_baseline(
                        pos_train, feature_cols, position, quick=quick
                    )
            last_trained_gw = gw
            print(f"Retrained models for GW {gw}")

        test_df = df[df["GW_global"] == gw]
        for position in POSITIONS:
            pos_test = test_df[test_df["position"] == position]
            if pos_test.empty or position not in bucket_cache:
                continue
            for model in bucket_cache[position]:
                pred_frames.append(prediction_frame_for_model(model, pos_test, feature_cols, position))
            if include_regression and position in regression_cache:
                point_pred = regression_cache[position].predict(pos_test[feature_cols])
                pred_frames.append(regression_prediction_frame(point_pred, pos_test, position, scheme))

    predictions = pd.concat(pred_frames, ignore_index=True)
    return report_predictions(predictions)


def evaluate_static_split(
    train_max_gw=152,
    test_min_gw=153,
    test_max_gw=183,
    quick=False,
    bucket_scheme_names=None,
    model_names=None,
):
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)
    train_df = df[df["GW_global"] <= train_max_gw]
    test_df = df[(df["GW_global"] >= test_min_gw) & (df["GW_global"] <= test_max_gw)]
    schemes = [get_bucket_scheme(name) for name in (bucket_scheme_names or ["coarse5"])]

    print(
        "\n=== Probabilistic bucket bake-off "
        f"(train GW<= {train_max_gw}, test GW {test_min_gw}-{test_max_gw}) ==="
    )
    print("Schemes: " + ", ".join(f"{scheme.name} ({scheme.n_buckets} buckets)" for scheme in schemes))
    if model_names is not None:
        print("Models: " + ", ".join(model_names))
    if quick:
        print("Quick mode: GBM iterations reduced to 80 for fast exploration.")

    pred_frames = []
    for scheme in schemes:
        print("Buckets for " + scheme.name + ": " + ", ".join(f"{i}={label}" for i, label in enumerate(scheme.labels)))
        for position in POSITIONS:
            pos_train = train_df[train_df["position"] == position]
            pos_test = test_df[test_df["position"] == position]
            if pos_train.empty or pos_test.empty:
                continue
            for model in candidate_models(quick=quick, scheme=scheme, model_names=model_names):
                print(f"Fitting {model.name:<24} scheme={scheme.name:<8} position={position}")
                model.fit(pos_train, feature_cols)
                pred_frames.append(prediction_frame_for_model(model, pos_test, feature_cols, position))

    predictions = pd.concat(pred_frames, ignore_index=True)
    return report_predictions(predictions, n_schemes=len(schemes))


def walk_forward_predictions_csv(
    start_gw=153,
    end_gw=183,
    retrain_every=4,
    quick=False,
    bucket_scheme_name="coarse5",
    model_name="catboost_bucket",
    use_tuned=True,
):
    """Walk-forward predictions CSV from the bucket model for MILP backtesting.

    Returns a dataframe in fpl/milp/optimize.py format:
    player_id, GW, name, position, team, value, predicted_total_points, actual_total_points
    """
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    feature_cols = features.feature_columns(df)
    scheme = get_bucket_scheme(bucket_scheme_name)

    print(
        f"\n=== Bucket model predictions CSV (GW {start_gw}-{end_gw}, model={model_name}, "
        f"tuned_params={use_tuned}) ==="
    )

    rows = []
    bucket_cache = {}
    last_trained_gw = None
    available_gws = sorted(set(df["GW_global"].unique()) & set(range(start_gw, end_gw + 1)))

    for gw in available_gws:
        if last_trained_gw is None or gw - last_trained_gw >= retrain_every:
            train_df = df[df["GW_global"] < gw]
            if train_df.empty:
                continue
            for position in POSITIONS:
                pos_train = train_df[train_df["position"] == position]
                if pos_train.empty:
                    continue
                model = candidate_models(
                    quick=quick, scheme=scheme, model_names=[model_name],
                    position=position, use_tuned=use_tuned,
                )[0]
                model.fit(pos_train, feature_cols)
                bucket_cache[position] = model
            last_trained_gw = gw
            print(f"Retrained models for GW {gw}")

        test_df = df[df["GW_global"] == gw].copy()
        test_df["predicted_total_points"] = 0.0

        for position in POSITIONS:
            mask = test_df["position"] == position
            if mask.any() and position in bucket_cache:
                model = bucket_cache[position]
                proba = model.predict_distribution(test_df.loc[mask, feature_cols])
                expected = expected_points_from_proba(proba, model.bucket_values_)
                test_df.loc[mask, "predicted_total_points"] = expected

        rows.append(test_df)

    result = pd.concat(rows, ignore_index=True)
    out_cols = ["player_id", "GW_global", "name", "position", "team", "value",
                "predicted_total_points", "total_points"]
    result = result[out_cols].rename(columns={"GW_global": "GW", "total_points": "actual_total_points"})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare probabilistic bucket models for FPL points.")
    parser.add_argument("--train-max-gw", type=int, default=152)
    parser.add_argument("--test-min-gw", type=int, default=153)
    parser.add_argument("--test-max-gw", type=int, default=183)
    parser.add_argument("--quick", action="store_true", help="Reduce GBM iterations for a faster smoke run.")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["coarse5"],
        choices=sorted(BUCKET_SCHEMES),
        help="Bucket schemes to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=MODEL_NAMES,
        help="Optional subset of probabilistic model candidates.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Walk-forward head-to-head over [test-min-gw, test-max-gw] instead of the static split; "
        "includes the tuned production CatBoost regression as baseline.",
    )
    parser.add_argument("--retrain-every", type=int, default=4,
                        help="(walk-forward) Retrain every N gameweeks.")
    parser.add_argument("--no-regression-baseline", action="store_true",
                        help="(walk-forward) Skip the production CatBoost regression baseline.")
    parser.add_argument("--no-tuned-params", action="store_true",
                        help="(walk-forward) Use hand-set CatBoost defaults instead of the tuned "
                        "per-position params for the bucket models.")
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Export walk-forward bucket predictions to CSV for MILP backtesting (instead of evaluation metrics).",
    )
    parser.add_argument(
        "--export-model",
        type=str,
        default="catboost_bucket",
        choices=MODEL_NAMES,
        help="(export-predictions) Model to use for predictions export.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="(export-predictions) Output CSV path.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.export_predictions:
        preds = walk_forward_predictions_csv(
            start_gw=args.test_min_gw,
            end_gw=args.test_max_gw,
            retrain_every=args.retrain_every,
            quick=args.quick,
            bucket_scheme_name=args.schemes[0],
            model_name=args.export_model,
            use_tuned=not args.no_tuned_params,
        )
        output_path = args.output or str(config.PREDICTIONS_PATH)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(output_path, index=False)
        print(f"Saved {len(preds)} rows to {output_path}")
        return
    if args.walk_forward:
        evaluate_walk_forward(
            start_gw=args.test_min_gw,
            end_gw=args.test_max_gw,
            retrain_every=args.retrain_every,
            quick=args.quick,
            bucket_scheme_name=args.schemes[0],
            model_names=args.models,
            include_regression=not args.no_regression_baseline,
            use_tuned=not args.no_tuned_params,
        )
        return
    evaluate_static_split(
        train_max_gw=args.train_max_gw,
        test_min_gw=args.test_min_gw,
        test_max_gw=args.test_max_gw,
        quick=args.quick,
        bucket_scheme_names=args.schemes,
        model_names=args.models,
    )


if __name__ == "__main__":
    main()
