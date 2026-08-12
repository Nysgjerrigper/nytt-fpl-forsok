"""
Registry of cheap, non-data-hungry model types tried per position, so we can
compare them and blend the best ones into an ensemble instead of betting
everything on one algorithm.

Tree-based models (LightGBM, XGBoost, Random Forest, Extra Trees) get raw
features - they handle the NaNs at the start of a player's career fine
(LightGBM/XGBoost natively; the sklearn forests via the imputer below).
Linear/distance-based models (Ridge, ElasticNet, kNN, OLS, the SVR variants)
need imputation and scaling to behave.

"ols" (plain, unregularized multiple linear regression on the full feature
set) is the designated INDEX: the simple, textbook benchmark every other
model/baseline in this project is ultimately judged against, the way a
passive market index is the bar an active strategy has to clear - not just
another entry in the comparison table. See RESEARCH_LOG.md for the standing
verdict on whether anything actually beats it.

Every name registered here becomes a real NNLS blend candidate in
fpl.model.train (not just a printed comparison column) - including models
that a preliminary check found don't help (rbf_svr, xgboost - see
RESEARCH_LOG.md). They're kept rather than deleted: NNLS naturally assigns
them ~0 blend weight if they don't pull their weight, so leaving them in
costs nothing at inference time and preserves the comparison for anyone
revisiting the choice later.
"""
import json
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    num_leaves=31,
    min_data_in_leaf=30,
    learning_rate=0.05,
    n_estimators=300,
    verbosity=-1,
)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1,
)

# MAE loss rather than CatBoost's default RMSE: the LinearSVR result (RESEARCH_LOG.md)
# showed MAE-aligned objectives handle FPL's outlier hauls better than squared error.
# allow_writing_files=False stops CatBoost dumping a catboost_info/ dir into the CWD.
CATBOOST_PARAMS = dict(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="MAE",
    random_seed=0,
    verbose=0,
    allow_writing_files=False,
)

# RBF-kernel SVR is O(n^2)-O(n^3) in training rows - FPL's per-position row counts
# (tens of thousands) make a full fit impractical, especially inside a walk-forward
# backtest that refits every few gameweeks. Capping keeps it usable in the same
# pipeline as every other model, at the cost of not seeing all available training
# data (see RESEARCH_LOG.md for the standalone check that motivated this).
RBF_SVR_SAMPLE_CAP = 8000


class OptionalModelDependencyError(ImportError):
    """A research-only model was requested without its optional dependency."""


class _PyTabKitRegressor(BaseEstimator, RegressorMixin):
    """Lazy sklearn-compatible adapter for a PyTabKit regressor.

    PyTabKit is deliberately not a production dependency.  Importing this module and
    fitting the standing CatBoost model therefore never imports PyTabKit (or torch/FAISS).
    Numerical missing values are handled by the surrounding sklearn ``Pipeline``; since
    each call to :func:`fit_model` constructs a new pipeline, the imputer is fitted only
    on that fold's training rows.
    """

    CLASS_NAMES = {
        "realmlp": "RealMLP_TD_Regressor",
        "tabm": "TabM_D_Regressor",
        "tabr": "TabR_S_D_Regressor",
    }

    def __init__(self, model_name: str, random_state: int = 0,
                 model_params: Mapping[str, Any] | None = None):
        self.model_name = model_name
        self.random_state = random_state
        self.model_params = model_params

    def fit(self, X, y):
        if self.model_name == "tabr" and importlib.util.find_spec("faiss") is None:
            raise OptionalModelDependencyError(
                "tabr is a research-only expert; install pytabkit, FAISS (faiss-cpu), and skorch "
                "before running its study"
            )
        try:
            module = importlib.import_module("pytabkit")
            estimator_class = getattr(module, self.CLASS_NAMES[self.model_name])
        except (ImportError, AttributeError) as exc:
            extra = ", faiss-cpu, and skorch" if self.model_name == "tabr" else ""
            raise OptionalModelDependencyError(
                f"{self.model_name} is a research-only expert; install pytabkit{extra} "
                "using requirements-research.txt before running its study"
            ) from exc
        params = {"random_state": self.random_state, **dict(self.model_params or {})}
        self.estimator_ = estimator_class(**params)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).ravel()


class _RavelPredict(BaseEstimator, RegressorMixin):
    """PLSRegression.predict returns a 2-D (n, 1) array, which would silently
    mis-broadcast in the NNLS blend's weighted sum - flatten it to 1-D like
    every other regressor in the registry."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.estimator_ = self.estimator  # sklearn's check_is_fitted looks for a trailing-underscore attr
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).ravel()


class _CappedSampleEstimator(BaseEstimator, RegressorMixin):
    """Wraps an estimator, subsampling the training set down to `cap` rows before
    fitting if it's larger. Only affects fit(); predict() is unrestricted."""

    def __init__(self, estimator, cap=RBF_SVR_SAMPLE_CAP, random_state=0):
        self.estimator = estimator
        self.cap = cap
        self.random_state = random_state

    def fit(self, X, y):
        if len(X) > self.cap:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.cap, replace=False)
            X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


def _scaled_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _tree_pipeline(estimator):
    # sklearn tree ensembles (unlike LightGBM) can't take NaN directly.
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("model", estimator),
    ])


class TwoStageHurdle(BaseEstimator, RegressorMixin):
    """Minutes-aware two-stage forecast: E[points] = P(minutes > 0) x E[points | played].

    FPL points are EXACTLY zero whenever a player does not come on, and 59% of
    player-gameweek rows are 0-minute rows - so the expectation decomposes without
    approximation into a participation probability times a conditional-on-playing
    expectation (the classic hurdle model for zero-inflated targets). A single MAE
    regressor on the mixed target fits its conditional MEDIAN, which for any
    rotation-risk player is dragged toward 0 by the zero mass; here the zero mass is
    handled by a dedicated CatBoost classifier (Logloss) and the regression head
    (CatBoost, MAE, per-position tuned params when available) learns only from rows
    where the player actually played.

    fit() needs the row's MINUTES as a second training label (points alone cannot
    recover participation: a played row can legitimately score 0 or negative), which
    is why models.fit_model threads a `minutes=` argument through. predict() needs
    only the ordinary feature matrix, so a fitted instance is a drop-in
    PositionEnsemble member.
    """

    def __init__(self, position=None):
        self.position = position

    def fit(self, X, y, minutes):
        played = np.asarray(minutes, dtype=float) > 0

        reg_params = _tuned_params("catboost", self.position) or dict(CATBOOST_PARAMS)
        clf_params = dict(reg_params)
        clf_params["loss_function"] = "Logloss"

        # Degenerate participation (all played / none played) would crash a one-class
        # classifier; fall back to that constant probability instead.
        if played.all() or not played.any():
            self.p_played_const_ = float(played.mean())
            self.classifier_ = None
        else:
            self.p_played_const_ = None
            self.classifier_ = CatBoostClassifier(**clf_params)
            self.classifier_.fit(X, played.astype(int))

        if played.any():
            X_played = X[played] if not hasattr(X, "loc") else X.loc[played]
            self.regressor_ = CatBoostRegressor(**reg_params)
            self.regressor_.fit(X_played, np.asarray(y, dtype=float)[played])
        else:
            self.regressor_ = None  # nobody ever played: E[pts | played] unlearnable
        return self

    def predict(self, X):
        if self.regressor_ is None:
            return np.zeros(len(X))
        p_played = (
            np.full(len(X), self.p_played_const_)
            if self.classifier_ is None
            else self.classifier_.predict_proba(X)[:, 1]
        )
        return p_played * self.regressor_.predict(X)


class ThreeClassHurdle(BaseEstimator, RegressorMixin):
    """Minutes-model v2 (TODO 2.1 follow-on): a 3-class minutes stage instead of v1's binary.

        E[points] = P(cameo) x E[pts | cameo] + P(full) x E[pts | full]

    where cameo = 1-59 minutes and full = 60+ (the FPL appearance-points boundary: 1 pt
    for a cameo, 2 pts for 60+, and the clean-sheet eligibility threshold). v1's binary
    P(played) collapses those two regimes, but their conditional points distributions are
    structurally different - a cameo is capped near the 1-point appearance floor while a
    full game carries the whole upside tail - so one pooled E[pts | played] regressor is
    fitting a mixture. Splitting the stage lets each regressor learn a homogeneous target.

    Same plumbing contract as TwoStageHurdle: fit() needs the `minutes` training label
    (threaded by models.fit_model), predict() takes only features, so a fitted instance
    is a drop-in PositionEnsemble member. Classes absent from training (e.g. a slice where
    nobody ever played) degrade to constant probabilities / a zero regressor.
    """

    CAMEO_MAX = 59  # minutes; 60+ is a "full" appearance (FPL's 2-point boundary)

    def __init__(self, position=None):
        self.position = position

    def _minute_class(self, minutes):
        m = np.asarray(minutes, dtype=float)
        return np.where(m <= 0, 0, np.where(m <= self.CAMEO_MAX, 1, 2))

    def fit(self, X, y, minutes):
        y = np.asarray(y, dtype=float)
        klass = self._minute_class(minutes)

        reg_params = _tuned_params("catboost", self.position) or dict(CATBOOST_PARAMS)
        clf_params = dict(reg_params)
        clf_params["loss_function"] = "MultiClass"
        # CatBoost only allows `subsample` (present in tuned params) with a sampling
        # bootstrap; MultiClass defaults to bayesian, which rejects it.
        if "subsample" in clf_params and "bootstrap_type" not in clf_params:
            clf_params["bootstrap_type"] = "Bernoulli"

        present = np.unique(klass)
        if len(present) < 2:
            self.class_probs_const_ = {int(present[0]): 1.0}
            self.classifier_ = None
        else:
            self.class_probs_const_ = None
            self.classifier_ = CatBoostClassifier(**clf_params)
            self.classifier_.fit(X, klass)

        self.regressors_ = {}
        for c in (1, 2):  # class 0 (DNP) contributes exactly 0 points by construction
            mask = klass == c
            if mask.any():
                X_c = X.loc[mask] if hasattr(X, "loc") else X[mask]
                reg = CatBoostRegressor(**reg_params)
                reg.fit(X_c, y[mask])
                self.regressors_[c] = reg
        return self

    def _class_proba(self, X):
        out = np.zeros((len(X), 3), dtype=float)
        if self.classifier_ is None:
            for c, p in self.class_probs_const_.items():
                out[:, c] = p
            return out
        raw = self.classifier_.predict_proba(X)
        for i, cls in enumerate(np.asarray(self.classifier_.classes_, dtype=int)):
            out[:, cls] = raw[:, i]
        return out

    def predict(self, X):
        proba = self._class_proba(X)
        pred = np.zeros(len(X))
        for c, reg in self.regressors_.items():
            pred += proba[:, c] * reg.predict(X)
        return pred


class LambdaRankScorer(BaseEstimator, RegressorMixin):
    """Within-gameweek ranking model mapped back to the points scale (TODO 2.3, audit C3).

    Three "better metrics, fewer points" episodes suggested the MILP consumes within-GW
    RANKING, not point levels - this tests that mechanism directly. A LightGBM lambdarank
    objective optimizes NDCG over query groups; each group is one GW_global round (models
    are already per-position, so groups are effectively (GW, position) as the audit asked).

    Two departures from a stock ranker:
    - Relevance labels must be non-negative integers: points are clipped to [0, `label_max`]
      (negative rows - cards/OGs - carry no ranking signal worth separating from 0) and gains
      are LINEAR (label_gain=0..label_max), not the default 2^rel-1, which at 15+ point hauls
      would make a single haul dominate every gradient in the round.
    - The MILP's transfer penalty and chip thresholds are absolute-scale, so raw ranker scores
      are unusable directly; an isotonic regression fit on the training rows maps scores back
      to expected points monotonically - the ranking is preserved exactly, only the scale is
      restored.

    fit() needs the row's GW_global as a group label, threaded through models.fit_model the
    same way the hurdle's `minutes` label is. predict() takes only features (scoring needs no
    groups), so a fitted instance is a drop-in PositionEnsemble member.
    """

    def __init__(self, position=None, label_max=15):
        self.position = position
        self.label_max = label_max

    def fit(self, X, y, gw):
        gw = np.asarray(gw)
        order = np.argsort(gw, kind="stable")  # ranker requires group-contiguous rows
        X_ord = X.iloc[order] if hasattr(X, "iloc") else X[order]
        y_ord = np.asarray(y, dtype=float)[order]
        rel = np.clip(np.rint(y_ord), 0, self.label_max).astype(int)
        group_sizes = np.unique(gw[order], return_counts=True)[1]

        params = _tuned_params("lightgbm", self.position) or dict(LGB_PARAMS)
        params = {k: v for k, v in params.items() if k not in ("objective", "metric")}
        self.ranker_ = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=list(range(self.label_max + 1)),
            **params,
        )
        self.ranker_.fit(X_ord, rel, group=group_sizes)

        scores = self.ranker_.predict(X_ord)
        self.score_to_points_ = IsotonicRegression(out_of_bounds="clip").fit(scores, y_ord)
        return self

    def predict(self, X):
        return self.score_to_points_.predict(self.ranker_.predict(X))


PRODUCTION_FACTORIES = {
    "ols": lambda: _scaled_pipeline(LinearRegression()),
    "lightgbm": lambda: lgb.LGBMRegressor(**LGB_PARAMS),
    "ridge": lambda: _scaled_pipeline(Ridge(alpha=1.0)),
    "elasticnet": lambda: _scaled_pipeline(ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=5000)),
    "random_forest": lambda: _tree_pipeline(
        RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=0)
    ),
    "extra_trees": lambda: _tree_pipeline(
        ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=0)
    ),
    "knn": lambda: _scaled_pipeline(KNeighborsRegressor(n_neighbors=25, weights="distance")),
    # Preliminary check (RESEARCH_LOG.md): beat every other model at every position on a
    # static split - epsilon-insensitive loss tracks MAE/MASE more directly than the
    # squared-error models above, which FPL's zero-inflated, outlier-haul scoring rewards.
    "linear_svr": lambda: _scaled_pipeline(LinearSVR(C=1.0, max_iter=20000)),
    # Preliminary check: underperformed linear_svr and the ensemble at every position, and
    # doesn't scale to full per-position row counts - kept as a real (capped) blend
    # candidate rather than deleted, per project convention of not hiding negative results.
    "rbf_svr": lambda: _CappedSampleEstimator(_scaled_pipeline(SVR(kernel="rbf", C=1.0))),
    # Preliminary check: underperformed the already-tuned LightGBM at every position with
    # generic hyperparameters - kept in the registry (not tuned further) for the same reason.
    "xgboost": lambda: xgb.XGBRegressor(**XGB_PARAMS),
    # Third GBM flavor: ordered boosting avoids a subtle target-leakage bias LightGBM/XGBoost
    # share, and MAE loss aligns with what LinearSVR's win suggested matters here.
    "catboost": lambda: CatBoostRegressor(**CATBOOST_PARAMS),
    # Partial-least-squares regression: compresses the ~70 heavily-collinear rolling-window
    # features into orthogonal components before regressing - the classic econometrics answer
    # to exactly the collinearity that makes plain OLS trail Ridge/ElasticNet here.
    "pls": lambda: _scaled_pipeline(_RavelPredict(PLSRegression(n_components=20))),
    # Minutes-aware hurdle (audit C1): P(plays) x E[pts | played]. Needs the minutes
    # training label, so fit_model special-cases it below - the factory entry exists so
    # the name participates in MODEL_NAMES (comparison tables, bake-off, blends).
    "catboost_hurdle": lambda: TwoStageHurdle(),
    # Minutes-model v2 (TODO 2.1 follow-on): 3-class minutes stage (DNP / cameo / 60+),
    # each playing class with its own conditional points regressor. Same minutes-label
    # special-casing as catboost_hurdle. Backtested 2026-07-23: 2053 vs 2057, a dead tie
    # (sign test 15-15) - kept as a documented negative result, like catboost_hurdle.
    "catboost_hurdle3": lambda: ThreeClassHurdle(),
    # Within-GW LambdaRank mapped isotonically back to points (audit C3). Needs the GW_global
    # group label, so fit_model special-cases it below - the factory entry exists so the name
    # participates in MODEL_NAMES (comparison tables, bake-off, blends).
    "lgbm_rank": lambda: LambdaRankScorer(),
}

MODEL_NAMES = list(PRODUCTION_FACTORIES.keys())


@dataclass(frozen=True)
class ExpertSpec:
    """Complete, reproducible definition of a model-tournament expert.

    ``factory`` receives constructor parameters, a seed, and the position.  Search-space
    callables receive an Optuna-compatible trial and the seed, allowing tuning.py to stay
    algorithm-agnostic. ``preprocessing`` is recorded explicitly even where it is ``raw``
    so experiment metadata can prove how missing values were handled.
    """

    factory: Callable[[Mapping[str, Any], int, str | None], BaseEstimator]
    search_space: Callable[[Any, int], dict[str, Any]] | None
    default_params: Callable[[int], dict[str, Any]]
    preprocessing: str
    auxiliary_labels: tuple[str, ...]
    seed: int
    provenance: str
    research_only: bool = True


def _lgbm_space(trial, seed: int, *, objective: str = "regression") -> dict[str, Any]:
    return {
        "objective": objective,
        "metric": "mae",
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": seed,
        "verbosity": -1,
    }


def _xgb_space(trial, seed: int, *, objective: str = "reg:squarederror") -> dict[str, Any]:
    return {
        "objective": objective,
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "random_state": seed,
        "n_jobs": -1,
    }


def _catboost_space(trial, seed: int, *, loss: str = "MAE") -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "loss_function": loss,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
    }


def _hist_space(trial, seed: int, *, loss: str) -> dict[str, Any]:
    return {
        "loss": loss,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 600),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "random_state": seed,
    }


def _extra_trees_space(trial, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "n_jobs": -1,
        "random_state": seed,
    }


def _linear_svr_space(trial, seed: int) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.0, 1.0),
        "loss": trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
        "max_iter": 20000,
        "random_state": seed,
    }


def _pytab_space(trial, seed: int) -> dict[str, Any]:
    # Common PyTabKit sklearn-interface knobs. Model-specific defaults retain the
    # architecture decisions from the upstream implementation.
    return {
        "n_cv": 1,
        "n_refit": 0,
        "n_epochs": trial.suggest_categorical("n_epochs", [128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "verbosity": 0,
    }


def _defaults(params: Mapping[str, Any]) -> Callable[[int], dict[str, Any]]:
    """Return seed-aware, copy-on-read defaults for one registered expert."""
    def resolve(seed: int) -> dict[str, Any]:
        resolved = dict(params)
        for key in ("random_state", "random_seed"):
            if key in resolved:
                resolved[key] = seed
        return resolved
    return resolve


def _raw_lgbm(params, seed, position):
    return lgb.LGBMRegressor(**dict(params))


def _raw_xgb(params, seed, position):
    return xgb.XGBRegressor(**dict(params))


def _raw_catboost(params, seed, position):
    return CatBoostRegressor(**dict(params))


def _imputed_hist(params, seed, position):
    return _tree_pipeline(HistGradientBoostingRegressor(**dict(params)))


def _imputed_extra_trees(params, seed, position):
    return _tree_pipeline(ExtraTreesRegressor(**dict(params)))


def _scaled_linear_svr(params, seed, position):
    return _scaled_pipeline(LinearSVR(**dict(params)))


def _pytab_factory(model_name: str):
    def factory(params, seed, position):
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", _PyTabKitRegressor(model_name, random_state=seed, model_params=params)),
        ])
    return factory


EXPERT_SPECS: dict[str, ExpertSpec] = {
    "catboost_mae": ExpertSpec(_raw_catboost, lambda t, s: _catboost_space(t, s, loss="MAE"), _defaults(CATBOOST_PARAMS),
                                "raw_native_nan", (), 0, "CatBoost MAE control"),
    "catboost_rmse": ExpertSpec(_raw_catboost, lambda t, s: _catboost_space(t, s, loss="RMSE"), _defaults({**CATBOOST_PARAMS, "loss_function": "RMSE"}),
                                 "raw_native_nan", (), 0, "CatBoost squared-loss challenger"),
    "lightgbm_l2": ExpertSpec(_raw_lgbm, lambda t, s: _lgbm_space(t, s, objective="regression"), _defaults(LGB_PARAMS),
                               "raw_native_nan", (), 0, "LightGBM L2 control"),
    "lightgbm_l1": ExpertSpec(_raw_lgbm, lambda t, s: _lgbm_space(t, s, objective="regression_l1"), _defaults({**LGB_PARAMS, "objective": "regression_l1"}),
                               "raw_native_nan", (), 0, "LightGBM L1 challenger"),
    "lightgbm_huber": ExpertSpec(_raw_lgbm, lambda t, s: _lgbm_space(t, s, objective="huber"), _defaults({**LGB_PARAMS, "objective": "huber"}),
                                  "raw_native_nan", (), 0, "LightGBM Huber challenger"),
    "xgboost_squared_error": ExpertSpec(
        _raw_xgb, lambda t, s: _xgb_space(t, s, objective="reg:squarederror"),
        _defaults({**XGB_PARAMS, "objective": "reg:squarederror"}),
        "raw_native_nan", (), 0, "XGBoost squared-error control"),
    "xgboost_absolute_error": ExpertSpec(
        _raw_xgb, lambda t, s: _xgb_space(t, s, objective="reg:absoluteerror"),
        _defaults({**XGB_PARAMS, "objective": "reg:absoluteerror"}),
        "raw_native_nan", (), 0, "XGBoost absolute-error challenger"),
    "hist_gradient_boosting_absolute": ExpertSpec(
        _imputed_hist, lambda t, s: _hist_space(t, s, loss="absolute_error"),
        _defaults({"loss": "absolute_error", "random_state": 0}),
        "fold_local_constant_imputation", (), 0, "sklearn HistGradientBoosting absolute loss"),
    "hist_gradient_boosting_squared": ExpertSpec(
        _imputed_hist, lambda t, s: _hist_space(t, s, loss="squared_error"),
        _defaults({"loss": "squared_error", "random_state": 0}),
        "fold_local_constant_imputation", (), 0, "sklearn HistGradientBoosting squared loss"),
    "extra_trees_tuned": ExpertSpec(_imputed_extra_trees, _extra_trees_space, _defaults({"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 0}),
                                     "fold_local_constant_imputation", (), 0,
                                     "sklearn ExtraTrees tuned challenger"),
    "linear_svr_tuned": ExpertSpec(_scaled_linear_svr, _linear_svr_space, _defaults({"C": 1.0, "max_iter": 20000, "random_state": 0}),
                                    "fold_local_imputation_and_scaling", (), 0,
                                    "sklearn LinearSVR tuned challenger"),
    "realmlp": ExpertSpec(_pytab_factory("realmlp"), _pytab_space, _defaults({}),
                           "fold_local_constant_imputation", (), 0,
                           "PyTabKit RealMLP tuned-default regressor"),
    "tabm": ExpertSpec(_pytab_factory("tabm"), _pytab_space, _defaults({}),
                        "fold_local_constant_imputation", (), 0,
                        "PyTabKit TabM default regressor"),
    "tabr": ExpertSpec(_pytab_factory("tabr"), _pytab_space, _defaults({}),
                        "fold_local_constant_imputation", (), 0,
                        "PyTabKit TabR small-default regressor; requires FAISS"),
}

# Backwards-compatible production names are also first-class specs, while MODEL_NAMES
# deliberately remains the standing production bake-off.  Research experts are callable
# through FACTORIES/fit_model only when explicitly selected by an experiment.
EXPERT_SPECS.update({
    "catboost": ExpertSpec(_raw_catboost, lambda t, s: _catboost_space(t, s, loss="MAE"), _defaults(CATBOOST_PARAMS),
                            "raw_native_nan", (), 0, "Production CatBoost MAE", False),
    "lightgbm": ExpertSpec(_raw_lgbm, lambda t, s: _lgbm_space(t, s), _defaults(LGB_PARAMS),
                            "raw_native_nan", (), 0, "Production LightGBM L2", False),
    "xgboost": ExpertSpec(_raw_xgb, lambda t, s: _xgb_space(t, s), _defaults(XGB_PARAMS),
                           "raw_native_nan", (), 0, "Production XGBoost squared error", False),
})

# This is the explicit public registry for experiment configuration and policy files.
# MODEL_NAMES remains production-only so normal train/predict runs cannot accidentally
# activate research experts with optional dependencies.
REGISTERED_MODEL_NAMES = tuple(EXPERT_SPECS)


def build_registered_model(name: str, params: Mapping[str, Any] | None = None,
                           *, seed: int | None = None, position: str | None = None):
    """Build a registered tournament expert without fitting it."""
    if name not in EXPERT_SPECS:
        if name in PRODUCTION_FACTORIES and params is None:
            return PRODUCTION_FACTORIES[name]()
        raise ValueError(f"unknown expert {name!r}")
    spec = EXPERT_SPECS[name]
    resolved_seed = spec.seed if seed is None else seed
    if params is None:
        params = spec.default_params(resolved_seed)
    # A caller-provided seed is authoritative across multi-seed stability checks.
    params = dict(params)
    for key in ("random_state", "random_seed"):
        if key in params:
            params[key] = resolved_seed
    return spec.factory(params, resolved_seed, position)


FACTORIES = dict(PRODUCTION_FACTORIES)
for _expert_name in EXPERT_SPECS:
    if _expert_name not in FACTORIES:
        FACTORIES[_expert_name] = lambda name=_expert_name: build_registered_model(name)

_TUNABLE = {name for name, spec in EXPERT_SPECS.items() if spec.search_space is not None}


def _tuned_params(name, position):
    """Per-(position, model) params saved by fpl.model.tuning, or None to use the
    hand-set defaults above. Loaded from config.MODELS_DIR (gitignored artifacts):
    the tuner writing a file is what activates it - delete the file to revert to
    defaults. Position-aware because the pipeline trains one model per position and
    there is no reason GK and MID should share a depth/learning-rate."""
    if position is None or name not in _TUNABLE:
        return None
    path = config.MODELS_DIR / f"tuned_params_{position}_{name}.json"
    if not path.exists():
        return None
    loaded = json.loads(path.read_text())
    # Underscore-prefixed keys are provenance metadata written by tuning.save_best_params
    # (e.g. "_meta": the GW cap the search ran under), not constructor arguments.
    return {k: v for k, v in loaded.items() if not k.startswith("_")}


def fit_model(name, X, y, position=None, minutes=None, gw=None, *, params=None, seed=None):
    """Fit one registry member. `minutes` is the per-row minutes TRAINING label (required by
    the hurdle: participation cannot be recovered from points alone) and `gw` the per-row
    GW_global group label (required by the ranker: query groups cannot be recovered from
    features); both are ignored by every other member - callers with a training frame should
    always pass both."""
    if name == "catboost_hurdle":
        if minutes is None:
            raise ValueError("catboost_hurdle needs the `minutes` training label")
        return TwoStageHurdle(position=position).fit(X, y, minutes=minutes)
    if name == "catboost_hurdle3":
        if minutes is None:
            raise ValueError("catboost_hurdle3 needs the `minutes` training label")
        return ThreeClassHurdle(position=position).fit(X, y, minutes=minutes)
    if name == "lgbm_rank":
        if gw is None:
            raise ValueError("lgbm_rank needs the `gw` (GW_global) group label")
        return LambdaRankScorer(position=position).fit(X, y, gw=gw)
    params = _tuned_params(name, position) if params is None else params
    if name in EXPERT_SPECS:
        model = build_registered_model(name, params=params, seed=seed, position=position)
    else:
        if name not in FACTORIES:
            raise ValueError(f"unknown model {name!r}")
        model = FACTORIES[name]()
    model.fit(X, y)
    return model
