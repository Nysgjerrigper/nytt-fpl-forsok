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
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
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


FACTORIES = {
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
}

MODEL_NAMES = list(FACTORIES.keys())

# GBMs that fpl.model.tuning knows how to tune; only these ever get per-position params.
_TUNABLE = {"lightgbm": lgb.LGBMRegressor, "xgboost": xgb.XGBRegressor, "catboost": CatBoostRegressor}


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
    return json.loads(path.read_text())


def fit_model(name, X, y, position=None):
    params = _tuned_params(name, position)
    model = _TUNABLE[name](**params) if params is not None else FACTORIES[name]()
    model.fit(X, y)
    return model
