"""
Registry of cheap, non-data-hungry model types tried per position, so we can
compare them and blend the best ones into an ensemble instead of betting
everything on one algorithm.

Tree-based models (LightGBM, Random Forest, Extra Trees) get raw features -
they handle the NaNs at the start of a player's career fine (LightGBM
natively; the sklearn forests via the imputer below). Linear/distance-based
models (Ridge, ElasticNet, kNN) need imputation and scaling to behave.
"""
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    num_leaves=31,
    min_data_in_leaf=30,
    learning_rate=0.05,
    n_estimators=300,
    verbosity=-1,
)


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
}

MODEL_NAMES = list(FACTORIES.keys())


def fit_model(name, X, y):
    model = FACTORIES[name]()
    model.fit(X, y)
    return model
