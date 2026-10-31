"""Earlier-only convex CNN/tree blends, separate from production/promotion policy."""
from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from fpl import config

_KEYS = ["player_id", "position", "origin_gw", "GW", "fixture"]
_PREDICTION = "predicted_total_points"


def _validate_forecasts(frame: pd.DataFrame) -> None:
    required = {*_KEYS, _PREDICTION, "actual_total_points", "value", "minutes"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Temporal blend requires nonempty unique-column origin fixture forecasts")
    if missing := required.difference(frame.columns):
        raise ValueError(f"Temporal blend forecasts missing columns: {sorted(missing)}")
    for column in ("player_id", "origin_gw", "GW", "fixture"):
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"Temporal blend {column} requires numeric integer IDs/gameweeks")
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or (values < 1).any() or (values % 1 != 0).any():
            raise ValueError(f"Temporal blend {column} requires positive integer IDs/gameweeks")
    for column in (_PREDICTION, "actual_total_points", "value", "minutes"):
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"Temporal blend {column} must be numeric")
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ValueError(f"Temporal blend has nonfinite {column}")
    if (frame.minutes < 0).any() or (frame.value <= 0).any() or (frame.GW < frame.origin_gw).any():
        raise ValueError("Temporal blend has invalid minutes/prices/origin ordering")
    if set(frame.position) != set(config.ONFIELD_POSITIONS):
        raise ValueError("Temporal blend requires exactly all four positions")
    if frame.duplicated(_KEYS).any():
        raise ValueError("Temporal blend has duplicate origin fixture identity")


def _paired(cnn: pd.DataFrame, tree: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_forecasts(cnn)
    _validate_forecasts(tree)
    if set(cnn.columns) != set(tree.columns):
        raise ValueError("Temporal blend paired forecast schemas differ")
    a, b = (frame.sort_values(_KEYS).reset_index(drop=True) for frame in (cnn, tree))
    metadata = sorted(set(tree.columns) - {_PREDICTION})
    try:
        pd.testing.assert_frame_equal(a[metadata], b[metadata], check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Temporal blend paired pools/outcomes/fixture metadata differ") from exc
    return a, b


def _weight(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("CNN weight must be a finite number in [0, 1]")
    return float(value)


def fit_earlier_weights(
    cnn_forecasts: pd.DataFrame, tree_forecasts: pd.DataFrame, start_gw: int, end_gw: int,
) -> dict[str, float]:
    """Fit no-intercept convex MSE weights using only earlier current-origin rounds.

    Sum fixture predictions/targets before fitting. For d=CNN-tree, the unconstrained
    weight is dot(d,y-tree)/dot(d,d), clipped to [0,1]. Identical predictions retain
    the simpler tree. This is a forecast diagnostic, not an optimizer/promotion claim.
    """
    for value in (start_gw, end_gw):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("Weight-fitting bounds must be positive integer gameweeks")
    if start_gw > end_gw:
        raise ValueError("Weight-fitting start_gw must not exceed end_gw")
    cnn, tree = _paired(cnn_forecasts, tree_forecasts)
    admitted = (tree.origin_gw == tree.GW) & tree.GW.between(start_gw, end_gw)
    selected = tree.loc[admitted, ["player_id", "position", "GW", "actual_total_points", _PREDICTION]].copy()
    selected = selected.rename(columns={_PREDICTION: "tree"})
    selected["cnn"] = cnn.loc[admitted, _PREDICTION]
    rounds = selected.groupby(["player_id", "position", "GW"], as_index=False).agg(
        actual_total_points=("actual_total_points", "sum"), tree=("tree", "sum"), cnn=("cnn", "sum"))
    weights = {}
    for position in config.ONFIELD_POSITIONS:
        group = rounds.loc[rounds.position == position]
        if group.empty:
            raise ValueError(f"No earlier current-origin rounds for {position}")
        difference = group.cnn.to_numpy(dtype=float) - group.tree.to_numpy(dtype=float)
        residual = group.actual_total_points.to_numpy(dtype=float) - group.tree.to_numpy(dtype=float)
        denominator = float(np.dot(difference, difference))
        numerator = float(np.dot(difference, residual))
        if not np.isfinite(denominator) or not np.isfinite(numerator):
            raise ValueError("Temporal blend weight calculation overflowed")
        weights[position] = float(np.clip(numerator / denominator, 0., 1.)) if denominator > 0 else 0.
    return weights


def blend_forecasts(
    cnn: pd.DataFrame, tree: pd.DataFrame, weights: dict[str, float],
) -> pd.DataFrame:
    """Blend exact paired fixture forecasts while preserving tree row order/metadata.

    Pure endpoints copy their selected member exactly, avoiding needless arithmetic.
    Weights must be frozen from earlier data by the caller; labels are never fitted here.
    """
    sorted_cnn, _ = _paired(cnn, tree)
    if set(weights) != set(config.ONFIELD_POSITIONS):
        raise ValueError("Temporal blend weights must cover exactly all four positions")
    validated = {position: _weight(weight) for position, weight in weights.items()}
    result = tree.copy()
    lookup = sorted_cnn.set_index(_KEYS)[_PREDICTION]
    aligned_cnn = lookup.reindex(pd.MultiIndex.from_frame(tree[_KEYS])).to_numpy(dtype=float)
    original_tree = tree[_PREDICTION].to_numpy(dtype=float)
    output = original_tree.copy()
    for position, weight in validated.items():
        mask = tree.position.to_numpy() == position
        if weight == 1:
            output[mask] = aligned_cnn[mask]
        elif weight != 0:
            output[mask] = (1 - weight) * original_tree[mask] + weight * aligned_cnn[mask]
    if not np.isfinite(output).all():
        raise ValueError("Temporal blend produced nonfinite predictions")
    result[_PREDICTION] = output
    return result


class TemporalCNNTreeBlendRegressor(RegressorMixin, BaseEstimator):
    """Lazy explicit-parameter tree/CNN members with a caller-frozen convex weight."""

    def __init__(self, *, cnn_weight: float, cnn_params: dict[str, Any], tree_params: dict[str, Any],
                 seed: int = 0, position: str | None = None) -> None:
        self.cnn_weight = cnn_weight
        self.cnn_params = cnn_params
        self.tree_params = tree_params
        self.seed = seed
        self.position = position

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TemporalCNNTreeBlendRegressor:
        """Fit active members on the same training features/labels, skipping zero weights."""
        self.close()
        weight = _weight(self.cnn_weight)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("Temporal blend seed must be a nonnegative integer")
        if self.position is not None and self.position not in config.ONFIELD_POSITIONS:
            raise ValueError("Temporal blend has unsupported position")
        if not isinstance(self.cnn_params, dict) or not isinstance(self.tree_params, dict):
            raise ValueError("Temporal blend requires explicit CNN/tree parameter dictionaries")
        try:
            if weight < 1:
                from fpl.model.models import fit_model
                self.tree_ = fit_model("catboost_rmse", X, y, position=self.position,
                                       params=dict(self.tree_params), seed=self.seed)
            if weight > 0:
                from fpl.model.cnn_process import IsolatedSequenceCNNRegressor
                settings = dict(self.cnn_params)
                settings["seed"] = self.seed
                self.cnn_ = IsolatedSequenceCNNRegressor(**settings)
                self.cnn_.fit(X, y)
            self.weight_ = weight
            self.fitted_ = True
            return self
        except BaseException:
            self.close()
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict in original point units without fitting or reading target labels."""
        check_is_fitted(self, "weight_")
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Temporal blend has been closed or is not fitted")
        if self.weight_ == 0:
            result = np.asarray(self.tree_.predict(X), dtype=float)
        elif self.weight_ == 1:
            result = np.asarray(self.cnn_.predict(X), dtype=float)
        else:
            tree = np.asarray(self.tree_.predict(X), dtype=float)
            cnn = np.asarray(self.cnn_.predict(X), dtype=float)
            if any(values.shape != (len(X),) or not np.isfinite(values).all() for values in (tree, cnn)):
                raise ValueError("Temporal blend member returned invalid predictions")
            result = (1 - self.weight_) * tree + self.weight_ * cnn
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError("Temporal blend member returned invalid predictions")
        return result

    def close(self) -> None:
        """Close any retained CNN worker before refitting or disposing this estimator."""
        self.fitted_ = False
        cnn = getattr(self, "cnn_", None)
        if cnn is not None:
            cnn.close()
            del self.cnn_
        if hasattr(self, "tree_"):
            del self.tree_

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
