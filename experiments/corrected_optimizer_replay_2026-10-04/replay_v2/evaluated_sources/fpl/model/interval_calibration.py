"""Numerical signed CQR adjustments and interval diagnostics for research.

The signed nonconformity score and finite-sample order statistic follow Romano,
Patterson and Candes (2019), https://arxiv.org/abs/1905.03222. FPL observations
are temporally dependent; these calculations do not establish exchangeability
or the coverage guarantees that require it. Calibration data admission and
chronological separation are responsibilities of the calling research runner.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _arrays(*values: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Require equal, nonempty one-dimensional vectors with finite values."""
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if any(array.ndim != 1 or array.size == 0 for array in arrays):
        raise ValueError("interval inputs must be nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("interval inputs must have identical shapes")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("interval inputs must contain only finite values")
    return arrays


def _validate_alpha(alpha: float) -> None:
    """Require a finite miscoverage probability strictly between zero and one."""
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")


def _validate_bounds(lower: NDArray[np.float64], upper: NDArray[np.float64]) -> None:
    """Reject inverted intervals rather than repairing calibration inputs."""
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")


def conformal_adjustment(
    y: ArrayLike, lower: ArrayLike, upper: ArrayLike, *, alpha: float = 0.2,
) -> float:
    """Return the exact finite-sample signed CQR score order statistic.

    Scores are ``max(lower-y, y-upper)``. Select the one-based order statistic
    ``ceil((n+1)*(1-alpha))`` without interpolation, refusing an insufficient
    sample when that rank exceeds ``n``. Negative scores permit interval shrinkage.
    This formula alone gives no exchangeability guarantee for temporal FPL data.
    """
    y_array, lower_array, upper_array = _arrays(y, lower, upper)
    _validate_alpha(alpha)
    _validate_bounds(lower_array, upper_array)
    n = y_array.size
    k = ceil((n + 1) * (1 - alpha))
    if k > n:
        raise ValueError("insufficient calibration sample for the requested alpha")
    scores = np.maximum(lower_array - y_array, y_array - upper_array)
    return float(np.sort(scores)[k - 1])


def apply_adjustment(
    lower: ArrayLike, median: ArrayLike, upper: ArrayLike, adjustment: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expand or shrink bounds, conservatively retaining median bracketing.

    Return ``min(lower-adjustment, median)`` and
    ``max(upper+adjustment, median)``. Bracketing the median is a conservative
    modification to signed CQR that prevents crossed bounds after shrinkage;
    it does not establish exchangeability coverage guarantees for temporal data.
    """
    lower_array, median_array, upper_array = _arrays(lower, median, upper)
    _validate_bounds(lower_array, upper_array)
    if np.any(lower_array > median_array) or np.any(median_array > upper_array):
        raise ValueError("median must lie between lower and upper bounds")
    if not np.isscalar(adjustment) or not np.isfinite(adjustment):
        raise ValueError("adjustment must be a finite scalar")
    return (np.minimum(lower_array - adjustment, median_array),
            np.maximum(upper_array + adjustment, median_array))


def interval_metrics(
    y: ArrayLike, lower: ArrayLike, upper: ArrayLike, *, alpha: float = 0.2,
) -> dict[str, float]:
    """Return inclusive coverage, width, interval score and directional miss rates.

    The interval score is width plus ``2/alpha`` times the distance outside the
    interval. A value exactly on either boundary counts as covered and incurs no
    miss penalty. These empirical diagnostics make no temporal coverage guarantee.
    """
    y_array, lower_array, upper_array = _arrays(y, lower, upper)
    _validate_alpha(alpha)
    _validate_bounds(lower_array, upper_array)
    below, above = y_array < lower_array, y_array > upper_array
    width = upper_array - lower_array
    score = (width + 2 / alpha * (lower_array - y_array) * below
             + 2 / alpha * (y_array - upper_array) * above)
    return {"coverage": float(np.mean(~(below | above))),
            "mean_width": float(np.mean(width)),
            "interval_score": float(np.mean(score)),
            "lower_miss_rate": float(np.mean(below)),
            "upper_miss_rate": float(np.mean(above))}
