"""Exact signed CQR arithmetic and interval diagnostics, without model fitting."""
import numpy as np
import pytest

from fpl.model.interval_calibration import apply_adjustment, conformal_adjustment, interval_metrics


def test_exact_order_statistic_without_interpolation():
    # Scores [-2, -1, 0, 1], n=4, alpha=.4 => ceil(5*.6)=3 => score 0.
    y = [2, 1, 0, -1]
    assert conformal_adjustment(y, [0] * 4, [10] * 4, alpha=.4) == 0
    assert conformal_adjustment(y, [0] * 4, [10] * 4, alpha=.2) == 1


def test_negative_adjustment_shrinks_intervals():
    adjustment = conformal_adjustment([5] * 4, [0] * 4, [10] * 4)
    assert adjustment == -5
    lower, upper = apply_adjustment([0, 0], [5, 5], [10, 10], adjustment)
    np.testing.assert_array_equal(lower, [5, 5])
    np.testing.assert_array_equal(upper, [5, 5])


def test_median_bracketing_prevents_shrinkage_crossing():
    lower, upper = apply_adjustment([0, 0], [1, 9], [10, 10], -8)
    np.testing.assert_array_equal(lower, [1, 8])
    np.testing.assert_array_equal(upper, [2, 9])
    assert np.all(lower <= upper)


def test_positive_adjustment_expands_and_does_not_mutate_inputs():
    original_lower, original_upper = np.array([1., 2.]), np.array([3., 4.])
    lower, upper = apply_adjustment(original_lower, [2, 3], original_upper, 1.5)
    np.testing.assert_array_equal(lower, [-.5, .5])
    np.testing.assert_array_equal(upper, [4.5, 5.5])
    np.testing.assert_array_equal(original_lower, [1, 2])
    np.testing.assert_array_equal(original_upper, [3, 4])


def test_handcomputed_inclusive_coverage_and_proper_score():
    # Four width-2 intervals: two boundary hits and two distance-1 misses.
    # alpha=.2 => scores 2,2,12,12, mean7; lower/upper misses each1/4.
    result = interval_metrics([0, 2, -1, 3], [0] * 4, [2] * 4)
    assert result == {"coverage": .5, "mean_width": 2., "interval_score": 7.,
                      "lower_miss_rate": .25, "upper_miss_rate": .25}


def test_zero_width_boundary_is_covered():
    assert interval_metrics([2], [2], [2]) == {
        "coverage": 1., "mean_width": 0., "interval_score": 0.,
        "lower_miss_rate": 0., "upper_miss_rate": 0.}


@pytest.mark.parametrize("function", [conformal_adjustment, interval_metrics])
@pytest.mark.parametrize("inputs", [([], [], []), ([[1]], [[0]], [[2]]),
                                  ([1, 2], [0], [3, 3]), ([np.nan], [0], [2]),
                                  ([1], [-np.inf], [2]), ([1], [0], [np.inf]),
                                  ([1], [2], [0])])
def test_invalid_interval_inputs(function, inputs):
    with pytest.raises(ValueError):
        function(*inputs)


@pytest.mark.parametrize("function", [conformal_adjustment, interval_metrics])
@pytest.mark.parametrize("alpha", [0, 1, -.1, 1.1, np.nan, np.inf])
def test_invalid_alpha(function, alpha):
    with pytest.raises(ValueError, match="alpha"):
        function([1] * 4, [0] * 4, [2] * 4, alpha=alpha)


def test_insufficient_sample_refused():
    with pytest.raises(ValueError, match="insufficient"):
        conformal_adjustment([1] * 3, [0] * 3, [2] * 3)


@pytest.mark.parametrize("inputs", [([], [], []), ([[0]], [[1]], [[2]]),
                                  ([0, 0], [1], [2, 2]), ([np.nan], [1], [2]),
                                  ([0], [np.inf], [2]), ([2], [1], [0]),
                                  ([2], [1], [3]), ([0], [3], [2])])
def test_invalid_adjustment_bounds(inputs):
    with pytest.raises(ValueError):
        apply_adjustment(*inputs, 0)


@pytest.mark.parametrize("adjustment", [np.nan, np.inf, -np.inf, [1]])
def test_nonfinite_or_nonscalar_adjustment_rejected(adjustment):
    with pytest.raises(ValueError, match="adjustment"):
        apply_adjustment([0], [1], [2], adjustment)
