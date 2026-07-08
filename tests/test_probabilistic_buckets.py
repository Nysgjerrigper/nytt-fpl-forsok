import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model import probabilistic_buckets as buckets


def test_points_to_buckets_maps_fpl_outcomes():
    y = np.array([-1, 0, 1, 2, 3, 5, 6, 9, 10, 15])
    assert buckets.points_to_buckets(y).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_points_to_buckets_supports_finer_scheme():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13])
    scheme = buckets.get_bucket_scheme("fine8")
    assert buckets.points_to_buckets(y, scheme).tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7]


def test_expected_points_from_distribution_uses_training_bucket_values():
    proba = np.array([[0.2, 0.3, 0.0, 0.0, 0.5]])
    values = np.array([0.0, 2.0, 4.0, 8.0, 12.0])
    assert np.isclose(buckets.expected_points_from_proba(proba, values)[0], 6.6)


def test_normalize_proba_repairs_roundoff():
    proba = buckets.normalize_proba(np.array([[0.2, 0.2, 0.2, 0.2, 0.3]]))
    assert np.isclose(proba.sum(axis=1)[0], 1.0)


def test_multiclass_brier_zero_for_perfect_probabilities():
    y = np.array([0, 2, 4])
    proba = np.eye(buckets.N_BUCKETS)[y]
    assert buckets.multiclass_brier(y, proba) == 0.0
