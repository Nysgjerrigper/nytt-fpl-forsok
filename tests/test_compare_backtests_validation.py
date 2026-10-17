"""Admission and hand-calculated accounting regressions for paired comparisons."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from fpl.milp import compare_backtests as cb


def _run(points=(50.0, 60.0), **columns):
    return pd.DataFrame({"gameweek": [153, 154], "actual_total_points": points, **columns})


def test_transfer_hits_reverse_the_gross_ranking():
    """A earns four more gross points, but its two hits cost eight points."""
    candidate = _run((52, 62), alpha=[1, 1])
    control = _run(alpha=[0, 0])
    gross = cb.paired_gw_differences(candidate, control)
    net = cb.paired_gw_differences(candidate, control, points_col="actual_net_points")
    assert gross.to_dict() == {153: 2, 154: 2}
    assert gross.sum() == 4
    assert net.to_dict() == {153: -2, 154: -2}
    assert net.sum() == -4


def test_declared_net_points_match_derived_net_points():
    candidate = _run((52, 62), alpha=[1, 1], actual_net_points=[48, 58])
    control = _run(alpha=[0, 0], actual_net_points=[50, 60])
    assert cb.paired_gw_differences(
        candidate, control, points_col="actual_net_points"
    ).tolist() == [-2, -2]


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("column", ["gameweek", "actual_total_points"])
def test_required_columns_rejected_on_either_side(side, column):
    a, b = _run(), _run()
    if side == "a":
        a = a.drop(columns=column)
    else:
        b = b.drop(columns=column)
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, b)


@pytest.mark.parametrize("side", ["a", "b"])
def test_empty_frames_rejected_on_either_side(side):
    a, b = _run(), _run()
    if side == "a":
        a = a.iloc[:0]
    else:
        b = b.iloc[:0]
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, b)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 153.5, np.nan, np.inf, None, "invalid"])
def test_invalid_gameweek_rejected(bad):
    a = _run().astype({"gameweek": object})
    a.loc[0, "gameweek"] = bad
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, None, "invalid"])
def test_nonfinite_or_nonnumeric_points_rejected(bad):
    a = _run().astype({"actual_total_points": object})
    a.loc[0, "actual_total_points"] = bad
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run())


def test_duplicate_gameweek_rejected():
    a = _run()
    a["gameweek"] = [153, 153]
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run())


def test_different_gameweek_windows_rejected():
    a = _run()
    a["gameweek"] = [153, 155]
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run())


def test_unsorted_identical_windows_align_by_gameweek():
    candidate = _run((52, 65)).iloc[::-1]
    assert cb.paired_gw_differences(candidate, _run()).to_dict() == {153: 2, 154: 5}


@pytest.mark.parametrize("metric", ["points", "expected_points", "", None])
def test_unknown_points_metric_rejected(metric):
    with pytest.raises(ValueError):
        cb.paired_gw_differences(_run(), _run(), points_col=metric)


def test_net_comparison_requires_hit_counts_even_with_declared_net():
    a = _run(actual_net_points=[50, 60])
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, a, points_col="actual_net_points")


def test_net_comparison_requires_gross_points():
    a = _run(alpha=[0, 0], actual_net_points=[50, 60]).drop(columns="actual_total_points")
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, a, points_col="actual_net_points")


@pytest.mark.parametrize("metric", ["actual_total_points", "actual_net_points"])
@pytest.mark.parametrize("bad", [True, -1, 0.5, np.nan, np.inf, None, "invalid"])
def test_hit_counts_are_nonnegative_integers_when_present(metric, bad):
    a = _run(alpha=pd.Series([bad, 0], dtype=object))
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run(alpha=[0, 0]), points_col=metric)


@pytest.mark.parametrize("metric", ["actual_total_points", "actual_net_points"])
@pytest.mark.parametrize("net", [[50, 60], [np.nan, 56], [np.inf, 56]])
def test_declared_net_must_be_finite_and_agree_with_costs(metric, net):
    a = _run(alpha=[1, 1], actual_net_points=net)
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run(alpha=[0, 0]), points_col=metric)


def test_legacy_gross_comparison_does_not_require_hit_counts():
    assert cb.paired_gw_differences(_run((51, 62)), _run()).tolist() == [1, 2]


@pytest.mark.parametrize("protocol", ["autosubs_v2", "legacy_no_autosubs"])
def test_matching_declared_protocol_is_admitted(protocol):
    a = _run(scoring_protocol=[protocol, protocol])
    assert cb.paired_gw_differences(a, a).tolist() == [0, 0]


def test_mixed_scoring_protocol_within_a_run_rejected():
    a = _run(scoring_protocol=["autosubs_v2", "legacy_no_autosubs"])
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run())


def test_different_declared_scoring_protocols_rejected():
    a = _run(scoring_protocol=["autosubs_v2"] * 2)
    b = _run(scoring_protocol=["legacy_no_autosubs"] * 2)
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, b)


def test_one_sided_protocol_declaration_rejected():
    a = _run(scoring_protocol=["autosubs_v2"] * 2)
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, _run())


@pytest.mark.parametrize("protocol", ["unknown", "autosubs_v999"])
def test_unknown_declared_scoring_protocol_rejected(protocol):
    a = _run(scoring_protocol=[protocol] * 2)
    with pytest.raises(ValueError):
        cb.paired_gw_differences(a, a)


@pytest.mark.parametrize("function", [cb.moving_block_bootstrap_total, cb.sign_test])
@pytest.mark.parametrize("diffs", [[], [np.nan], [np.inf], [-np.inf], [[1, 2]], 1.0, ["invalid"]])
def test_statistical_helpers_reject_invalid_difference_vectors(function, diffs):
    with pytest.raises(ValueError):
        function(diffs)


@pytest.mark.parametrize("parameter", ["n_boot", "block_len"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, np.nan, np.inf, None, "3"])
def test_bootstrap_count_parameters_require_positive_integers(parameter, value):
    with pytest.raises(ValueError):
        cb.moving_block_bootstrap_total([1, 2, 3], **{parameter: value})


@pytest.mark.parametrize("seed", [True, -1, 0.5, np.nan, np.inf, None, "0"])
def test_bootstrap_seed_requires_nonnegative_integer(seed):
    with pytest.raises(ValueError):
        cb.moving_block_bootstrap_total([1, 2, 3], seed=seed)


def test_bootstrap_caps_block_length_to_window_and_is_reproducible():
    first = cb.moving_block_bootstrap_total([1, -2], n_boot=20, block_len=100, seed=4)
    second = cb.moving_block_bootstrap_total([1, -2], n_boot=20, block_len=100, seed=4)
    assert first == second
    assert first["block_len"] == 2
    assert first["total_diff"] == first["ci_low"] == first["ci_high"] == -1


def test_compare_selects_net_metric_and_preserves_unverified_input_evidence(tmp_path):
    path_a, path_b = tmp_path / "a.csv", tmp_path / "b.csv"
    _run((52, 62), alpha=[1, 1]).to_csv(path_a, index=False)
    _run(alpha=[0, 0]).to_csv(path_b, index=False)
    report = cb.compare(path_a, path_b, n_boot=20, points_col="actual_net_points")
    assert report["total_a"] == 106
    assert report["total_b"] == 110
    assert report["total_diff"] == -4
    assert report["metric"] == "actual_net_points"
    assert report["gameweeks"] == [153, 154]
    assert report["evidence_status"] == "UNVERIFIED"
    expected_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in (path_a, path_b)}
    assert set(report["input_sha256"].values()) == expected_hashes
    assert {"boot_mean", "ci_low", "ci_high", "p_a_better", "n_gws", "block_len", "sign_test"} <= report.keys()
