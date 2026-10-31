"""
Tests for fpl.milp.compare_backtests: the paired block-bootstrap + sign test that puts
an uncertainty interval on realized-points backtest comparisons (audit finding B3).

Synthetic squad_selection frames only - the contract under test is statistical
plumbing (pairing, resampling, CI behaviour), not the MILP itself: identical runs
must give a zero interval, a large constant gap must exclude zero, pure noise must
straddle zero, and mismatched windows must refuse to compare.
"""

import numpy as np
import pandas as pd
import pytest

from fpl.milp import compare_backtests as cb


def _run_df(points_by_gw):
    return pd.DataFrame({"gameweek": list(points_by_gw.keys()),
                         "actual_total_points": list(points_by_gw.values())})


def test_identical_runs_give_zero_interval():
    run = _run_df({gw: 50 + gw for gw in range(153, 184)})
    diffs = cb.paired_gw_differences(run, run.copy())
    r = cb.moving_block_bootstrap_total(diffs, n_boot=500)
    assert r["total_diff"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    s = cb.sign_test(diffs)
    assert s["wins_a"] == 0 and s["wins_b"] == 0 and s["ties"] == len(run)


def test_constant_gap_excludes_zero():
    gws = range(153, 184)
    a = _run_df({gw: 60.0 for gw in gws})
    b = _run_df({gw: 55.0 for gw in gws})
    diffs = cb.paired_gw_differences(a, b)
    r = cb.moving_block_bootstrap_total(diffs, n_boot=500)
    assert r["total_diff"] == pytest.approx(5.0 * len(diffs))
    assert r["ci_low"] > 0.0  # every resample of a constant series sums the same
    assert cb.sign_test(diffs)["p_value"] < 0.001


def test_pure_noise_straddles_zero():
    rng = np.random.default_rng(1)
    gws = list(range(153, 184))
    noise = rng.normal(0, 10, size=len(gws))
    a = _run_df(dict(zip(gws, 50 + noise)))
    b = _run_df({gw: 50.0 for gw in gws})
    diffs = cb.paired_gw_differences(a, b)
    r = cb.moving_block_bootstrap_total(diffs, n_boot=2000)
    assert r["ci_low"] < 0.0 < r["ci_high"]


def test_mismatched_windows_raise():
    a = _run_df({gw: 50 for gw in range(153, 184)})
    b = _run_df({gw: 50 for gw in range(154, 185)})
    with pytest.raises(ValueError, match="windows differ"):
        cb.paired_gw_differences(a, b)


def test_duplicate_gameweeks_raise():
    a = _run_df({gw: 50 for gw in range(153, 160)})
    dup = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        cb.paired_gw_differences(dup, a)


def test_compare_end_to_end(tmp_path):
    gws = range(153, 184)
    a_path, b_path = tmp_path / "a.csv", tmp_path / "b.csv"
    _run_df({gw: 70.0 for gw in gws}).to_csv(a_path, index=False)
    _run_df({gw: 66.0 for gw in gws}).to_csv(b_path, index=False)
    r = cb.compare(str(a_path), str(b_path), n_boot=200)
    assert r["total_a"] == pytest.approx(70.0 * 31)
    assert r["total_b"] == pytest.approx(66.0 * 31)
    assert r["total_diff"] == pytest.approx(4.0 * 31)
    assert r["sign_test"]["wins_a"] == 31


def test_block_bootstrap_wider_than_iid_under_autocorrelation():
    # A strongly positively autocorrelated series: blocks that preserve runs of same-sign
    # values must produce a wider total-sum distribution than iid shuffling, which is the
    # entire reason compare_backtests uses a block bootstrap (squad carryover).
    rng = np.random.default_rng(2)
    n = 40
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.9 * d[t - 1] + rng.normal(0, 1)
    blocked = cb.moving_block_bootstrap_total(d, n_boot=3000, block_len=5, seed=3)
    iid = cb.moving_block_bootstrap_total(d, n_boot=3000, block_len=1, seed=3)
    assert (blocked["ci_high"] - blocked["ci_low"]) > (iid["ci_high"] - iid["ci_low"])


def test_holm_bonferroni_preserves_input_order_and_steps_down():
    # Sorted p-values are .01 and .04.  At alpha=.05 the first clears .025,
    # then the second clears .05; return order must match the unsorted input.
    assert cb.holm_bonferroni([0.04, 0.01]) == [True, True]
    # Once the smallest fails .025, Holm stops even though the larger value
    # would clear its own .05 threshold if incorrectly considered alone.
    assert cb.holm_bonferroni([0.04, 0.03]) == [False, False]


def test_promotion_gate_requires_every_registered_condition():
    standard = {"total_diff": 150.0, "ci_low": 5.0}
    origin = {"total_diff": 20.0, "ci_low": -35.0}
    passed = cb.promotion_gate(standard, origin, [4, 8, 2], holm_pass=True)
    assert passed["promote"] is True

    unstable = cb.promotion_gate(standard, origin, [4, -1, 2], holm_pass=True)
    assert unstable["promote"] is False
    assert unstable["checks"]["seed_direction_stable"] is False

    weak_origin = cb.promotion_gate(standard, {"total_diff": 20, "ci_low": -41},
                                    [4, 8, 2], holm_pass=True)
    assert weak_origin["promote"] is False


def test_sign_test_exposes_exact_one_sided_null_p_value():
    result = cb.sign_test([1.0] * 8 + [-1.0] * 2 + [0.0])
    assert result["wins_a"] == 8
    assert result["ties"] == 1
    assert result["p_a_better"] == pytest.approx(0.0546875)


@pytest.mark.parametrize('column', ['scoring_protocol','points_accounting'])
def test_labels_require_matching_protocol_and_accounting(column):
    a=_run_df({1:50.,2:60.});b=a.copy()
    a[column]='autosubs_v2' if column=='scoring_protocol' else 'net_after_transfer_hits'
    b[column]='legacy_no_autosubs' if column=='scoring_protocol' else 'gross_before_hits'
    with pytest.raises(ValueError,match=f'different {column}'):cb.paired_gw_differences(a,b)
    with pytest.raises(ValueError,match=f'both runs must declare {column}'):cb.paired_gw_differences(a,b.drop(columns=column))


@pytest.mark.parametrize('column', ['scoring_protocol','points_accounting'])
@pytest.mark.parametrize('values', [[None,None],['',''],[' ',' '],['one','two'],['one',None],[1,1]])
def test_each_protocol_label_is_single_nonmissing_nonblank_string(column,values):
    a=_run_df({1:50.,2:60.});a[column]=values
    with pytest.raises(ValueError,match='one nonmissing nonblank string'):cb.paired_gw_differences(a,a.copy())


def test_valid_labels_and_unlabelled_legacy_preserve_differences():
    a=_run_df({1:55.,2:65.});b=_run_df({1:50.,2:60.})
    legacy=cb.paired_gw_differences(a,b)
    for frame in (a,b):
        frame['scoring_protocol']='autosubs_v2';frame['points_accounting']='net_after_transfer_hits'
    pd.testing.assert_series_equal(cb.paired_gw_differences(a,b),legacy)
    assert list(legacy)==[5.,5.]


@pytest.mark.parametrize('column', ['gameweek','actual_total_points'])
@pytest.mark.parametrize('value', [np.nan,np.inf,-np.inf,True,'50'])
def test_finite_real_numeric_columns_required(column,value):
    a=_run_df({1:50.,2:60.})
    a[column]=a[column].astype(object if isinstance(value,(bool,str)) else float)
    a.loc[0,column]=value
    with pytest.raises(ValueError,match='finite numeric'):cb.paired_gw_differences(a,a.copy())


@pytest.mark.parametrize('value',[0.,-1.,1.5])
def test_positive_whole_gameweek_required(value):
    a=_run_df({1:50.,2:60.});a['gameweek']=a.gameweek.astype(float);a.loc[0,'gameweek']=value
    with pytest.raises(ValueError,match='positive integer'):cb.paired_gw_differences(a,a.copy())


def test_empty_frames_missing_and_duplicate_columns_refused():
    a=_run_df({1:50.,2:60.})
    for invalid in (a.iloc[:0],pd.concat([a,a[['gameweek']]],axis=1)):
        with pytest.raises(ValueError,match='nonempty frame with unique'):cb.paired_gw_differences(invalid,a)
    for column in a:
        with pytest.raises(ValueError,match='required columns'):cb.paired_gw_differences(a.drop(columns=column),a)
    a['actual_total_points']=a.actual_total_points.astype('Float64');a.loc[0,'actual_total_points']=pd.NA
    with pytest.raises(ValueError,match='finite numeric'):cb.paired_gw_differences(a,a.copy())


@pytest.mark.parametrize('values',[[],[[1.,2.]],np.array(1.),[np.nan],[np.inf],[-np.inf],[1.,np.nan]])
@pytest.mark.parametrize('function',[cb.sign_test,cb.moving_block_bootstrap_total])
def test_numerical_series_must_be_nonempty_1d_finite(function,values):
    with pytest.raises(ValueError,match='one-dimensional'):function(values)


@pytest.mark.parametrize('name', ['n_boot','block_len','seed'])
@pytest.mark.parametrize('value',[True,False,-1,1.5,'1'])
def test_bootstrap_configuration_integer_guards(name,value):
    with pytest.raises(ValueError,match=name):cb.moving_block_bootstrap_total([1.,-1.],**{name:value})


@pytest.mark.parametrize('name',['n_boot','block_len'])
def test_bootstrap_counts_cannot_be_zero(name):
    with pytest.raises(ValueError,match=name):cb.moving_block_bootstrap_total([1.,-1.],**{name:0})


def test_large_valid_block_still_caps_at_series_length():
    result=cb.moving_block_bootstrap_total([1.,2.],n_boot=3,block_len=99,seed=0)
    assert result['block_len']==2 and result['total_diff']==3.
    assert result['ci_low']==result['ci_high']==3.


@pytest.mark.parametrize('function',[cb.sign_test,cb.moving_block_bootstrap_total])
def test_complex_differences_cannot_silently_discard_imaginary_part(function):
    with pytest.raises(ValueError,match='finite real'):function(np.array([1.+2.j]))
