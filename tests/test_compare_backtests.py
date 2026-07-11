"""
Tests for fpl.milp.compare_backtests: the paired block-bootstrap + sign test that puts
an uncertainty interval on realized-points backtest comparisons (audit finding B3).

Synthetic squad_selection frames only - the contract under test is statistical
plumbing (pairing, resampling, CI behaviour), not the MILP itself: identical runs
must give a zero interval, a large constant gap must exclude zero, pure noise must
straddle zero, and mismatched windows must refuse to compare.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
