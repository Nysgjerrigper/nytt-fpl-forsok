"""
Uncertainty on realized-points backtest comparisons.

Every modeling decision in this project is ultimately judged by the difference in
`actual_total_points` between two MILP backtest runs over the same gameweek window
(fpl/milp/optimize.py's squad_selection CSVs). Those verdicts have so far been point
estimates - "2107 vs 2059, buckets lose by 48" - with no sense of whether 48 points
over 31 gameweeks is signal or noise (audit finding B3). This module puts an interval
on that difference so future +/-50-point verdicts are calibrated:

    python -m fpl.milp.compare_backtests runA.csv runB.csv

- **Paired**: both runs saw the same gameweeks, so we analyze the per-GW difference
  d_t = points_A(t) - points_B(t), which nets out common gameweek difficulty (a
  blank-heavy GW hurts both configurations alike).
- **Moving-block bootstrap** (Kunsch 1989) rather than iid resampling: the MILP carries
  its squad across gameweeks, so consecutive d_t are autocorrelated - a bad transfer
  poisons several following weeks. Resampling contiguous blocks preserves that local
  dependence; an iid bootstrap would understate the variance of the total.
- **Sign test** as a distribution-free second opinion: just "how often did A beat B",
  with a binomial p-value. Robust to the heavy-tailed per-GW hauls that can dominate
  the bootstrap mean.

Neither test proves a config is better - one 31-GW window is one draw - but a CI that
straddles zero says "do not conclude anything from this gap", which is exactly the
guardrail the standing decision rule (realized points over MASE) was missing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

POINTS_COL = "actual_total_points"
GW_COL = "gameweek"


def paired_gw_differences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.Series:
    """Per-gameweek realized-points difference (A - B), indexed by gameweek.

    Both frames must be squad_selection CSVs (one row per gameweek) covering the SAME
    gameweek set - a paired analysis over mismatched windows would be comparing
    different seasons' difficulty, so that raises instead of silently intersecting.
    """
    for name, df in (("A", df_a), ("B", df_b)):
        if POINTS_COL not in df.columns:
            raise ValueError(f"run {name} has no '{POINTS_COL}' column - was the backtest "
                             "run on predictions without actuals (a live run)?")
        if df[GW_COL].duplicated().any():
            raise ValueError(f"run {name} has duplicate gameweek rows")
    a = df_a.set_index(GW_COL)[POINTS_COL].sort_index()
    b = df_b.set_index(GW_COL)[POINTS_COL].sort_index()
    if not a.index.equals(b.index):
        only_a = sorted(set(a.index) - set(b.index))
        only_b = sorted(set(b.index) - set(a.index))
        raise ValueError(f"gameweek windows differ (only in A: {only_a}, only in B: {only_b}); "
                         "a paired comparison needs identical windows")
    return a - b


def moving_block_bootstrap_total(diffs, n_boot: int = 10000, block_len: int = 3,
                                 seed: int = 0) -> dict:
    """Bootstrap distribution of the TOTAL points difference over the window.

    Resamples the GW-ordered difference series in contiguous blocks of `block_len`
    (default 3 ~ n^(1/3) for a 31-GW window, the standard block-length rule of thumb)
    until the original length is reached, sums each replicate, and returns the mean,
    a 95% percentile CI, and P(total > 0). With block_len=1 this degrades to the iid
    bootstrap, which assumes away the squad-carryover autocorrelation - keep >= 2
    unless you know the runs share no state across gameweeks.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        raise ValueError("empty difference series")
    block_len = max(1, min(block_len, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    # Moving blocks: all n - block_len + 1 overlapping windows are candidates.
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    totals = d[idx].sum(axis=1)
    lo, hi = np.percentile(totals, [2.5, 97.5])
    return {
        "total_diff": float(d.sum()),
        "boot_mean": float(totals.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_a_better": float(np.mean(totals > 0)),
        "n_gws": n,
        "block_len": block_len,
    }


def sign_test(diffs) -> dict:
    """Exact paired sign tests on per-GW wins, with ties dropped.

    ``p_a_better`` is the valid one-sided p-value for the centered null that A
    is no more likely to win a gameweek than B.  It is suitable for the Holm
    family in the promotion protocol; a bootstrap probability is not a null
    p-value and must never be used there.
    """
    d = np.asarray(diffs, dtype=float)
    wins_a = int(np.sum(d > 0))
    wins_b = int(np.sum(d < 0))
    n_eff = wins_a + wins_b
    if n_eff == 0:
        two_sided = one_sided = 1.0
    else:
        result = binomtest(wins_a, n_eff, 0.5)
        two_sided = float(result.pvalue)
        one_sided = float(binomtest(wins_a, n_eff, 0.5, alternative="greater").pvalue)
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": int(np.sum(d == 0)),
            "p_value": two_sided, "p_a_better": one_sided}


def holm_bonferroni(p_values, alpha: float = 0.05) -> list[bool]:
    """Family-wise Holm decisions, returned in the caller's original order.

    The MoE promotion protocol admits at most two final candidates (the hard
    position map and the optional MID gate).  Testing both against CatBoost
    without a multiplicity correction would make an already reused evaluation
    window easier to overfit.  Holm is uniformly at least as powerful as plain
    Bonferroni while preserving the same family-wise error guarantee.
    """
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must all be finite and in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    order = np.argsort(values)
    rejected = np.zeros(len(values), dtype=bool)
    # Holm is step-down: once one ordered hypothesis fails, every larger
    # p-value must fail too even if it happens to clear its looser threshold.
    for rank, idx in enumerate(order):
        threshold = alpha / (len(values) - rank)
        if values[idx] > threshold:
            break
        rejected[idx] = True
    return rejected.tolist()


def promotion_gate(standard_report: dict, origin_report: dict, seed_diffs,
                   holm_pass: bool, origin_floor: float = -40.0) -> dict:
    """Evaluate the pre-registered MoE production-promotion conditions.

    ``standard_report`` and ``origin_report`` are outputs from :func:`compare`
    with the candidate as run A and CatBoost as run B.  ``seed_diffs`` contains
    the realized-point differences for the positional winner refits at seeds
    0/1/2.  The function intentionally returns every condition as evidence
    rather than only a boolean, so a failed experiment says exactly which gate
    stopped it and can be logged without interpretation drift.
    """
    seed_diffs = [float(x) for x in seed_diffs]
    checks = {
        "standard_points_better": float(standard_report["total_diff"]) > 0.0,
        "origin_points_better": float(origin_report["total_diff"]) > 0.0,
        "standard_ci_excludes_zero": float(standard_report["ci_low"]) > 0.0,
        "origin_ci_not_materially_negative": float(origin_report["ci_low"]) >= float(origin_floor),
        "seed_direction_stable": len(seed_diffs) == 3 and all(x > 0.0 for x in seed_diffs),
        "holm_pass": bool(holm_pass),
    }
    return {"promote": all(checks.values()), "checks": checks,
            "seed_diffs": seed_diffs, "origin_floor": float(origin_floor)}


def compare(path_a: str, path_b: str, n_boot: int = 10000, block_len: int = 3,
            seed: int = 0) -> dict:
    """Full comparison of two squad_selection CSVs; returns the combined report dict."""
    df_a, df_b = pd.read_csv(path_a), pd.read_csv(path_b)
    diffs = paired_gw_differences(df_a, df_b)
    report = moving_block_bootstrap_total(diffs, n_boot=n_boot, block_len=block_len, seed=seed)
    report["sign_test"] = sign_test(diffs)
    report["total_a"] = float(df_a[POINTS_COL].sum())
    report["total_b"] = float(df_b[POINTS_COL].sum())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Paired block-bootstrap CI + sign test on the realized-points difference "
                    "between two MILP backtest runs (squad_selection CSVs) over the same window.")
    parser.add_argument("run_a", help="squad_selection CSV of configuration A")
    parser.add_argument("run_b", help="squad_selection CSV of configuration B")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--block-len", type=int, default=3,
                        help="Bootstrap block length in gameweeks (default 3; >=2 to respect "
                             "squad-carryover autocorrelation).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    r = compare(args.run_a, args.run_b, n_boot=args.n_boot, block_len=args.block_len, seed=args.seed)
    s = r["sign_test"]
    print(f"A: {Path(args.run_a).name}  total {r['total_a']:.0f}")
    print(f"B: {Path(args.run_b).name}  total {r['total_b']:.0f}")
    print(f"\nTotal difference (A - B) over {r['n_gws']} GWs: {r['total_diff']:+.0f} points")
    print(f"95% block-bootstrap CI (block={r['block_len']}, n={args.n_boot}): "
          f"[{r['ci_low']:+.0f}, {r['ci_high']:+.0f}]   P(A better) = {r['p_a_better']:.3f}")
    print(f"Sign test: A won {s['wins_a']} GWs, B won {s['wins_b']} ({s['ties']} ties), "
          f"two-sided p = {s['p_value']:.3f}")
    if r["ci_low"] <= 0.0 <= r["ci_high"]:
        print("\nVerdict: the CI straddles zero - this window does NOT distinguish the two "
              "configurations; do not promote/demote on this gap alone.")
    else:
        better = "A" if r["total_diff"] > 0 else "B"
        print(f"\nVerdict: the CI excludes zero - configuration {better} is credibly better "
              "on this window (still one window; see RESEARCH_LOG.md protocol).")


if __name__ == "__main__":
    main()
