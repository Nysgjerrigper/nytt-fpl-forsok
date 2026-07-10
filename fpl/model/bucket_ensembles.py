"""
Ensemble analysis over saved bucket walk-forward predictions.

`fpl.model.probabilistic_buckets --walk-forward --save-predictions <csv>` dumps
one row per (scheme, model, player, gameweek) with expected points and the full
bucket distribution. This module answers the combination questions as pure
post-processing on that CSV - no model refitting:

1. blend      - w * bucket E[pts] + (1-w) * regression E[pts], with the weight
                chosen on the FIRST half of the window and scored on the second
                (an in-window weight search would flatter the blend).
2. captaincy  - rank captain candidates by E[pts] from one scheme tilted by a
                haul probability from another (e.g. fine8 E[pts] x binary9
                P(>9)): the "binomial for captaincy, fine buckets for
                forecasting" hybrid.
3. average    - equal-weight self-ensemble of E[pts] across schemes/models.

All strategies are FIXED formulas (no fitted parameters) except the blend
weight, so full-window scores are honest for 2 and 3; metrics are also reported
on the second half alone as a stability check against strategy-picking noise.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl.model.metrics import bias, mae, rmse, spearman_by_group, top1_capture, total_calibration
from fpl.model.probabilistic_buckets import REGRESSION_BASELINE, print_table

# A player can have two fixtures in one gameweek (double gameweeks), so
# (player_id, GW_global, position) is not unique; the cumcount disambiguates
# fixture rows, which appear in identical order for every source because all
# prediction frames are built from the same test_df slice.
KEY = ["player_id", "GW_global", "position"]


def load_sources(path):
    """Read a saved prediction CSV and return (ev_wide, prob_wide, gws).

    ev_wide: one row per player-fixture with an E[pts] column per source
    ("<scheme>/<model>", the regression baseline collapsed to "regression").
    prob_wide: same shape holding each source's P(top bucket) - for a binary
    scheme that is P(points > threshold); for multiclass schemes the p_haul
    (P(>=10)) column is used instead, NaN where the scheme can't express it.
    """
    df = pd.read_csv(path, low_memory=False)
    df["source"] = df["scheme"].astype(str) + "/" + df["model"].astype(str)
    df.loc[df["model"] == REGRESSION_BASELINE, "source"] = "regression"
    df["dup"] = df.groupby(["source"] + KEY).cumcount()
    index_cols = KEY + ["dup", "actual_points"]

    # Tilt probability: top-bucket probability for binary schemes, p_haul for
    # multiclass ones (the regression has neither and stays NaN).
    is_binary = df["n_buckets"] == 2
    df["tilt_prob"] = np.where(is_binary, df.get("p_bucket_1", np.nan), df["p_haul"])

    ev_wide = df.pivot_table(index=index_cols, columns="source", values="expected_points").reset_index()
    prob_wide = df.pivot_table(index=index_cols, columns="source", values="tilt_prob").reset_index()
    gws = sorted(df["GW_global"].unique())
    return ev_wide, prob_wide, gws


def score_ev(frame, pred_col):
    """Point-forecast quality of one E[pts] column: accuracy, calibration,
    per-(position, GW) ranking, and captaincy top-1 capture on MID/FWD."""
    sub = frame.dropna(subset=[pred_col])
    y = sub["actual_points"].to_numpy()
    p = sub[pred_col].to_numpy()
    groups = sub["position"].astype(str) + "_" + sub["GW_global"].astype(str)
    pool = sub[sub["position"].isin(["MID", "FWD"])]
    return {
        "ev_mae": mae(y, p),
        "ev_rmse": rmse(y, p),
        "bias": bias(y, p),
        "total_calibration": total_calibration(y, p),
        "spearman_pos_gw": spearman_by_group(y, p, groups),
        "cap_ev": top1_capture(pool, "GW_global", "actual_points", pred_col),
    }


def blend_analysis(ev_wide, bucket_col, gws, reg_col="regression"):
    """Honest weight search for w*bucket + (1-w)*regression.

    The window is split in half by gameweek: every candidate weight is scored
    on the first half, the weight with the best fit-half Spearman is then
    scored on the SECOND half only, next to the pure endpoints, so the reported
    number never benefits from picking the weight on the data it is judged on.
    """
    fit_gws = set(gws[: len(gws) // 2])
    frame = ev_wide.dropna(subset=[bucket_col, reg_col]).copy()
    fit = frame[frame["GW_global"].isin(fit_gws)].copy()
    heldout = frame[~frame["GW_global"].isin(fit_gws)].copy()

    weights = np.round(np.arange(0.0, 1.01, 0.1), 2)
    fit_scores = {}
    for w in weights:
        col = f"__blend_{w}"
        fit[col] = w * fit[bucket_col] + (1 - w) * fit[reg_col]
        fit_scores[w] = score_ev(fit, col)["spearman_pos_gw"]
    best_w = max(fit_scores, key=fit_scores.get)
    print(f"\nBlend fit-half spearman by weight ({bucket_col} vs {reg_col}):")
    print("  " + "  ".join(f"w={w:.1f}:{fit_scores[w]:.4f}" for w in weights))

    rows = []
    for label, w in [(f"pure {reg_col}", 0.0), (f"chosen w={best_w:.1f}", best_w), (f"pure {bucket_col}", 1.0)]:
        col = f"__eval_{w}"
        heldout[col] = w * heldout[bucket_col] + (1 - w) * heldout[reg_col]
        rows.append({"strategy": label, **score_ev(heldout, col)})
    print_table(
        f"Blend evaluated on held-out second half (GW>{max(fit_gws)})",
        rows,
        ["strategy", "ev_mae", "ev_rmse", "bias", "total_calibration", "spearman_pos_gw", "cap_ev"],
    )
    return best_w, rows


def captaincy_analysis(ev_wide, prob_wide, base_cols, tilt_cols, gws):
    """Compare captain-pick strategies on the MID/FWD pool.

    Strategies: each base E[pts] alone, each tilt probability alone, and every
    base x (1 + tilt) product. Reported as top-1 capture (share of gameweeks
    where the picked player lands in the actual top scorers) on the full
    window and on the second half alone as a stability check.
    """
    missing = [c for c in base_cols if c not in ev_wide.columns] + [
        c for c in tilt_cols if c not in prob_wide.columns
    ]
    if missing:
        print(f"Captaincy: skipping missing sources {missing}")
        base_cols = [c for c in base_cols if c in ev_wide.columns]
        tilt_cols = [c for c in tilt_cols if c in prob_wide.columns]
    merged = ev_wide.merge(
        prob_wide[KEY + ["dup"] + tilt_cols],
        on=KEY + ["dup"],
        how="left",
        suffixes=("", "__p"),
    )
    pool = merged[merged["position"].isin(["MID", "FWD"])].copy()
    half_gws = set(gws[len(gws) // 2:])

    strategies = {}
    for base in base_cols:
        strategies[f"E[{base}]"] = pool[base]
        for tilt in tilt_cols:
            tcol = f"{tilt}__p" if f"{tilt}__p" in pool.columns else tilt
            strategies[f"E[{base}]*(1+{tilt})"] = pool[base] * (1.0 + pool[tcol])
    for tilt in tilt_cols:
        tcol = tilt if tilt in pool.columns else f"{tilt}__p"
        strategies[f"{tilt} alone"] = pool[tcol]

    rows = []
    for name, series in strategies.items():
        pool["__rank"] = series
        sub = pool.dropna(subset=["__rank"])
        second = sub[sub["GW_global"].isin(half_gws)]
        rows.append({
            "strategy": name,
            "cap_full": top1_capture(sub, "GW_global", "actual_points", "__rank"),
            "cap_2nd_half": top1_capture(second, "GW_global", "actual_points", "__rank"),
        })
    rows.sort(key=lambda r: -(r["cap_full"] if np.isfinite(r["cap_full"]) else -1))
    print_table("Captaincy strategies (MID/FWD top-1 capture)", rows, ["strategy", "cap_full", "cap_2nd_half"])
    return rows


def average_analysis(ev_wide, source_groups, gws):
    """Equal-weight E[pts] self-ensembles, scored on the full window with the
    second half broken out as a stability check."""
    half_gws = set(gws[len(gws) // 2:])
    rows = []
    for name, cols in source_groups.items():
        missing = [c for c in cols if c not in ev_wide.columns]
        if missing:
            print(f"Skipping average '{name}': missing sources {missing}")
            continue
        col = f"__avg_{name}"
        ev_wide[col] = ev_wide[cols].mean(axis=1)
        full = score_ev(ev_wide, col)
        second = score_ev(ev_wide[ev_wide["GW_global"].isin(half_gws)], col)
        rows.append({
            "strategy": name,
            **full,
            "spearman_2nd_half": second["spearman_pos_gw"],
            "cap_ev_2nd_half": second["cap_ev"],
        })
    print_table(
        "Equal-weight E[pts] ensembles (full window)",
        rows,
        ["strategy", "ev_mae", "ev_rmse", "bias", "total_calibration",
         "spearman_pos_gw", "cap_ev", "spearman_2nd_half", "cap_ev_2nd_half"],
    )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ensemble analysis over saved bucket walk-forward predictions.")
    parser.add_argument("--predictions", required=True, help="CSV from --walk-forward --save-predictions.")
    parser.add_argument("--bucket-source", default=None,
                        help="E[pts] source for the regression blend, e.g. 'fine8/catboost_bucket'.")
    parser.add_argument("--captaincy-bases", nargs="+", default=None,
                        help="E[pts] sources to captain from, e.g. fine8/catboost_bucket regression.")
    parser.add_argument("--captaincy-tilts", nargs="+", default=None,
                        help="Probability sources for the haul tilt, e.g. binary9/catboost_bucket.")
    parser.add_argument("--average", nargs="+", default=None, action="append",
                        help="Sources to average into one E[pts]; repeatable for several groups.")
    args = parser.parse_args(argv)

    ev_wide, prob_wide, gws = load_sources(args.predictions)
    available = [c for c in ev_wide.columns if c not in KEY + ["dup", "actual_points"]]
    print(f"Sources: {available}")
    print(f"Gameweeks: {gws[0]}-{gws[-1]} ({len(gws)})")

    if args.bucket_source:
        blend_analysis(ev_wide, args.bucket_source, gws)
    if args.captaincy_bases and args.captaincy_tilts:
        captaincy_analysis(ev_wide, prob_wide, args.captaincy_bases, args.captaincy_tilts, gws)
    if args.average:
        groups = {f"avg{i}({'+'.join(cols)})": cols for i, cols in enumerate(args.average, 1)}
        average_analysis(ev_wide, groups, gws)


if __name__ == "__main__":
    main()
