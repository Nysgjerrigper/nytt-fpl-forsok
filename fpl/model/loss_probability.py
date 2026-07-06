"""
EXPLORATORY: reframe FPL forecasting as a PROBABILITY-OF-LOSS problem.

This is a parallel, experimental view (like fpl.model.probabilistic) - it does NOT
feed the squad optimizer, which still consumes the point-forecast ensemble's single
number. It exists to answer one question: does modelling the *probability and magnitude
of bad/good outcomes* add anything the mean point-forecast throws away?

The finance analogy the whole idea rests on
------------------------------------------------------------------------------------
A finance MSc doesn't evaluate an asset by its expected return alone - two assets with
the same E[return] are not the same decision if one can halve your capital. Downside-risk
theory formalises this: Roy's (1952) safety-first criterion picks the portfolio that
minimises P(return < disaster level); Value-at-Risk and Expected Shortfall/CVaR quantify
the loss tail; the Sortino ratio penalises only downside deviation. All of these start
from a *distribution of outcomes* and read a *probability of loss* off it, rather than
collapsing everything to a mean.

FPL points are an almost pathological case for this, empirically (whole dataset, 162,981
player-gameweeks):
  * 61% of rows are exactly 0 points, and 59% are players who didn't even take the pitch.
  * Among players who DID play, 68% still returned <= 2 points.
  * Only 1.7% of rows are "hauls" (>= 10 pts) - the rare event captaincy lives or dies on.
So the base case is LOSS (a blank), and the thing that wins leagues is a rare right-tail
event. That is exactly the shape - frequent small losses, rare large gains - where an
expected-value forecast is least informative and a probability-of-loss / probability-of-haul
framing should carry independent signal.

What this module models
------------------------------------------------------------------------------------
Two binary events per position, the "loss" and the "win" bracketing the mean:
  * BLANK  (total_points <= 2): the downside event. P(blank) is a direct probability of
    loss - high for boom-or-bust and rotation-risk players the mean forecast smooths over.
  * HAUL   (total_points >= 10): the captaincy upside event. P(haul) is a probability of
    the right-tail gain, i.e. the quantity a variance-blind mean forecast cannot see.

Both are LightGBM binary classifiers reusing the SAME fpl.features inputs and the SAME
GW153-183 static split as fpl.model.train, so results are directly comparable.

How it is scored (accuracy is useless here, on purpose)
------------------------------------------------------------------------------------
The blank base rate is ~86%, so "always predict blank" is 86% accurate and tells you
nothing - the intermittent-series trap that motivated MASE over MAE, now in classification
form. We use PROPER SCORING RULES and ranking/calibration instead:
  * Brier score + log-loss: proper scoring rules the probabilities are judged by.
  * ROC-AUC: does the probability RANK players correctly (all decisions are comparisons).
  * Expected calibration error: are the probabilities honest (does "30%" happen ~30% of
    the time) - required if a downstream safety-first optimiser is ever to trust them.
  * Base rate: the floor every number above must beat to be worth anything.

Does it change decisions? (the point of the whole exercise)
------------------------------------------------------------------------------------
Captaincy is a pure right-tail bet - you want the single biggest score, not the safest.
So we compare, per gameweek over the MID+FWD pool, the actual points captured by the
captain you'd pick when RANKING BY:
  * expected points (what the production pipeline does today),
  * P(haul) (this module's upside probability),
  * a simple risk-adjusted blend of the two.
If P(haul) captures more actual captain points than E[points], the loss/upside framing
is adding signal the mean forecast discards. We also report the rank correlation between
P(haul) and E[points]: if it's ~1.0 the reframing is redundant; well below 1.0 means there
is independent information here worth pursuing.

Run:  python -m fpl.model.loss_probability
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fpl import config, features
from fpl.model.models import LGB_PARAMS
from fpl.model.metrics import top1_capture

POSITIONS = ["GK", "DEF", "MID", "FWD"]

BLANK_THRESHOLD = 2    # total_points <= 2  -> "blank" (the downside / loss event)
HAUL_THRESHOLD = 10    # total_points >= 10 -> "haul"  (the upside / captaincy event)


def _lgb_classifier():
    """LightGBM binary classifier sharing the point-forecast models' tree config
    (depth / learning rate / regularisation) so only the objective changes - same
    reasoning as fpl.model.probabilistic reusing LGB_PARAMS for quantile regression."""
    params = {k: v for k, v in LGB_PARAMS.items() if k not in ("objective", "metric")}
    return lgb.LGBMClassifier(objective="binary", **params)


# --- proper scoring rules & calibration ------------------------------------------------
def brier(y, p):
    """Brier score = mean squared error of a probability forecast. Proper scoring rule;
    lower is better; equals the outcome variance if you just predict the base rate."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p, eps=1e-15):
    """Cross-entropy / log-loss. Proper scoring rule that punishes confident wrong
    probabilities far harder than Brier - the honest metric for rare-event calibration."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    """ROC-AUC: probability the model ranks a random positive above a random negative.
    Ranking quality is what actually matters - every FPL decision compares players."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p)))


def expected_calibration_error(y, p, bins=10):
    """Bin predictions into `bins` equal-width buckets and average |mean predicted -
    mean realised| weighted by bucket size. 0 = perfectly calibrated. A safety-first
    optimiser can only trust these probabilities if this is small."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1])  # bins-1 interior edges -> bucket ids 0..bins-1
    err = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def evaluate(train_max_gw=152, test_min_gw=153, test_max_gw=183):
    """Fit per-position BLANK and HAUL probability models on GW<=train_max_gw, score them
    on the same GW153-183 window fpl.model.train uses, and test whether the resulting
    upside probability makes better captaincy decisions than the mean point-forecast."""
    raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
    df = features.build_feature_frame(raw)
    fcols = features.feature_columns(df)
    tr = df[df["GW_global"] <= train_max_gw]
    te = df[(df["GW_global"] >= test_min_gw) & (df["GW_global"] <= test_max_gw)].copy()

    print(f"\n=== Probability-of-loss forecasting: static split (train GW<={train_max_gw}, "
          f"test GW{test_min_gw}-{test_max_gw}) ===")

    # --- 1. BLANK (loss) and HAUL (upside) probability models, per position -------------
    for event, thresh, cmp in [("BLANK <=2 (loss)", BLANK_THRESHOLD, "le"),
                               ("HAUL >=10 (upside)", HAUL_THRESHOLD, "ge")]:
        print(f"\n--- P({event}) --- proper scoring rules; base rate is the floor to beat")
        print(f"{'Position':<9}{'base_rate':<11}{'brier':<9}{'log_loss':<11}{'auc':<8}{'cal_err':<9}")
        for pos in POSITIONS:
            trp, tep = tr[tr["position"] == pos], te[te["position"] == pos]
            if trp.empty or tep.empty:
                continue
            y_tr = (trp["total_points"] <= thresh).astype(int) if cmp == "le" else (trp["total_points"] >= thresh).astype(int)
            y_te = (tep["total_points"] <= thresh).astype(int) if cmp == "le" else (tep["total_points"] >= thresh).astype(int)
            if y_tr.nunique() < 2:
                continue
            clf = _lgb_classifier().fit(trp[fcols], y_tr)
            p = clf.predict_proba(tep[fcols])[:, 1]
            print(f"{pos:<9}{y_te.mean():<11.3f}{brier(y_te, p):<9.4f}{log_loss(y_te, p):<11.4f}"
                  f"{auc(y_te, p):<8.3f}{expected_calibration_error(y_te, p):<9.4f}")

    # --- 2. Decision-relevance: does P(haul) pick better captains than E[points]? --------
    # Captaincy is a pure right-tail bet, so pool MID+FWD (where captains come from) and,
    # per gameweek, compare the actual points captured by the top-ranked player under three
    # rankings. If P(haul) beats E[points] here, the upside probability carries signal the
    # mean forecast discards; if the two rankings correlate ~1.0, the reframing is redundant.
    cap_pos = ["MID", "FWD"]
    trc = tr[tr["position"].isin(cap_pos)]
    tec = te[te["position"].isin(cap_pos)].copy()

    ev_model = lgb.LGBMRegressor(**LGB_PARAMS).fit(trc[fcols], trc["total_points"])
    haul_model = _lgb_classifier().fit(trc[fcols], (trc["total_points"] >= HAUL_THRESHOLD).astype(int))

    tec["ev"] = ev_model.predict(tec[fcols])
    tec["p_haul"] = haul_model.predict_proba(tec[fcols])[:, 1]
    # Risk-adjusted blend: expected points scaled up by upside probability. Deliberately
    # simple (one obvious functional form), just to see if ANY upside tilt helps - not tuned.
    tec["blend"] = tec["ev"] * (1.0 + tec["p_haul"])

    print("\n--- Captaincy decision-relevance (MID+FWD pool, per gameweek) ---")
    print("top1_capture = actual points of your #1 pick / the true best possible each GW "
          "(1.0 = always captained the week's top scorer)")
    for label, col in [("rank by E[points]", "ev"), ("rank by P(haul)", "p_haul"),
                       ("rank by blend E[pts]x(1+P(haul))", "blend")]:
        cap = top1_capture(tec, "GW_global", "total_points", col)
        print(f"  {label:<34} top1_capture = {cap:.3f}")

    rho, _ = spearmanr(tec["ev"], tec["p_haul"])
    print(f"\n  Spearman(E[points], P(haul)) = {rho:.3f}  "
          f"(near 1.0 => P(haul) is redundant with the mean forecast; "
          f"lower => independent upside signal)")


if __name__ == "__main__":
    evaluate()
