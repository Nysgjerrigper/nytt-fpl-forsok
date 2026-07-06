# Research log

A running log of major decisions, experiments, and their results for this project - what was
tried, why, and what actually happened, so results are reproducible and don't need to be
re-derived from git history or re-litigated later. Newest entries at the top. See `CLAUDE.md`
for the current architecture; this file is the history of *why* it looks that way.

## 2026-07-06 - Tier-1 upgrade batch: new features, combination bake-off, mean-vs-median diagnostics

One coordinated batch implementing the research review's Tier-1 recommendations. New code:
~37 new features in `fpl/features.py` (EWMA form halflife-3, per-90 rates, opponent recent-form
strength `opp_*_roll6`, shifted xP, plus a semantic split of the old mislabeled `_season_avg` -
which was actually a CAREER mean - into honest `_season_avg` (resets per season) and `_career_avg`;
78 -> 115 columns); season-aware NaN handling (xG family / `starts` stay NaN pre-2022-23 instead of
being 0-filled as fake zeroes); decision-aligned metrics (`rmse`, `bias`, `total_calibration`,
`spearman_by_group`, `top1_capture`); robust combiners (equal-weight top-k, ridge stacking) +
`fit_weights` dispatcher; a combination bake-off + diagnostics table in `fpl.model.train`; Optuna
tuning module (`fpl/model/tuning.py`, time-ordered CV, not yet run at scale); experiment logger
(`fpl/experiment.py` -> `experiments/results.csv`); GitHub Actions CI. All adversarially reviewed
for leakage (opponent-merge verified shift-strict; new-feature/target correlations sane at
+-0.03-0.5). Results on the standard GW153-183 split:

**1. Combination bake-off: `single:catboost` wins at ALL four positions** (eval-half MASE
0.502/0.689/0.702/0.788 vs NNLS 0.547/0.782/0.797/0.884; equal-weight top-k is second everywhere,
ridge worst). Production ensembles are now CatBoost-only per position, chosen empirically by the
bake-off rather than assumed. Confirms the GW169-226 head-to-head (next entry) on a second window.

**2. The new features are roughly a WASH for CatBoost on this window** - honest negative-ish
result: full-window CatBoost MASE moved 0.513->0.517 (GK), 0.705->0.706 (DEF), 0.731->0.730 (MID),
0.849->0.847 (FWD). The feature expansion + semantic fixes neither helped nor hurt headline
accuracy here. They're kept: the semantic fixes are correctness issues regardless, the opponent/
EWMA features may pay off after hyperparameter tuning (still untuned), and nothing regressed.

**3. Mean-vs-median diagnostics confirm the suspected trap, with a twist.** CatBoost (MAE loss)
is heavily level-miscalibrated: bias -0.32 to -0.60, total_calibration 0.44-0.63 - its forecasts
sum to only ~half the points actually scored (median-flattening, exactly as predicted). The NNLS
blend is far better calibrated (0.80-0.99). BUT CatBoost still RANKS better where it matters:
higher top1_capture (captaincy quality) at every position - MID 0.565 vs 0.420 is a huge gap -
with near-equal Spearman. So: best ranker = worst calibrated. **Implication for the MILP: a
uniformly deflated forecast ranks players fine, but the MILP's -4-point transfer penalty and chip
logic are ABSOLUTE-scale - deflated predictions make transfers look relatively more expensive and
will distort those decisions. Before the backtest re-baseline, CatBoost's level should be
recalibrated (scalar or isotonic fit on the holdout window) - see TODO.**

## 2026-07-05 - Walk-forward head-to-head: CatBoost-only beats the 12-member blend everywhere

The decisive follow-up to the entry below. Honest walk-forward over GW169-226 (20 gameweeks,
step 3, members retrained each step, blend weights fit ONCE on GW153-168 - strictly before the
window, no leakage). Pooled MASE:

| Position | catboost-only | linear_svr-only | 12-member NNLS blend |
|---|---|---|---|
| GK  | **0.438** | 0.439 | 0.462 |
| DEF | **0.770** | 0.899 | 0.838 |
| MID | **0.677** | 0.690 | 0.780 |
| FWD | **0.701** | 0.749 | 0.753 |
| weighted avg | **0.684** | 0.738 | 0.761 |

**CatBoost-only wins at every position, ~10% ahead of the blend on the weighted average** - even
though the blend WEIGHTS CatBoost heavily at three positions, diluting it with noisier members
costs accuracy. This is the textbook Clemen (1989) forecast-combination result: with many
collinear members and a short weight-fitting window, weight-estimation error swamps the
theoretical gain from combining. Consequences implemented same-day: (a) `single:<model>` strategy
in `train.fit_holdout_weights` + `--weight-strategy` on `fpl.model.predict`; (b) a combination
bake-off in `evaluate_static_split` (single:catboost vs NNLS vs equal-weight top-k vs ridge
stacking, all on identical held-out rows) that picks the production combiner per position; (c)
top-k / ridge combiners added to `fpl/model/ensemble.py` as the literature's standard remedies,
so "simple average of a few good models" gets a fair shot against both extremes.

## 2026-07-04 - Expanded model registry: CatBoost is the new best single model; ensemble now trails it

Added six techniques in one push (user request to "try everything"): LinearSVR, capped-sample RBF
SVR, XGBoost, CatBoost (MAE loss), PLS regression (20 components), and an empirical-Bayes
hierarchical shrinkage baseline (player mean shrunk toward position mean - the cheap conjugate
version of a Bayesian hierarchical model, lives in `baselines.py` since it needs player identity).
All kept in the registry regardless of result, per project convention. MASE on the GW153-183
static split, best performers:

| Position | catboost | linear_svr | rbf_svr | pls | xgboost | eb_shrink | ensemble* (12-member) | old ensemble (7-member) |
|---|---|---|---|---|---|---|---|---|
| GK  | **0.513** | 0.513 | 0.571 | 0.602 | 0.661 | 0.832 | 0.534 | 0.588 |
| DEF | **0.705** | 0.713 | 0.758 | 0.847 | 0.831 | 1.090 | 0.747 | 0.799 |
| MID | **0.731** | 0.751 | 0.785 | 0.854 | 0.881 | 1.060 | 0.806 | 0.799 |
| FWD | **0.849** | 0.869 | 0.898 | 1.050 | 1.066 | 1.328 | 0.853 | 0.959 |

Findings, honestly stated: **CatBoost with MAE loss is the best single model at every position** -
by a wide margin over everything that existed before this run (the MAE-aligned-loss hypothesis
from the LinearSVR check held, and CatBoost's ordered boosting beat LightGBM's tuned config
outright). LinearSVR is a close second. PLS gave no benefit over OLS (the collinearity idea didn't
pay). XGBoost and EB-shrinkage underperformed (EB worse than the rolling-mean baseline everywhere -
a fixed prior_strength=10 pooled over six seasons of drifting scoring rules is too blunt); both
kept as near-zero-weight registry members / comparison columns. **BUT the blended ensemble now
LOSES to standalone CatBoost at every position** (and the MID blend didn't even pick CatBoost) -
classic NNLS weight overfitting on a ~15-GW half-window with 12 collinear members. See TODO.md;
the holdout-weights fix below may partly address it.

## 2026-07-04 - Code/concept review: four modeling fixes, dead code deleted

Full review of `fpl/` at user request (bugs, dead code, SWE quality, conceptual errors). Dead code
deleted; four conceptual errors found and fixed same-day:

1. **Blend-weight leakage into the actual-points backtest.** train.py fit blend weights on the
   first half of GW153-183, saved them, and predict.py's walk-forward backtest reused them over
   that same window - the first half's predictions used weights fit on their own outcomes. Fixed
   with `train.fit_holdout_weights`: weights now always fit on a window strictly before whatever
   is being predicted (predict.py: 16 GWs before --start-gw; train.py/run_week.py: last 16 played
   GWs for production). **The 1966/1880 backtest numbers are modestly inflated by the old scheme**
   (their comparison remains fair - both leaked identically); re-baseline before quoting them
   against future runs.
2. **Live fixture staleness.** run_week.py copied each player's last-played row into future GWs,
   so live predictions scored next week's fixture with last week's FDR, and horizon GWs were
   feature-identical except home/away. Now: per-GW FDR from the FPL API's fixtures endpoint
   (incl. DGW averaging, matching fetch.py). Found in passing and also fixed: API team names
   ("Spurs") never matched dataset names ("Tottenham"), silently dropping those teams' players
   from every live prediction.
3. **Live forecasts excluded each player's most recent match** (shifted features reused as
   "current form" end one game early). Fixed with a synthetic next-GW row per player whose
   shifted features legitimately include everything played (`build_live_snapshot`, unit-tested),
   plus an active-in-last-38-GWs filter so departed players stay out of the optimizer pool.
4. **Index-check asymmetry** - ensemble was scored on the test window's 2nd half but OLS on the
   full window. Both now scored on the same held-out rows.

Backtests were never affected by 2-3 (their rows carry correct features); live mode was. None of
this changes any relative comparison already logged.

## 2026-07-04 - Refreshed `report/main.typ` and `notebooks/eda.ipynb` for the current 6-season state

The report and EDA notebook were last generated against the 4-season (2022-23+) history and were
stale after the history extension to 2020-21 and the fixture/minutes/probabilistic work (both
below) - still citing 1900 points and the old GW77-107 window.

**Re-ran `notebooks/eda.ipynb`** against the current 6-season `master_dataset.csv` (162,981 rows).
Per-season/position descriptive stats now cover 2020-21 through 2025-26 (previously 2022-23
onward only) - distribution shape is stable across all six seasons (medians flat at 1-2 points
everywhere). One new finding from the extra history: the Augmented Dickey-Fuller test on average
DEF points-per-gameweek no longer rejects the unit-root null over the full six seasons (p=0.41,
was stationary with 4 seasons) - plausibly the 2025-26 `defensive_contribution` scoring-rule
change shifting the series' mean. GK/MID/FWD remain stationary. Doesn't change any modeling
choice (rolling/shifted features already track a drifting mean), but is a reason for caution
around any future fixed-mean baseline (AR(1)/ARIMA) specifically at DEF.

**Rewrote `report/main.typ`** to reflect the current state: 6-season data/history-extension
section (with its own MASE table), the fixture/minutes/probabilistic results and the unresolved
1966->1880 actual-points regression, an updated backtest table (1526/1811/1900/**1966**/1880),
updated abstract/conclusion (25% -> 29% improvement figure, since that's now measured against the
6-season 1966 result rather than the 4-season 1900 one), and discussion-section notes on the new
GK-OLS and DEF-non-stationarity findings. The original 4-season forecasting-technique comparison
table (SES/Theta/Croston/AR1/ARIMA vs ensemble) was kept as-is with a caveat that it predates the
history extension - re-running all seven baselines on the current 6-season/GW153-183 window was
judged not worth doing right now (the *relative* ranking between simple techniques is not expected
to change with more history; only the absolute ensemble/OLS numbers, which are already refreshed
elsewhere in the report). Recompiled to `report/main.pdf` via `typst compile`, clean (no warnings).

## 2026-07-03 - Forecasting hardening: MASE, constraint tests, legacy cleanup

**Context:** repo previously had no formal error metric beyond MAE and no test suite at all.
Decided to prioritize hardening the existing forecast+MILP pipeline over adding new squad-
optimization techniques (see next entry for why the MILP direction was dropped).

**Added MASE** (`fpl/model/metrics.py`) alongside MAE, since FPL points are an intermittent
series (most players score 0 most weeks) where raw MAE is hard to read as "good" or "bad"
without a floor to compare against. Scale fit on train-only data (no leakage into the metric
itself). Result on the GW77-107 backtest window: ensemble MASE < 1 (beats the naive
last-gameweek forecast) for GK/DEF/MID, but **> 1 for FWD (~1.07)** - forwards are the hardest
position to forecast, a finding MAE alone didn't make obvious.

**Added `tests/test_optimize_constraints.py`** - first test in the repo. Verifies the MILP
optimizer's output always satisfies squad size (2/5/5/3), budget (<=1000 in the 0.1m-unit
scale), max-3-per-club, and starting-XI formation constraints. Confirmed it actually catches
regressions by temporarily breaking a constraint and watching it fail, then reverting.

**Deleted `legacy/MILP Py/` and `legacy/R Forecast/`** (~26MB: 8 near-duplicate MILP scripts,
old R/LSTM code, Keras weights, EDA plots) per explicit user direction - already in git
history / a public repo, and the code itself was low quality (copy-pasted per position, no
shared functions), so not worth curating as a reference. Kept `legacy/baseline_outputs/`
since `fpl/model/train.py` actively reads `Validation_Predictions_Clean_v2.csv` from it as
the old-LSTM comparison benchmark.

## 2026-07-03 - Read Venter & van Vuuren (2024) in full; deprioritized MILP work

Read the complete paper (the original inspiration for `fpl/milp/optimize.py`). Its §4 MILP
formulation matches what's already implemented almost exactly (closing-window MIP, budget/
formation/club constraints, -4-per-extra-transfer penalty, same Kristiansen-et-al lineage) -
confirmed there's nothing new to add on the optimizer side. User's own read: "the MILP model
is already outdated [i.e. solved]... the most important part of the project is the
forecasting." Decision: **no further MILP work planned**; forecasting quality is the priority.

The paper's own conclusion supports this: their case study (2020/21 season) placed in the top
4.08% of ~8.24M FPL managers using a full-season-lookahead MILP over decent-but-not-exceptional
forecasts - the paper credits the *lookahead horizon* and *forecast quality*, not any
sophistication in the MIP itself, for the result.

## 2026-07-03 - Tested Croston's method as a forecast baseline: rejected

The Venter paper found Croston's method (built for intermittent demand) among its
better-performing individual forecasters. Implemented it (`fpl/model/baselines.py`) and
tested it pooled across each position (GK/DEF/MID/FWD) against the existing rolling-average
baseline and the production ensemble.

**Result: rejected.** Croston underperforms everything, at every position, worst at FWD:

| Position | ensemble MASE | baseline MASE | croston MASE |
|---|---|---|---|
| GK  | 0.64 | 0.69 | 0.89 |
| DEF | 0.87 | 0.94 | 1.06 |
| MID | 0.87 | 0.98 | 1.10 |
| FWD | 1.07 | 1.13 | 1.39 |

Why: Croston only updates its internal state on non-zero observations, so it's slow to react
when a player goes cold (loses form, gets benched) - it keeps forecasting off older, higher
scoring patterns. The Venter paper's actual result came from selecting the *best method per
individual player*, not from Croston being universally good - pooling it across an entire
position (as tested here) hides that per-player heterogeneity. Left in the codebase as a
labeled comparison column in `fpl/model/train.py`'s output (documents that this was tried),
not blended into the production ensemble.

**Per-player model selection** (rather than the current one-ensemble-per-position) is the
real structural idea worth exploring later if forecasting work continues - a genuine
architecture change, not attempted yet.

## 2026-07-03 - Tested econometrics/financial-forecasting baselines: naive drift, SES, Holt, AR(1)

Added four more classic time-series baselines (`fpl/model/baselines.py`) - naive drift, simple
exponential smoothing (SES), Holt's linear trend (double exponential smoothing), and a pooled
AR(1) (single-lag OLS autoregression, fit per position) - to check whether "more principled"
econometric methods beat the ad-hoc rolling-average baseline the pipeline already uses.

**Result: mixed, mostly negative.** MASE on the GW77-107 static split:

| Position | ensemble | baseline (roll3) | naive drift | SES | Holt | croston | AR(1) |
|---|---|---|---|---|---|---|---|
| GK  | 0.64 | 0.69 | 0.76 | **0.68** | 0.72 | 0.89 | 0.87 |
| DEF | 0.87 | 0.94 | 1.04 | **0.90** | 0.95 | 1.06 | 1.06 |
| MID | 0.87 | 0.98 | 1.09 | **0.94** | 0.99 | 1.10 | 1.11 |
| FWD | 1.07 | 1.13 | 1.20 | **1.11** | 1.15 | 1.39 | 1.29 |

Only **simple exponential smoothing (fixed alpha=0.3)** beats the existing rolling-average
baseline, at every position - a small, genuine improvement, but still well behind the full
ensemble everywhere. Naive drift, Holt's linear trend, and the pooled AR(1) all underperform
the existing baseline: player points don't have a persistent linear trend worth extrapolating
(Holt's extra trend term adds noise rather than signal), and a single pooled AR(1) coefficient
per position is too coarse next to the ~70-feature ML models already in the ensemble. None of
these are being blended into the production ensemble - the existing 6-model ensemble wins
comfortably everywhere, and SES's improvement over the ad-hoc baseline isn't large enough on
its own to be worth the added surface area. Left as comparison columns in
`fpl/model/train.py`'s output.

## 2026-07-04 - Added fixture difficulty, minutes projection, and probabilistic forecasting

Three additions requested together, all validated the same honest way.

**Fixture difficulty + fixture-window features** (`fpl/data/fetch.py`, `fpl/features.py`). Merged
the official FPL Fixture Difficulty Rating (FDR, 1-5) from each season's `fixtures.csv` onto every
player row by (team, GW): `fixture_difficulty` (this GW's opponent) and `fixture_difficulty_next3`
(mean FDR over this + the next two scheduled fixtures - the "easy run of fixtures" signal). These
are NOT shifted/leakage: fixture lists and their ratings are published before a gameweek is played,
so a model predicting GW t genuinely knows who each team plays at t, t+1, t+2. 100% merge coverage
across all six seasons (double gameweeks averaged to one value per team-GW).

**Minutes-projection ("nailedness") features** (`fpl/features.py::add_minutes_features`):
`start_rate_roll5` (rolling fraction of last 5 games started - uses the `starts` column where
present, falls back to a minutes>=60 proxy for 2020-21/2021-22) and `mins60_rate_roll5` (rolling
fraction with a full 60+-minute appearance, robust across all seasons since `minutes` always
exists). Both shifted one GW like the other rolling features - a player who won't start scores ~0,
so projecting minutes from recent starts is one of the single most predictive FPL signals.

**Result: ensemble MASE improved at every position** (same GW153-183 window):

| Position | before (6-season) | + fixture + minutes |
|---|---|---|
| GK  | 0.595 | 0.588 |
| DEF | 0.830 | **0.799** |
| MID | 0.811 | 0.799 |
| FWD | 0.984 | **0.959** |

DEF improved most, exactly where fixture difficulty should matter most (clean-sheet dependence).
Walk-forward MASE also improved (0.777 -> 0.772). Ensemble still beats the OLS index everywhere.

**BUT the actual-points backtest REGRESSED: 1880, down from 1966 (-86) on the same GW153-183
window.** This is a genuine, important divergence - forecast accuracy (MASE) improved at every
position, yet the squads the optimizer built from those "better" forecasts scored fewer real
points. It's exactly the failure mode CLAUDE.md's "Backtesting reference point" warns about (MASE
improvements don't translate 1:1 to points once the MILP is in the loop), showing up for real for
the first time this project. Most likely cause: fixture-difficulty features make the mean forecast
smoother/more regressed-to-the-mean, which nudges the optimizer toward "safe" nailed players and
away from the high-ceiling differentials that actually haul on the 2x captain multiplier - the MILP
maximizes *expected* points and is blind to variance/upside. This is precisely the gap the new
probabilistic module (below) exists to close, and it's why the actual-points backtest, not MASE, is
the project's gold-standard check. **Decision: the features are committed (requested, well-tested,
and genuinely better forecasters) but this points regression is flagged as unresolved - do NOT
treat fixture+minutes features as a settled net win until the captaincy/variance interaction is
understood (see TODO.md). If actual points are the only thing that matters, reverting to the
1966-scoring config is a one-line change (drop FIXTURE_FEATURES/MINUTES_FEATURES from
features.feature_columns).**

**Probabilistic forecasting** (`fpl/model/probabilistic.py`, `tests/test_probabilistic.py`). New,
separate module (does NOT touch the point-forecast ensemble or the squad optimizer): per-position
LightGBM quantile regression at p10/p50/p90, giving a prediction interval + median per
player-gameweek instead of a single number. Motivation is captaincy/risk: two players with equal
expected points aren't equal decisions - a boom-or-bust forward has more upside on the 2x captain
multiplier. Evaluated with pinball loss (the proper scoring rule these models minimize) and
interval coverage. Coverage of the [p10, p90] band came out 0.88-0.93 vs the ideal 0.80 - the
intervals are slightly too wide (mildly under-confident), expected with zero-inflated data where
the p10 quantile pins at 0 for blank-prone players; FWD was best-calibrated at 0.88. Usable as-is
for relative uncertainty ranking; tightening calibration (e.g. conformal adjustment) is a possible
follow-up. Quantile-crossing is repaired by row-wise sorting; a unit test guards the monotonicity.

## 2026-07-04 - Extended history back to 2020-21 (from 2022-23): a clear win

Extended `config.DEFAULT_START_SEASON` from 2022-23 to 2020-21 to give models more history per
player. Verified empirically first (checked vaastav's raw column headers season by season) that
this does NOT cost the power predictors: `bps`, `ict_index`, `influence`, `creativity`, `threat`
are present all the way back to 2016-17, and `position`/`team` come directly from `merged_gw.csv`
back to 2020-21 (older seasons would need a `players_raw.csv` join `fetch.py` doesn't do). What IS
lost for 2020-21/2021-22 rows: the Opta expected-goals family (`expected_goals`, `expected_assists`,
`expected_goal_involvements`, `expected_goals_conceded`) and `starts`, all only present from
2022-23 - NaN for the older two seasons, which LightGBM handles natively.

Dataset grew from ~113k rows (4 seasons) to ~163k (6 seasons). Because `GW_global` is
season-ORDINAL, the backtest window moved: the 2024-25-GW1-31 validation window that was GW77-107
is now GW153-183 (updated `evaluate_static_split` defaults and the walk-forward start-GW in
`train.py`, plus CLAUDE.md's numbering examples).

**Result: a clear improvement at every position.** MASE on the same 2024-25-GW1-31 window:

| Position | ensemble, 4 seasons | ensemble, 6 seasons | OLS index, 6 seasons |
|---|---|---|---|
| GK  | 0.637 | 0.595 | **0.579** (now beats ensemble) |
| DEF | 0.874 | 0.830 | 0.842 |
| MID | 0.871 | 0.811 | 0.869 |
| FWD | 1.067 | **0.984** | 1.056 |

Two notable findings: (1) **FWD's MASE dropped below 1.0 for the first time** across every
experiment this project has run - the extra history specifically helps the hardest position; and
(2) **OLS now edges the ensemble at GK** (0.579 vs 0.595) - the first position where the index
wins, plausibly because GK is a small/simple-signal position that benefits more from raw data
volume than from ensemble complexity. Worth revisiting the GK ensemble blend (see TODO.md).

**Actual-points backtest** (fpl.model.predict walk-forward -> fpl.milp.optimize, same GW153-183
window, horizon 3): **1966 actual points, up from the 1900 reference (+3.5%)** - the MASE
improvement translated into real squad points, not just a metric that looked better in isolation.

**Known imprecision left for tomorrow:** `fpl/features.py` fills missing stat columns with 0.0
before computing rolling averages, so for 2020-21/2021-22 the absent xG-family columns are encoded
as "this player recorded exactly 0 xG" rather than "this metric didn't exist yet." Results improved
despite this, but it's an imperfect encoding - see TODO.md.

## 2026-07-03 - Added plain OLS regression as the designated "index" benchmark

User (finance background) asked for a simple OLS regression to serve as the project's designated
*index* - the plain, unregularized benchmark every other model/baseline is ultimately judged
against, the same way a passive market index is the bar an active strategy has to clear, rather
than just another row in the comparison table.

Added `"ols"` to `fpl/model/models.py::FACTORIES` (plain `LinearRegression`, same ~70-feature
input and imputation/scaling pipeline as Ridge/ElasticNet - the only difference is no
regularization) and an explicit "index check" print block in `fpl/model/train.py` comparing the
ensemble's MASE against OLS's, per position, with an explicit "beats index" / "does NOT beat
index" verdict rather than requiring someone to eyeball the full comparison table.

**Result: the ensemble beats the OLS index at every position.**

| Position | OLS (index) MASE | ensemble MASE | verdict |
|---|---|---|---|
| GK  | 0.658 | 0.637 | beats index |
| DEF | 0.900 | 0.874 | beats index |
| MID | 0.927 | 0.871 | beats index |
| FWD | 1.158 | 1.067 | beats index |

Plain OLS also turns out to be a genuinely strong baseline in its own right - better than every
time-series baseline tested so far (SES, Theta, Croston, Holt, naive drift, AR(1), ARIMA) and the
old ad-hoc rolling-average baseline at every position, only trailing the regularized linear models
(Ridge, ElasticNet) by a small margin, as expected given the ~70 engineered features include a lot
of mutually correlated rolling-window statistics that unregularized OLS can't down-weight. This is
a reassuring result: it confirms the production ensemble's added complexity (6 model types,
NNLS-blended) is earning its keep against a simple, honest benchmark, not just against a weak
straw-man baseline.

## 2026-07-03 - Per-player model selection planning branch; Typst report; EDA notebook

Spun off a background planning agent (not implementation) to assess the per-player forecasting
model selection idea flagged as the real remaining structural direction in the entry below - see
`PER_PLAYER_MODEL_SELECTION_PLAN.md` on branch `worktree-agent-afeff933546ce7d37`. Its key
finding: ~35% of players active in the current season have zero prior-season history, and median
prior history among the rest is only 38 gameweeks - capping the technique's plausible reach to
roughly 60-65% of the live pool. Recommendation: a cautious, single-position pilot before any
larger commitment, not a full implementation yet.

Added `notebooks/eda.ipynb` - a Python rebuild of the thesis-era R EDA (`legacy/R Forecast/EDA.R`,
since deleted): descriptive statistics, boxplots, KDE density plots, QQ plots, season comparisons,
a ridgeline plot, and formal ADF/Shapiro-Wilk tests, all per position. Confirms empirically what
motivated MASE over MAE in the first place: `total_points` is heavily right-skewed and
zero-inflated at every position, and Shapiro-Wilk rejects normality decisively everywhere.

Added `report/main.typ` (compiles to `report/main.pdf` via `typst compile`) - a project status
report summarizing the architecture, evaluation methodology, and all forecasting experiments and
their results from the entries below, for anyone who wants the project state without reading this
whole log.

## 2026-07-03 - Tested Theta method and per-player ARIMA: both rejected, SES remains best baseline

Added the two remaining econometric/financial-forecasting techniques the Venter paper flagged
as relatively strong that hadn't been tried yet: the **Theta method** (averages a linear trend
line with an exponentially-smoothed curvature-doubled line - a standout simple method in the
M3/M4 forecasting competitions) and **per-player ARIMA(1,0,1)** (via `statsmodels`, the one new
dependency added for this - fit once per player on `train_df`, not re-fit every gameweek, since
per-row refitting at this scale would be far too slow).

**Result: both rejected as production baselines.** MASE on the GW77-107 static split:

| Position | ensemble | baseline (roll3) | SES (best so far) | theta | ARIMA |
|---|---|---|---|---|---|
| GK  | 0.64 | 0.69 | 0.68 | 0.70 | 0.86 |
| DEF | 0.87 | 0.94 | 0.90 | 0.91 | 1.10 |
| MID | 0.87 | 0.98 | 0.94 | 0.95 | 1.14 |
| FWD | 1.07 | 1.13 | 1.11 | 1.14 | 1.36 |

Theta beats the ad-hoc rolling-average baseline at DEF/MID but is roughly tied or slightly
worse at GK/FWD - and loses to SES at every position. ARIMA underperforms the existing
baseline everywhere, likely because a single fixed-order (1,0,1) fit per player, estimated
once and never updated, can't adapt to the short, noisy, zero-heavy series FPL points produce.

**Standing verdict across all baselines tested so far: SES is the best of the simple methods,
but nothing beats the full ensemble anywhere.** No further econometric baselines planned unless
a new candidate technique is specifically proposed - the marginal-return pattern here (test
honestly, most things lose to the existing ensemble) is now well established.

## 2026-07-03 - Explored Matthews et al. (2012) Bayesian belief-state MDP as a parallel branch

User found the paper's Bayesian RL approach (belief-state MDP + Q-learning over simulated
match outcomes) interesting, despite it being unrelated to and much larger in scope than the
production pipeline. Built as an independent, isolated-worktree branch
(`bayesian_manager/`, branch `worktree-agent-a22ee9f2acc804a35`) so it couldn't block or
entangle the main hardening work.

**Result:** myopic manager scored 1083 points, Q-learning variant scored 847 points (made zero
transfers all 31 gameweeks - a real hyperparameter finding, not a bug), both over the same
GW77-107 backtest window where the production pipeline scores 1900. Well behind, as expected
given the necessary simplifications (proxy absence detection, simplified club goal model, no
bonus-points/cards/saves simulation, single-step lookahead instead of full Bellman search).
Sitting on its own branch, not merged, not blocking anything.

## Earlier history

The project began as a Master's thesis: LSTM forecasting (R/Keras) + a hand-copied MILP
squad optimizer (8 near-duplicate Python scripts, one per experiment variant). Rewritten into
the current single Python pipeline (`fpl/`) so it could run weekly during a season rather than
as a one-off validation - see the "Replace LSTM+R pipeline with Python GBM ensemble +
consolidated MILP" commit for that transition. Validated against the old system on the same
GW77-107 backtest window: old LSTM+MILP scored 1526, new LightGBM+MILP scored 1811, the new
6-model ensemble+MILP scored 1900 (see `CLAUDE.md`'s "Backtesting reference point").
