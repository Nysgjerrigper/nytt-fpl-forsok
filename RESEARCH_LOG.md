# Research log

A running log of major decisions, experiments, and their results for this project - what was
tried, why, and what actually happened, so results are reproducible and don't need to be
re-derived from git history or re-litigated later. Newest entries at the top. See `CLAUDE.md`
for the current architecture; this file is the history of *why* it looks that way.

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
