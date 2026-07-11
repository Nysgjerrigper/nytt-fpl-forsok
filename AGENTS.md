# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

**Language rule (PO decision, 2026-07-11):** always answer the user in English, and write everything
that lands in the repo (docs, comments, commit messages, log entries) in English - even when the user
writes to you in Norwegian. The project's ML/finance terminology does not translate cleanly.

## What this repo is

A Fantasy Premier League (FPL) points-prediction + squad-optimization system, originally a Master's thesis
(LSTM forecasting in R + MILP squad selection in Python). It has since been rewritten into a single Python
pipeline (`fpl/`) so it can actually be run weekly during a season, not just as a one-off academic validation.
The old R/LSTM code and the original 8 near-duplicate MILP scripts have been deleted (they added no value once
`fpl/` replaced them, and the code itself was copy-pasted per position with no shared functions - see git
history if you need to look at them). `legacy/baseline_outputs/` is the one thing kept from that era: it holds
the old LSTM's validation predictions, read by `fpl/model/train.py` purely as a fixed benchmark input.

## Setup & commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

LightGBM on macOS requires OpenMP: `brew install libomp` if you hit a `libomp.dylib not loaded` error.

```bash
python -m fpl.data.fetch                              # (re)build Datasett/master_dataset.csv
python -m fpl.model.train                              # compare model types per position, save final ensembles
python -m fpl.model.predict --start-gw N --end-gw M    # walk-forward predictions CSV for backtesting
python -m fpl.milp.optimize --start-gw N --max-gw M --horizon H   # run the squad optimizer on a predictions CSV
python -m fpl.run_week --team-id <id> --horizon 3       # weekly driver: refresh data, retrain, recommend transfers
```

`pytest tests/` runs a lightweight regression suite (currently: a sanity check that the MILP optimizer's output
always satisfies budget/position/club/formation constraints - see `tests/test_optimize_constraints.py`). This
guards against constraint-violation bugs; it does NOT tell you whether a modeling/optimizer change is actually
better. For that, re-run `fpl.model.train` (prints MAE and MASE per model/position vs. a rolling-average
baseline and the old LSTM - see `fpl/model/metrics.py` for why MASE matters here) and/or a `fpl.milp.optimize`
backtest, and check the resulting `actual_total_points` sum against a prior run - see "Backtesting" below.

## Gameweek numbering (important, easy to get wrong)

There is no per-season GW1-38 reset in this codebase. `GW_global` (built in `fpl/data/fetch.py`) is a single
ascending counter across all seasons: the counter starts at `config.DEFAULT_START_SEASON`, so GW numbering is
season-ORDINAL, not calendar-fixed - it SHIFTS if you change the start season. With the current start of
2020-21: 2020-21 = GW 1-38, 2021-22 = GW 39-76, 2022-23 = GW 77-114, 2023-24 = GW 115-152, 2024-25 = GW
153-190, 2025-26 = GW 191-228 (`config.GWS_PER_SEASON = 38`). (Before history was extended back from 2022-23,
these were all 76 lower - e.g. 2024-25 was GW 77-114; this is why the backtest window moved from GW77-107 to
GW153-183.) `fpl/milp/optimize.py` derives wildcard-half boundaries from this (`math.ceil(start_gw / 38)`), so
never assume a raw `GW` column value maps directly to a real-world gameweek without checking which season it's in.

Season/gameweek discovery is dynamic, not hardcoded: `fpl/data/fetch.py` queries the GitHub API for what
season folders exist and checks whether `merged_gw.csv` is present (season finished) or falls back to fetching
`gw1.csv`, `gw2.csv`, ... one at a time until a 404 (season in progress). Don't reintroduce a hardcoded season
list or a hardcoded "last gameweek" constant - that was the recurring maintenance problem with the old R script.

## Architecture

**Data flow:** `fpl/data/fetch.py` (pulls + cleans vaastav's FPL GitHub data, resolves opponent-team IDs via
each season's `teams.csv`, merges official fixture-difficulty ratings from `fixtures.csv`, applies name/team
corrections in `fpl/config.py`) -> `Datasett/master_dataset.csv` -> `fpl/features.py` (~115 features per row:
shifted rolling-window form, TWO expanding-mean horizons per stat - `_season_avg` resets each season,
`_career_avg` doesn't; the pre-2026-07-06 `_season_avg` was actually the career mean under a wrong name -
plus EWMA form (halflife 3), per-90 rates, minutes-projection "nailedness", opponent recent-form strength
(`opp_*_roll6`, merged by opponent with a strict one-GW shift), shifted xP, and fixture-difficulty features.
Everything player-derived is shifted so no row sees its own outcome; fixture features are known-ahead, so not
shifted. Stats absent for a whole season - the Opta xG family and `starts` before 2022-23 - stay NaN rather
than being 0-filled: "not collected" is not "zero") -> `fpl/model/` (trains per-position models on
`features.feature_columns(df)`) -> predictions CSV -> `fpl/milp/optimize.py` (turns predicted points into an
actual squad/transfer/chip decision). `fpl/run_week.py` chains all of this for a live weekly run, additionally
pulling fixtures/current-squad state from the official FPL API (`fantasy.premierleague.com/api/...`), which
vaastav's historical dumps don't have.

**Modeling:** Four independent models per position (GK/DEF/MID/FWD) rather than one global model - position
determines what stats matter, and separating them avoids one position's scale dominating. The registry
(`fpl/model/models.py::FACTORIES`: LightGBM, XGBoost, CatBoost, OLS, Ridge, ElasticNet, PLS, Random Forest,
Extra Trees, kNN, LinearSVR, a capped-sample RBF SVR) keeps every model type ever tried - including ones that
don't help - per this project's practice of reporting negative results rather than hiding them. HOW the
registry members are combined is chosen empirically per position by a combination bake-off in
`fpl/model/train.py::evaluate_static_split`: `single:catboost` (best single model, no blending) vs NNLS vs
equal-weight top-k vs ridge stacking (`fpl/model/ensemble.py::fit_weights`), all scored on the same held-out
rows. This exists because a GW169-226 walk-forward head-to-head found CatBoost-only BEATS the 12-member NNLS
blend at every position (weighted MASE 0.684 vs 0.761) - the classic Clemen-1989 result that estimated
combination weights lose to the best single model when members are many and collinear. Production/backtest
weights are always fit via `train.fit_holdout_weights` on a window strictly BEFORE whatever gets predicted
(never reuse evaluation-window weights - that was a real leakage bug, see RESEARCH_LOG.md 2026-07-04). Saved
ensembles live in `fpl/models/<POSITION>.*` (gitignored - regenerate with `python -m fpl.model.train`).

Tree models (LightGBM, XGBoost, CatBoost) get raw features including NaNs (a player's first few gameweeks have
no rolling history yet; pre-2022-23 rows have no xG family at all) since they handle missing values natively.
Everything else in `fpl/model/models.py` gets wrapped in a `SimpleImputer` (+`StandardScaler` for
linear/distance models) because sklearn estimators can't take NaN input.

**Metrics** (`fpl/model/metrics.py`): MAE/MASE remain the headline accuracy numbers, but note the mean-vs-median
trap: MAE/MASE are minimized by predicting the conditional MEDIAN, while the MILP consumes conditional MEANS
and captaincy needs the upside tail - so a model can improve MASE yet build worse squads (observed for real:
see the fixture/minutes 1966->1880 regression in RESEARCH_LOG.md). The diagnostics that expose this - `rmse`
(mean-aligned), `bias`, `total_calibration`, `spearman_by_group` (ranking quality), `top1_capture` (captaincy
quality) - are printed by `fpl.model.train` alongside the MASE table; don't judge a modeling change by MASE
alone. `fpl/model/tuning.py` (Optuna, time-ordered expanding-window CV) tunes the GBM hyperparameters;
`fpl/experiment.py` appends run results to `experiments/results.csv` so experiments stay reproducible; CI
(`.github/workflows/ci.yml`) runs pytest on every push.

**Probabilistic forecasting:** `fpl/model/probabilistic.py` is a separate, parallel view - per-position LightGBM
quantile regression (p10/p50/p90) giving a prediction interval per player-gameweek instead of a single number,
for uncertainty-aware analysis like captaincy. It reuses the same `fpl.features` inputs but does NOT feed the
squad optimizer (which still consumes the point-forecast ensemble's single-number CSV). Evaluated with pinball
loss + interval coverage (`python -m fpl.model.probabilistic`), not MAE/MASE.

**Codex branch note - probabilistic buckets:** On branch `codex/probabilistic-bucket-models`, read
`HANDOFF.md` before continuing. `fpl/model/probabilistic_buckets.py` is a forecasting-only bake-off that
changes the target from numeric `total_points` regression to a five-bucket probability distribution
(`<=0`, `1-2`, `3-5`, `6-9`, `>=10`). It is a real model-family change, not just a data reshape. Keep it
parallel to production until a walk-forward comparison against the tuned CatBoost regression exists. Do not
wire bucket probabilities into `fpl.milp.optimize` without an explicit decision and a backtest.

**MILP optimizer:** `fpl/milp/optimize.py` is Kristiansen et al.'s formulation (budget, formation constraints,
captain/vice-captain, transfer costs, wildcard/free-hit/bench-boost/triple-captain chip logic), solved with PuLP
+ CBC as a rolling horizon (re-solved every gameweek over a `--horizon`-week lookahead, only the first
gameweek's decision is locked in before rolling forward). It supports two modes: fresh-build (default - assumes
an empty squad and full budget, used for backtests) and continuing an existing squad
(`--initial-squad`/`--initial-budget`/`--initial-ft`, used by `run_week.py` for real weekly use). Chip CLI args
(`--wc1-gw` etc.) use `0` as "disabled" - a gameweek value of `0` never matches a real GW, so this is not the
same as `None`; don't "fix" this by converting `0` to `None`; that inverts the semantics (see the fixed bug in
git history if touching this code).

## Backtesting reference point

The pipeline was validated against the old system by running both through the *same* MILP on the *same*
2024-25-season-GW1-31 window: old LSTM+MILP scored 1526 actual points, new LightGBM+MILP scored 1811, new
6-model ensemble+MILP scored 1900 - all with history starting at 2022-23 (so that window was GW77-107 then).
After extending history back to 2020-21 (see RESEARCH_LOG.md), the same 2024-25-GW1-31 window is now GW153-183,
and the 6-model ensemble+MILP scored 1966 there (+3.5%). When changing the modeling or MILP code, re-running
this comparison (`fpl.model.predict` walk-forward predictions into `fpl.milp.optimize`, same GW range) is the
way to check whether a change actually helps, not just whether MAE looks better in isolation - MAE improvements
don't always translate 1:1 into more actual points once the optimizer is in the loop.

## Known limitation

`fpl/run_week.py`'s live fixture/current-squad fetching from the official FPL API can only be exercised
end-to-end once a season is set up on the FPL site (typically a few weeks before its GW1) - it could not be
tested live as of this writing since the 2026-27 season hadn't opened yet.
