# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
ascending counter across all seasons: season N's gameweeks occupy `((N-1) * 38 + 1)` to `(N * 38)`
(`config.GWS_PER_SEASON = 38`). E.g. 2022-23 = GW 1-38, 2023-24 = GW 39-76, 2024-25 = GW 77-114, 2025-26 = GW
115-152. `fpl/milp/optimize.py` derives wildcard-half boundaries from this (`math.ceil(start_gw / 38)`), so
never assume a raw `GW` column value maps directly to a real-world gameweek without checking which season it's in.

Season/gameweek discovery is dynamic, not hardcoded: `fpl/data/fetch.py` queries the GitHub API for what
season folders exist and checks whether `merged_gw.csv` is present (season finished) or falls back to fetching
`gw1.csv`, `gw2.csv`, ... one at a time until a 404 (season in progress). Don't reintroduce a hardcoded season
list or a hardcoded "last gameweek" constant - that was the recurring maintenance problem with the old R script.

## Architecture

**Data flow:** `fpl/data/fetch.py` (pulls + cleans vaastav's FPL GitHub data, resolves opponent-team IDs via
each season's `teams.csv`, applies name/team corrections in `fpl/config.py`) -> `Datasett/master_dataset.csv`
-> `fpl/features.py` (adds shifted rolling-window and season-to-date form features per player, so no row ever
sees its own outcome) -> `fpl/model/` (trains per-position models on `features.feature_columns(df)`) ->
predictions CSV -> `fpl/milp/optimize.py` (turns predicted points into an actual squad/transfer/chip decision).
`fpl/run_week.py` chains all of this for a live weekly run, additionally pulling fixtures/current-squad state
from the official FPL API (`fantasy.premierleague.com/api/...`), which vaastav's historical dumps don't have.

**Modeling:** Four independent models per position (GK/DEF/MID/FWD) rather than one global model - position
determines what stats matter, and separating them avoids one position's scale dominating. Per position, six
cheap model types are trained (`fpl/model/models.py`: LightGBM, Ridge, ElasticNet, Random Forest, Extra Trees,
kNN) and blended into a `PositionEnsemble` (`fpl/model/ensemble.py`) via non-negative least squares weights.
Those blend weights are fit on one half of a held-out test window and evaluated on the other half
(`fpl/model/train.py::evaluate_static_split`) specifically so the reported ensemble accuracy isn't inflated by
fitting weights and evaluating them on the same rows. Saved ensembles live in `fpl/models/<POSITION>.*`
(gitignored - regenerate with `python -m fpl.model.train`, don't expect them to exist in a fresh clone).

Tree models (LightGBM) get raw features including NaNs (a player's first few gameweeks have no rolling history
yet) since LightGBM handles missing values natively. Everything else in `fpl/model/models.py` gets wrapped in a
`SimpleImputer` (+`StandardScaler` for linear/distance models) because sklearn estimators can't take NaN input.

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
GW77-107 window (2024-25 season, GW1-31): old LSTM+MILP scored 1526 actual points, new LightGBM+MILP scored
1811, new 6-model ensemble+MILP scored 1900. When changing the modeling or MILP code, re-running this
comparison (`fpl.model.predict` walk-forward predictions into `fpl.milp.optimize`, same GW range) is the way to
check whether a change actually helps, not just whether MAE looks better in isolation - MAE improvements don't
always translate 1:1 into more actual points once the optimizer is in the loop.

## Known limitation

`fpl/run_week.py`'s live fixture/current-squad fetching from the official FPL API can only be exercised
end-to-end once a season is set up on the FPL site (typically a few weeks before its GW1) - it could not be
tested live as of this writing since the 2026-27 season hadn't opened yet.
