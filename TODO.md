# TODO

Working notes for what to pick up next. Newest planning at the top; check `RESEARCH_LOG.md` for
the full "why" behind each item. Nothing here is urgent - the repo is in a clean, committed,
pushed state.

## Left open at sign-off (2026-07-04)

### 1. Decide how far back to extend history (the main open decision)
History was just extended from 2022-23 to **2020-21**, which was a clear win (+3.5% backtest
points, MASE down at every position - see RESEARCH_LOG.md). The open question is whether to go
further:
- **Keep 2020-21** (current state) - done, validated, committed. No action needed if you're happy.
- **Push to 2016-17** (max history, 10 seasons) - would need two pieces of new work:
  (a) a `players_raw.csv` join in `fpl/data/fetch.py` to recover `position`/`team`, which aren't
      in `merged_gw.csv` before 2020-21; and
  (b) accepting the loss of `xP` (FPL's own expected-points column, only present from 2020-21).
  Only worth it if a quick backtest shows it beats the 1966-point 2020-21 result - diminishing
  returns are likely.

### 2. Fix the xG-family zero-fill imprecision in `fpl/features.py`
`build_feature_frame` fills missing stat columns with `0.0` before computing rolling features. For
the newly-added 2020-21/2021-22 seasons, the Opta xG columns (`expected_goals`, `expected_assists`,
`expected_goal_involvements`, `expected_goals_conceded`) and `starts` don't exist, so they're
currently encoded as "player recorded exactly 0 xG" rather than "metric didn't exist yet." Options:
leave xG NaN (LightGBM handles it natively; only the linear models need the 0-fill via their
imputer), or add a per-column "was this stat available this season" flag. Low-risk, and results
already improved despite the imprecision - so this is a refinement, not a bug blocking anything.

### 3. Revisit the GK ensemble blend
With the extended history, plain OLS now slightly beats the 6-model ensemble at GK (MASE 0.579 vs
0.595) - the first position where the index wins. Worth checking whether the GK ensemble's NNLS
blend is overfitting on a small/simple-signal position, or whether GK should just use a simpler
model. Cheap to investigate via `fpl.model.train`'s per-position blend-weight printout.

## Still-open threads from earlier (not started, lower priority)

### 4. Per-player forecasting model selection - pilot
Planning doc exists at `PER_PLAYER_MODEL_SELECTION_PLAN.md` on branch
`experimental/per-player-model-selection` (pushed to GitHub). Recommendation there was a
single-position pilot before any full commit, given ~35% of active players have no prior-season
history. Not started.

### 5. Bayesian belief-state MDP manager - simplifications
`bayesian_manager/` on branch `experimental/bayesian-mdp-manager` (pushed) underperforms the
production pipeline (1083 vs 1966 pts). If ever revisited, its README lists the highest-leverage
fixes: a real absence list, a proper birth-process club model, and full point-category simulation
(bonus points especially). Not a priority.

## Housekeeping
- `report/main.typ` / `main.pdf` still cite the 1900-point backtest and the old GW77-107 window in
  a couple of places. Regenerate/update it to the 1966 / GW153-183 numbers when convenient
  (`typst compile report/main.typ report/main.pdf`).
