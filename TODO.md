# TODO

Working notes for what to pick up next. Newest planning at the top; check `RESEARCH_LOG.md` for
the full "why" behind each item. Nothing here is urgent - the repo is in a clean, committed,
pushed state.

## Code/concept review findings (2026-07-04)

From a full review of `fpl/` (errors, dead code, modeling concepts). Dead code deleted
(`features.split_by_position`, the `LGB_PARAMS` re-export chain, unused numpy import in
optimize.py). The four modeling-concept fixes were implemented the same day:

- **[FIXED] Live fixture staleness** - `run_week.build_future_predictions` now sets
  `fixture_difficulty`/`fixture_difficulty_next3` per future GW from the FPL API's fixture list
  (which carries official FDRs), instead of copying last week's values. Also fixed in passing:
  API team names ("Spurs") are now corrected to dataset names ("Tottenham") before lookup -
  previously those teams' players silently dropped out of every live prediction.
- **[FIXED] Live forecasts excluded each player's most recent match** - `run_week` now builds a
  synthetic next-GW row per player (`build_live_snapshot`), whose shifted features legitimately
  include everything played. Also filters the live pool to players active in the last 38 GWs.
  Guarded by `tests/test_live_snapshot.py`.
- **[FIXED] Blend-weight leakage into the backtest** - new `train.fit_holdout_weights` fits NNLS
  weights on a window strictly before whatever gets predicted. `predict.py` fits its own weights
  on the 16 GWs before `--start-gw` (no longer reads saved weights); `train.py` saves production
  weights fit on the last 16 played GWs; `run_week` does the same at run time. NOTE: the 1966 /
  1880 backtest numbers predate this fix and are modestly inflated - re-baseline before the next
  comparison (comparisons between them remain fair; both leaked identically).
- **[FIXED] Index-check asymmetry** - the "beats index" verdict now compares OLS and the ensemble
  on the same held-out rows.

Still open from the review:

- **[MEDIUM][rule currency] MILP free-transfer cap is outdated for live use.** `Q_bar=2` matches
  the pre-2024-25 rule; FPL now allows banking up to 5 FTs. Also the sell-at-current-value
  simplification (real FPL: purchase price + half the profit, rounded down) overstates budget
  growth. Both fine for historical comparison, wrong for live 2026-27 play. (MILP deprioritized
  per user - log only.)
- **[LOW][opportunity] `xP` (FPL's own pre-match expected points) is in the dataset for all six
  seasons and never used as a feature.** A shifted/rolling xP would inject FPL's own model as a
  feature - cheap to test.
- **[LOW][robustness] Player identity is name-based** (`pd.factorize(name)`) - two players sharing
  a name merge into one id (the Ben Davies patch exists for exactly this). A players_raw.csv
  id-based join would be sturdier if ever extending pre-2020-21.

## New from the Tier-1 batch (2026-07-06)

- **[HIGH] Recalibrate CatBoost's level before the MILP backtest re-baseline.** The bake-off made
  CatBoost-only the production forecaster (best ranker: top1_capture 0.35-0.57, beats every blend),
  but its MAE loss median-flattens the LEVEL: bias -0.32..-0.60, total_calibration 0.44-0.63 - its
  forecasts sum to roughly half the points actually scored. The MILP's -4 transfer penalty and chip
  thresholds are absolute-scale, so deflated forecasts will suppress transfers/chips. Fit a simple
  level recalibration (scalar multiplier, or isotonic for shape) on the pre-window holdout, THEN
  run the pending backtest re-baseline (`fpl.model.predict --weight-strategy single:catboost` ->
  MILP over GW153-183) and compare honestly against the (inflated, old-scheme) 1966/1880.
- **[MEDIUM] Run the Optuna tuner at scale.** `fpl/model/tuning.py` exists (time-ordered CV) but
  hasn't been run with real trial budgets. Tune catboost per position first (it's the production
  model), then lightgbm/xgboost; save via save_best_params and wire tuned params into models.py
  factories (needs position-aware factories - see tuning.py integration notes).
- **[LOW] New features were a wash on the static window** (CatBoost MASE ~unchanged, see
  RESEARCH_LOG 2026-07-06). Revisit after tuning - untuned trees may simply not be exploiting the
  new opponent/EWMA columns yet. If still a wash, consider pruning to keep the feature set lean.

## RESOLVED (2026-07-06): NNLS-ensemble-vs-CatBoost question
The walk-forward head-to-head (GW169-226) and the static-window bake-off both found
single:catboost beats NNLS/top-k/ridge at every position. Production is now CatBoost-only per
position, selected empirically by the bake-off in `fpl.model.train` each run (so if a future
feature/tuning change makes a blend win again, production follows automatically).

## Done since last sign-off (2026-07-04)
- Fixture-difficulty + fixture-window features (`fetch.py`/`features.py`) - improved MASE at every
  position, DEF/FWD most. See RESEARCH_LOG.md.
- Minutes-projection ("nailedness") features - `start_rate_roll5`, `mins60_rate_roll5`.
- Probabilistic forecasting module (`fpl/model/probabilistic.py`) - quantile regression, pinball +
  coverage eval, unit tests. Intervals slightly over-cover (see follow-up below).

## New follow-ups from that work
- **[HIGH] Resolve the fixture/minutes MASE-vs-points divergence.** Fixture+minutes features
  improved MASE at every position but the actual-points backtest DROPPED 1966 -> 1880 on GW153-183.
  Forecast got more accurate, squads scored fewer real points. Likely the smoother mean forecast
  makes the MILP prefer safe nailed players over high-ceiling captain differentials. Before trusting
  these features as a net win, investigate: (a) is it noise? re-run the backtest with a different
  horizon / a second window; (b) does feeding the probabilistic p90/upside into captain selection
  recover the points? (c) if not, consider dropping FIXTURE_FEATURES/MINUTES_FEATURES from
  `features.feature_columns` to revert to the 1966 config. This is the gold-standard-metric
  regression, so it outranks the MASE improvement until understood.
- **Calibrate the probabilistic intervals.** [p10,p90] coverage is 0.88-0.93 vs the ideal 0.80 -
  slightly under-confident. A conformal-prediction adjustment on a held-out slice would tighten it.
- **Use the probabilistic output for captaincy.** The p90/upside signal is exactly what should drive
  captain choice (upside on the 2x multiplier); currently the MILP just uses expected points. Could
  feed a variance/upside term into the captain selection.
- **Dedicated minutes model.** Current minutes features are lag-based projections; a proper two-stage
  model (predict expected minutes, feed into the points model) may help further, especially for
  rotation-risk players.

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
