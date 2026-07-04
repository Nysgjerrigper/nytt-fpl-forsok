# TODO

Working notes for what to pick up next. Newest planning at the top; check `RESEARCH_LOG.md` for
the full "why" behind each item. Nothing here is urgent - the repo is in a clean, committed,
pushed state.

## Code/concept review findings (2026-07-04)

From a full review of `fpl/` (errors, dead code, modeling concepts). Dead code already deleted
(`features.split_by_position`, the `LGB_PARAMS` re-export chain, unused numpy import in
optimize.py). What remains, by severity:

- **[HIGH][live-mode bug] `run_week.py` uses stale fixture features for future gameweeks.**
  `build_future_predictions` copies each player's last-played-GW row - including its
  `fixture_difficulty`/`fixture_difficulty_next3` - and only updates opponent/was_home from the
  API. So live predictions score next week's fixture with LAST week's difficulty, and all
  horizon GWs get identical features except was_home (the model can't tell the GW2 opponent from
  the GW3 opponent). Fix: the FPL API's `/fixtures/` response already carries
  `team_h_difficulty`/`team_a_difficulty` - set the fixture features per future GW from that.
  Doesn't affect backtests (predict.py rows carry their own correct fixture features).
- **[HIGH][live-mode bug] Live forecasts exclude each player's most recent match.** The snapshot
  row's rolling features are shifted by one GW (correct for training), so when reused as "current
  form" they end one game early - the freshest, most informative game a player just played never
  enters the live forecast. Fix: recompute unshifted rolling stats as-of "after the last played
  GW" for the live snapshot only.
- **[MEDIUM][evaluation leakage] Blend weights leak into the actual-points backtest.** Weights are
  fit on the FIRST half of GW153-183 (train.py), saved, then predict.py's walk-forward backtest
  reuses them over that same window - so GW153-168 predictions use weights fit on those rows' own
  outcomes. The 1966/1880 headline numbers are modestly inflated; the 1966-vs-1880 COMPARISON is
  still fair (both leak identically). Fix: fit backtest weights on a window strictly before the
  backtest (e.g. last ~15 GWs of training data).
- **[LOW][evaluation inconsistency] The "index check" compares different eval sets.** Ensemble
  MASE is measured on the test window's 2nd half (correct holdout), but OLS/others on the full
  window. If the halves differ in difficulty, the "beats index" verdict is biased. Fix: report
  the index check on the same 2nd-half rows for both.
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
