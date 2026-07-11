# TODO

Working notes for what to pick up next. Newest planning at the top; check `RESEARCH_LOG.md` for
the full "why" behind each item. Nothing here is urgent - the repo is in a clean, committed,
pushed state.

## Audit follow-ups (2026-07-11) — see `AUDIT_2026-07-11.md` for full rationale

A full-repo audit produced the findings referenced below by ID (A1-A4 bugs, B1-B5
evaluation, C1-C5 prediction, D1-D2 optimizer, E1-E8 engineering). Items are grouped
into four clusters by dependency. **Cluster 1 repairs the measurement system itself and
must complete before Cluster 2's experiments are judged** — evaluating a new model
against a baseline with known leakage/optimism repeats the mistake that produced the
inflated 1966 headline. Clusters 3 and 4 are independent of the modeling work and can
run in parallel. Absorbs these pre-existing TODO entries: "Tune lightgbm/xgboost"
(2026-07-06 → item 2.4), "Dedicated minutes model" (2026-07-04 follow-ups → item 2.1),
"MILP free-transfer cap is outdated" (2026-07-04 review → item 3.4), "xP never used as
a current-GW feature" (2026-07-04 review → item 2.2), "Player identity is name-based"
(2026-07-04 review → item 4.8).

**Blocked on PO answers first** (audit §9): (Q1) what the project optimizes for now —
live 2026-27, portfolio, or both (moves Cluster 3 up/down); (Q2) whether the 2026-07-06
Optuna run really used GW<153 only (decides if item 1.2 is a docs gap or a
re-verification); (Q3) approval to freeze GW191-221 as a one-shot confirmation window
(item 1.5 is a real commitment — the window is never reused for selection).

### Cluster 1 — Repair the measurement system (do first, in this order)

- **[1.1][DONE 2026-07-11][bug A1] Live-path production parity.** Fixed on
  `fix/audit-cluster1`: `config.PRODUCTION_WEIGHT_STRATEGY` (= `single:catboost`) is now
  the one definition of the production model, consumed as the default by predict.py AND
  run_week.py; both fit through the new shared `train.fit_position_ensembles` (position-
  aware, so tuned params load live). The never-loaded save path was deleted
  (`train_final_ensembles`, `PositionEnsemble.save/load`, the stale `fpl/models/<POS>.*`
  artifacts); train.py's bake-off now warns if its empirical winner disagrees with the
  config constant. Verified: one-GW walk-forward smoke run fits catboost=1.00 per position.
- **[1.2][DONE 2026-07-11 (code)][repro A2] `--train-max-gw` in `fpl.model.tuning`** —
  done: folds capped at `config.TUNING_TRAIN_MAX_GW` (152) by default, cap recorded in the
  params JSON under `_meta` (stripped before the constructor splat), bucket module's
  hardcoded 152s now point at the same constant. **Still open:** depending on the PO's Q2
  answer, re-run the CatBoost tuning under the cap and re-verify 2107 (compute).
- **[1.3][MEDIUM][leakage A3] Shift by gameweek, not by row, in `features.py`.** In
  double gameweeks (8.6% of rows) the second fixture's shifted features currently
  include the first fixture of the SAME gameweek — information unavailable at the
  deadline. Fix all player-level shifted features (rolling, EWMA, per-90, minutes, xP)
  to use strictly-earlier `GW_global` values; also make the sort stable. Add a DGW
  leakage unit test next to the existing guards in `test_features_advanced.py`.
  Then **re-baseline**: re-run the GW153-183 backtest and update the standing number
  (2107 will move slightly). *Effort: ~1 day + one backtest run.*
- **[1.4][MEDIUM][method B2] Origin-based horizon backtest.** predict.py's walk-forward
  gives the MILP lookahead forecasts for t+1/t+2 built with information through t/t+1 —
  live mode can never have that. Add an export mode where, for each origin GW t, all of
  t..t+h are predicted with features frozen at t (same freezing run_week does), and run
  the MILP per origin set. Measure the gap vs the standard protocol once; this is the
  honest deploy-expectation number. Combine with 1.3's re-baseline so there is ONE new
  standing baseline, not two in sequence. *Effort: 1-2 days.*
- **[1.5][HIGH][method B1] One-shot confirmation backtest on the frozen 2025-26 window
  (GW191-221).** Needs PO approval (Q3) and should run AFTER 1.1-1.4 so it certifies the
  final protocol. Run the frozen production config there exactly once, report the number
  whatever it is (RESEARCH_LOG + experiments/results.csv), and never use the window for
  selection afterwards. *Effort: compute only.*
- **[1.6][DONE 2026-07-11 (tool)][method B3] Uncertainty on realized-points comparisons.**
  Done: `python -m fpl.milp.compare_backtests runA.csv runB.csv` — paired per-GW
  moving-block bootstrap CI on the total points difference + binomial sign test, with
  unit tests (`tests/test_compare_backtests.py`). **Still open:** the retroactive
  2107-vs-2059 run (the per-GW squad_selection CSVs are not on disk; regenerate them with
  the 1.3/1.4 re-baseline and run the comparison then), and folding "every realized-points
  verdict carries a CI" into HANDOFF's standing rules once used in anger.
- **[1.7][DONE 2026-07-11][metric B4] MASE scale consistent** — `train.py` (static split,
  bake-off, walk-forward) now uses per-position naive scales like `tuning.py`. MASE tables
  printed before 2026-07-11 are NOT comparable with new ones (see RESEARCH_LOG 2026-07-11);
  realized-points numbers are unaffected.
- **[1.8][LOW][scoring B5] Auto-subs + vice-captain activation in backtest scoring** —
  currently ignored, so absolute realized-points understate real FPL play equally for
  all configs. Either implement a simple auto-sub simulation in optimize.py's scoring
  block or document the omission in the report. *Effort: ~1 day or a footnote.*

### Cluster 2 — Forecast improvements (gated on Cluster 1's new baseline)

Judge every item here on the realized-points MILP backtest vs the post-1.3/1.4
baseline, with a 1.6 uncertainty interval — never on MASE movement alone.

- **[2.1][HIGH][C1] Dedicated minutes model** (two-stage: P(start)/E[minutes] × points
  per 90). 59% of rows are 0-minute rows and blank-prediction is dominated by the
  minutes signal; the current `start_rate_roll5` is a lagged proxy. Standard
  architecture in strong FPL systems. Most likely single biggest forecast gain.
  *Effort: 3-5 days.*
- **[2.2][MEDIUM][C2] Current-GW xP as a feature** — legitimate (xP is pre-match) IF the
  vaastav stamping is verified first: check per season whether raw xP correlates with
  outcomes beyond the plausible; if any season looks post-match, keep the lagged forms
  there. *Effort: hours + verification.*
- **[2.3][MEDIUM][C3] LambdaRank experiment** — three "better metrics, fewer points"
  episodes all point at within-GW ranking as what the MILP actually consumes. Train
  LightGBM lambdarank grouped by (GW, position), map monotonically to the points scale
  for the MILP's absolute-scale terms, backtest. A negative result is likely and fine —
  it tests the mechanism directly. *Effort: 1-2 days.*
- **[2.4][MEDIUM][C4] Tune LightGBM/XGBoost** so the registry ranking is
  tuned-vs-tuned, then re-check the combination bake-off (a tuned blend might beat
  tuned CatBoost). Mostly compute. Use the 1.2-capped tuner.
- **[2.5][LOW/PARKED][C5] Bookmaker odds features** — strongest exogenous signal in the
  football-prediction literature, but historical odds acquisition is a real data
  project. Park until 2.1-2.4 are exhausted.

### Cluster 3 — Live readiness for 2026-27 (needs 1.1; otherwise independent)

Do before the season opens on the FPL site; none of it affects backtests.

- **[3.1][HIGH][A4b] Use API availability in run_week** — filter or scale predictions by
  `status` / `chance_of_playing_next_round` from bootstrap-static. Cheapest real gain
  for live play; without it the optimizer happily buys injured players on good form.
- **[3.2][MEDIUM][A4a] Current prices from the API** — override the snapshot's stale
  `value` (last played row in vaastav's dump) with bootstrap `now_cost` so deadline
  budget math is right.
- **[3.3][MEDIUM][A4c] DGW handling in live horizon** — `build_team_fixture_map` keeps
  only the first fixture of a double gameweek while backtests sum per fixture; emit one
  prediction row per fixture (or scale by fixture count) so live stops undervaluing DGW
  players.
- **[3.4][MEDIUM][D1] Update FT rule to the 5-banked-transfers era** (`Q_bar=2`→5, one
  line + a backtest to see if horizon behaviour changes). Document the sell-price
  simplification and the chips-disabled backtest convention as known limitations.
- **[3.5][LOW][D2] FT/chip accounting test** — the solver's FT logic is duplicated in
  the Python state-rollover (`optimize.py:373-386`); a 5-6 GW synthetic test asserting
  the FT trajectory closes the silent-divergence class the constraint test doesn't cover.

### Cluster 4 — Engineering hygiene (anytime; 4.1 early, it reduces recurrence risk)

- **[4.1][MEDIUM][E1] One shared walk-forward harness** — predict.py,
  `probabilistic_buckets.evaluate_walk_forward`, and `walk_forward_predictions_csv` are
  three near-copies of "retrain every N, predict GW"; `fit_holdout_weights` and
  `fit_level_calibration` duplicate the member-training loop. A1 was exactly the class
  of bug this duplication breeds. *Effort: ~1 day; do alongside/after 1.1.*
- **[4.2][MEDIUM][E2] Reproducibility pinning** — lock dependency versions (pip-tools or
  a constraints file) and extend `experiment.py` to log library versions + data state
  (max GW, row count) per run. A CatBoost version bump can silently move 2107.
- **[4.3][LOW][E3] Stop tracking `master_dataset.csv`** (33 MB regenerable file; .git is
  159 MB because of it). Gitignore it, keep the thesis-era raw CSVs. History rewrite is
  optional and the PO's call.
- **[4.4][LOW][E4] Proper packaging** — `pyproject.toml` + `pip install -e .`, drop the
  `sys.path.insert` hack from every module.
- **[4.5][LOW][E5] Gate the rejected slow baselines** (per-row Theta refits, per-player
  ARIMA) behind `--with-baselines` in train.py so the default run is fast.
- **[4.6][LOW][E6] Feature-frame cache** — parquet keyed on (dataset hash, feature
  version); every entrypoint currently rebuilds ~163k rows of features from scratch.
- **[4.7][LOW][E7] Docs consolidation** — CLAUDE.md/AGENTS.md are near-duplicates (keep
  one canonical + pointer); README still says `legacy/` holds the R code (deleted);
  report/main.typ still cites stale numbers (also flagged under Housekeeping below).
- **[4.8][LOW][E8] Data-quality guards in fetch.py** — assert FDR merge coverage and
  duplicate-row counts after joins (the repo's own logging rule), replace the `iterrows`
  in `fetch_fixture_difficulty`, and move player identity from name-factorize to
  element-ID joins (also fixes silent player drops in live name-matching).

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

- **[RESOLVED 2026-07-06, negative + re-baseline done] Level recalibration & backtest re-baseline.**
  Both completed same day - see RESEARCH_LOG 2026-07-06 for the full matrix. Headlines: honest
  NNLS 1869, honest CatBoost 1856 (statistical tie; CatBoost kept for simplicity/cost), level
  calibration HURT (1800; cross-position budget reallocation, not the predicted transfer
  suppression). **The real current baseline is ~1870, not 1966** - the old headline was flattered
  by weight leakage. `--calibrate-level` kept in predict.py, off by default, as a documented
  negative result. Judge future changes on realized points, never MASE alone (demonstrated twice
  now).
- **[DONE 2026-07-06 - big win] Optuna tuning for CatBoost: 2107 realized points (+251 vs
  untuned, new record; see RESEARCH_LOG).** Params in fpl/models/tuned_params_*_catboost.json,
  auto-loaded by the position-aware fit_model. Follow-ups now open:
  - **[HIGH] Re-run `python -m fpl.model.train`** so production ensembles + the bake-off use the
    tuned params (saved models predate them).
  - **[MEDIUM] Tune lightgbm/xgboost too** so the registry ranking is default-luck-free, and
    re-check the bake-off (a tuned blend might beat tuned CatBoost).
  - **[LOW] Stability: more trials (50-100) and a second seed/window** before trusting 2107 as
    THE number rather than a very good draw.
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
