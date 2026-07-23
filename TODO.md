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
  hardcoded 152s now point at the same constant. Residual closed same day: tuning re-run
  under the cap; old-vs-new params a statistical tie on GW153-183 (2041 vs 2060, CI
  [-139, +140]); clean params adopted as production. Q2 formally unprovable and moot.
- **[1.3][DONE 2026-07-11][leakage A3] Shift by gameweek, not by row, in `features.py`.**
  Done (commit `d4ebdc2`): `enforce_gameweek_level_shift` broadcasts the round's
  first-row values of all 109 player-shifted columns within each (player, GW) group;
  sort made stable; tamper-based DGW guards added. Re-baseline run: standard-protocol
  backtest fell 2107 -> **2041** (-66, the measured size of the DGW leak).
- **[1.4][DONE 2026-07-11][method B2] Origin-based horizon backtest.** Done (commit
  `ac927c4`): `fpl.model.predict --origin-based` freezes form at each origin's deadline
  via the live snapshot path; `optimize.py` solves each GW from its own origin set.
  Measured once: origin-based scores **1936** vs standard 2041 — lookahead optimism
  +105, 95% CI [+21, +206], P(standard better)=0.991. Standing rule (RESEARCH_LOG
  2026-07-11): comparisons run the standard protocol vs 2041; deployment claims quote
  1936. Found in passing: live opp_* staleness, logged as 3.6.
- **[1.5][DONE 2026-07-11][method B1] One-shot confirmation backtest on GW191-221.** Run
  exactly once after 1.1-1.4 + the capped re-tuning certified the config: **1705**
  standard / **1499** origin-based (the window offered MORE raw points than GW153-183, so
  the ~355-point drop vs 2060 is winner's curse, not season scarcity - audit B1
  vindicated). **The window is now SPENT for selection, permanently.** Full ladder and
  standing rules in RESEARCH_LOG 2026-07-11.
- **[1.6][DONE 2026-07-11 (tool)][method B3] Uncertainty on realized-points comparisons.**
  Done: `python -m fpl.milp.compare_backtests runA.csv runB.csv` — paired per-GW
  moving-block bootstrap CI on the total points difference + binomial sign test, with
  unit tests (`tests/test_compare_backtests.py`). Used in anger 2026-07-11 on the
  standard-vs-origin gap (+105, CI [+21, +206]) and folded into HANDOFF's standing
  rules. The once-planned retroactive 2107-vs-2059 run is MOOT: both sides ran on
  pre-DGW-fix features and that baseline no longer stands (see RESEARCH_LOG).
- **[1.7][DONE 2026-07-11][metric B4] MASE scale consistent** — `train.py` (static split,
  bake-off, walk-forward) now uses per-position naive scales like `tuning.py`. MASE tables
  printed before 2026-07-11 are NOT comparable with new ones (see RESEARCH_LOG 2026-07-11);
  realized-points numbers are unaffected.
- **[1.8][LOW][scoring B5] Auto-subs + vice-captain activation in backtest scoring** —
  currently ignored, so absolute realized-points understate real FPL play equally for
  all configs. Either implement a simple auto-sub simulation in optimize.py's scoring
  block or document the omission in the report. *Effort: ~1 day or a footnote.*

### Cluster 2 — Forecast improvements (gate lifted 2026-07-11: baseline is 2086 since the 2026-07-16 xP zero-round mask)

Judge every item here on the standard-protocol realized-points MILP backtest vs **2086**
(GW153-183, capped-tuned params), with a 1.6 uncertainty interval — never on MASE movement
alone. Note the measured CI width on this window is ~±140 points: differences inside that
are ties. Deployment claims use the honesty ladder (RESEARCH_LOG 2026-07-11; ~1500/31 GWs).
GW191-221 is spent and must never be used for selection.

- **[2.1][v1 DONE 2026-07-11 - tie][C1] Dedicated minutes model.** v1 hurdle
  (`catboost_hurdle` registry member: P(min>0) × E[pts|played]) scored **2085 vs 2060 —
  a statistical tie** (CI [-128, +202], sign test 14-15), while sweeping every forecast
  diagnostic at every position (first diagnostic-sweeper to not LOSE points). Production
  unchanged; member kept in the registry. Open v2 ideas if revisited: 3-class minutes
  stage (0/cameo/60+), or cross-fitted P(played)/E[min] as features into the production
  regressor. See RESEARCH_LOG 2026-07-11.
- **[2.2][DONE 2026-07-16 - NEGATIVE][C2] Current-GW xP as a feature** — the stamping
  verification FAILED: vaastav scrapes xP (= API `ep_this`) after each round and FPL revises
  it post-match. Statistical checks looked plausible; the backtest scored an impossible 2915
  vs 2060, confirming the leak. Permanently unusable from this source. By-catch kept: all-zero
  dump rounds now masked to NaN before the lagged xP forms (2086 vs 2060, tie - correctness
  fix). Live-only idea: fetch `ep_this` pre-deadline in run_week (see Cluster 3 / 3.1).
  Full story: RESEARCH_LOG 2026-07-16.
- **[2.3][DONE 2026-07-18 - NEGATIVE][C3] LambdaRank experiment** — tested and CLEARLY
  lost: 1825 vs 2086 (CI [-421, -93], sign test 10-20), not a tie. The strong ranking
  hypothesis is weakened: even with an isotonic points-scale map, pure within-round
  ordering destroys ~12% of realized points — the level/tail information a regressor
  keeps carries real squad value. Code (`lgbm_rank` member + tests) lives on unmerged
  branch `exp/lambdarank`. Full story: RESEARCH_LOG 2026-07-18.
- **[2.4][DONE 2026-07-18 - production confirmed][C4] Tune LightGBM/XGBoost** — tuned
  (Optuna 50 trials/position, GW<=152 cap) and the bake-off re-run tuned-vs-tuned:
  `single:catboost` still wins at every position (tuned LGBM/XGB never get within 0.09
  MASE of CatBoost; best blend `top_k` also trails everywhere). No promotion, no MILP
  backtest needed. Params in gitignored `fpl/models/tuned_params_*_{lightgbm,xgboost}.json`
  (regenerate via `python -m fpl.model.tuning`). Run: `tuned_lgbm_xgb_bakeoff` in
  experiments/results.csv; RESEARCH_LOG 2026-07-18.
- **[2.5][LOW/PARKED][C5] Bookmaker odds features** — strongest exogenous signal in the
  football-prediction literature, but historical odds acquisition is a real data
  project. Park until 2.1-2.4 are exhausted.

### Cluster 3 — Live readiness for 2026-27 (needs 1.1; otherwise independent)

Do before the season opens on the FPL site; none of it affects backtests.

- **[3.1][DONE 2026-07-19 (code)][A4b] Use API availability in run_week** — implemented on
  `feature/api-availability`: `availability_multipliers` maps bootstrap-static `status` /
  `chance_of_playing_next_round` to a per-player factor (a=1.0; d=chance/100 else 0.75;
  i/s/u=chance/100 else 0.0), `apply_availability` scales every horizon GW's prediction
  (conservative: an out-today player is down-weighted across the whole horizon; re-assessed
  each run). Unit-tested with mocked bootstrap (`tests/test_availability.py`). Residual:
  like all of run_week's live path, cannot be exercised end-to-end until the 2026-27
  season opens on the FPL site (CLAUDE.md known limitation).
- **[3.2][DONE 2026-07-19 (code)][A4a] Current prices from the API** — implemented on
  `feature/live-prices`: `live_prices`/`apply_live_prices` override the snapshot's stale
  `value` with bootstrap `now_cost` (same 0.1m units) for every matchable player, snapshot
  price kept when unmatched. Buy-price-for-everyone simplification documented in the
  docstring (sell-price nuance stays with 3.4). Mocked-bootstrap tests alongside 3.1's in
  `tests/test_availability.py`. Same live-verification residual as 3.1.
- **[3.3][DONE 2026-07-19 (code)][A4c] DGW handling in live horizon** — implemented on
  `feature/live-dgw`: `build_team_fixture_map` now returns EVERY fixture per team and
  `build_future_predictions` emits one prediction row per fixture via a team-merge (a DGW
  player carries both fixtures' predicted points; the optimizer already sums per
  (player, GW), same as backtest CSVs). Per-fixture opponent/home/FDR;
  `fixture_difficulty_next3` averages across all fixtures in the window; blank-GW teams
  still drop out. Mocked DGW test in `test_live_snapshot.py`. Same live-verification
  residual as 3.1/3.2.
- **[3.4][DONE 2026-07-20 - policy kept at 2][D1] Update FT rule to the 5-banked era** —
  investigated with a full 2x2 backtest (cap {2,5} x horizon {3,5}): cap 5 costs ~100-135
  points at both horizons (noisy forecasts make predicted-indifferent banking deferrals
  realized-costly; cap 2's use-it-or-lose-it pressure is protective). PO decision: `Q_bar`
  is a banking POLICY, kept at 2 (`config.MILP_MAX_FREE_TRANSFERS`); baseline stays 2086.
  Shipped: `--initial-ft` above the policy cap is now honored instead of infeasible
  (max(cap, initial_ft) bound + unit test), and the sell-price + chips-disabled
  conventions are documented in CLAUDE.md. RESEARCH_LOG 2026-07-20.
- **[3.5][LOW][D2] FT/chip accounting test** — the solver's FT logic is duplicated in
  the Python state-rollover (`optimize.py:373-386`); a 5-6 GW synthetic test asserting
  the FT trajectory closes the silent-divergence class the constraint test doesn't cover.
- **[3.6][DONE 2026-07-11] Live opp_* features are stale.** Fixed same day it was found:
  the helper moved to `features.team_form_asof` (shared by the origin-based export and
  run_week), and `build_future_predictions` now maps each future GW's actual upcoming
  opponent to that opponent's trailing form instead of carrying the snapshot's previous-
  fixture values. Guarded by a mocked-fixture test in `test_live_snapshot.py`.

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
