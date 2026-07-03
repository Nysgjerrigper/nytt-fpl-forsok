# Per-Player Forecasting Model Selection: Feasibility & Design Plan

Status: **planning only, not implemented**. This document explores whether the
current per-position ensemble architecture (`fpl/model/train.py`,
`fpl/model/ensemble.py`) should be replaced or augmented with Venter & van
Vuuren's (2024) approach of selecting the best-performing forecasting method
**per individual player** rather than pooling one blended model across an
entire position. See `RESEARCH_LOG.md`'s 2026-07-03 entries for the baseline
experiments (Croston, naive drift, SES, Holt, Theta, pooled AR(1), per-player
ARIMA) that motivated this document, and its explicit flag that per-player
selection was "the real structural idea worth exploring later ... not
attempted yet."

## 1. Feasibility check: how much history does a player actually have?

Queried `Datasett/master_dataset.csv` directly (113,270 rows, 1,628 unique
`player_id`s across all seasons in the dataset, 2022-23 through the in-progress
2025-26 / GW115+).

**All-time rows per player** (one row = one gameweek the player was in a squad,
whether they played or not):

| stat | value |
|---|---|
| n players | 1,628 |
| mean | 69.6 rows |
| median | 54 rows |
| p25 | 38 rows |
| p10 | 22 rows |
| p5 | 14 rows |
| players with <10 rows | 50 (3%) |
| players with <20 rows | 145 (9%) |
| players with <38 rows (< 1 season) | 343 (21%) |

By position (median rows / p25 rows): GK 53/38, DEF 52.5/38, MID 49/38, FWD
38/38 (n=187/542/719/215 players respectively). FWD is both the smallest pool
and has the least median history - consistent with FWD already being the
hardest position to forecast (RESEARCH_LOG's MASE > 1 finding for FWD).

**More important number - rows with `minutes > 0`** (i.e. the player actually
played; a squad-listed but unused player contributes almost no time-series
signal for a points forecaster):

| stat | value |
|---|---|
| n players (played at least once) | 1,134 |
| mean | 40.4 played rows |
| median | 28 played rows |
| p25 | 9 played rows |
| p10 | 2 played rows |
| p5 | 1 played row |
| players with <10 played rows | 285 (25%) |
| players with <20 played rows | 439 (39%) |

**Most decision-relevant number - prior history available for players active
in the CURRENT season** (GW_global >= 115, i.e. 2025-26, as of this writing):
841 distinct players appear in the current season. Of those:

- **293 (35%) have ZERO prior-season rows at all** - pure new entrants (fresh
  promotions, new signings, players who didn't have a top-flight FPL record
  before). There is no history to run per-player CV on at all for over a
  third of the player pool a live weekly run actually needs to score.
- Median prior history across all 841: 38 rows (one season). 25th percentile:
  **0 rows**. Only the top quartile has more than one full season (113+ rows)
  of backing history.

**Conclusion: per-player model selection is not feasible for a large minority
to plurality of the live player pool.** Rolling-origin CV needs enough folds
to reliably distinguish "method A is genuinely better for this player" from
noise - with typically-cited rules of thumb wanting at least ~15-20 held-out
points per candidate comparison to keep a rank-order decision from being
essentially a coin flip, and FPL's points being noisy/intermittent (many
zeros), the real requirement is probably higher than a plain regression
setting. A large fraction of the pool (leading candidates for "someone the
manager might actually want to transfer in" - new signings, promoted-team
players, breakout youngsters) fails even the lowest bar (10 played rows), let
alone a bar high enough to make CV-based method ranking trustworthy.

**Recommendation: a minimum-history threshold, with graceful fallback.**
Concretely:

- **>= 20 gameweeks of total history AND >= 10 rows with `minutes > 0`**:
  eligible for per-player model selection.
- Below that threshold: fall back to the existing position-level ensemble
  prediction unmodified. No per-player CV attempted - there just isn't enough
  signal to trust a per-player ranking.
- This means, on current data, roughly the bottom third-to-40% of the
  currently-relevant player pool never leaves the existing architecture. That
  is a feature, not a bug: it bounds the blast radius of a new, noisier
  mechanism to the players who actually have enough history to support it,
  and it fails safe on both a manager's most planning-critical picks (players
  they already own, who by definition have accumulated history) and the ones
  where a wrong per-player call would matter least to notice (single-gameweek
  punts on players with almost no track record anyway, where the existing
  position ensemble's cross-sectional features are arguably a *better* source
  of signal than that player's own thin history - see below).

## 2. Candidate model pool per player

This is the central design tension the task description already flags, so
being explicit about it up front:

- The **6 existing ML models** (`fpl/model/models.py`: LightGBM, Ridge,
  ElasticNet, Random Forest, Extra Trees, kNN) are **cross-sectional** - they
  fit shared coefficients/tree splits across every player at a position,
  using ~70 engineered features per row (`fpl/features.py`). A single
  player's own ~30-50 rows cannot refit one of these from scratch; there
  isn't remotely enough data, and doing so would defeat the entire reason
  they work (borrowing strength across hundreds of players who share
  positional scoring patterns). **These cannot be "reused per player" as
  literally retrained per-player models.**
- The **existing per-player time-series baselines** in
  `fpl/model/baselines.py` (Croston, naive drift, SES, Holt, Theta, and
  per-player ARIMA via `fit_predict_arima_per_player`) plus the pooled AR(1)
  (which is position-pooled, not per-player, so it's a weaker fit for this
  specific architecture but cheap to include) are **fundamentally per-player**
  already - each is computed independently per `player_id`, using only that
  player's own ordered history. These are the natural candidate pool for a
  literal "pick the best method for player X" mechanism.

**Resolution - the position-level ensemble prediction is itself one candidate
in the per-player race, not replaced by it.** This matches Venter's own
setup more closely than it first appears: their ~15 candidates included
several ensemble combinations of top individual methods, and the winning
choice per player was whichever method (individual or ensembled) had the
lowest historical MASE for that player specifically. Concretely, the
per-player candidate pool becomes:

1. `position_ensemble_pred` - the existing `PositionEnsemble.predict()`
   output for that player's rows (cross-sectional signal, already strong
   per CLAUDE.md's backtest numbers).
2. `ses_pred` - best-performing simple per-player baseline so far
   (RESEARCH_LOG: beats the ad-hoc rolling baseline at every position).
3. `theta_pred` - second-best per-player baseline, worth keeping in the pool
   since it won at some positions (DEF/MID) even though SES won overall.
4. `croston_pred` - kept in the pool *despite* losing pooled, precisely
   because the open question this document exists to answer is whether it
   wins for specific low-minutes/rotation-risk players even though it loses
   position-wide. Excluding it would beg the question.
5. `naive_drift_pred`, `holt_pred`, per-player `arima_pred` - kept as
   low-cost stragglers for completeness/robustness, even though none beat
   SES pooled; a per-player selector should be allowed to prove they're
   sometimes right rather than assuming they never are.
6. (Optional, cheap to add) a simple **average of {ensemble, SES, theta}** as
   an explicit ensemble-of-candidates entry, mirroring Venter's inclusion of
   combined variants, since their paper's own conclusion was that "the
   ensembled methods performed better than the individual forecasting
   methods" - worth testing whether that holds here too before assuming a
   single best method per player is the right unit of selection.

This keeps the cross-sectional ML machinery exactly as-is (no per-player
retraining of LightGBM etc.) while giving per-player selection something
real to choose between: whether *this specific player's own history* (via
one of the TS baselines) predicts them better than *what similar players at
their position tend to do* (the ensemble).

## 3. Selection mechanism

**Rolling-origin CV per player**, matching this project's existing
walk-forward philosophy (`fpl/model/train.py::walk_forward_evaluate`,
`fpl/model/predict.py::walk_forward_predictions`) rather than inventing a new
validation style:

- For each eligible player (>= 20 total rows, >= 10 played rows, per §1), take
  their full ordered row history up to "now" (the most recent GW strictly
  before the one being predicted - same leakage discipline as the rest of the
  pipeline).
- Split this into an expanding sequence of origins: for origin `k` (starting
  once at least ~10 rows of history exist), compute each candidate method's
  one-step-ahead forecast for row `k+1` using only rows `<= k`, and compare
  to the actual `total_points` at `k+1`. This reuses exactly the one-step
  recursive contract the baselines already implement
  (`fpl/model/baselines.py`'s per-player functions already produce
  `forecasts[i]` using only `values[0..i-1]` - no new leakage-safety code
  needed for those). For `position_ensemble_pred`, the equivalent is: what
  would the *already-trained-at-that-point* position ensemble have predicted
  for that player's row `k+1`? This requires either (a) re-running the
  ensemble's `.predict()` on that historical row's stored feature vector
  (cheap - the ensemble is already trained walk-forward in `predict.py`;
  reuse whatever ensemble was live as of that GW rather than retraining one
  per fold), or (b) accepting a slightly optimistic proxy (today's final
  ensemble backfit on that row) if (a) is judged not worth the engineering
  cost initially - flag which was used, since it affects how trustworthy the
  ensemble's slot in the per-player comparison is.
- Aggregate each candidate's per-player errors into that player's MASE
  (`fpl/model/metrics.py::mase`), using the **existing global/position-pooled
  scale** (`naive_lag1_scale`) as the denominator rather than a per-player
  scale - the metrics module already documents why a per-player scale would
  distort low-scoring players (their tiny naive-diff floor would blow up
  their MASE, making already-marginal players even harder to differentiate
  than they need to be for this purpose). Pick whichever candidate has the
  lowest per-player MASE across those folds as that player's forecaster for
  the upcoming GW.

**Refit/re-selection cadence:** match `predict.py`'s existing
`--retrain-every` pattern (default: every 4 GWs in the CLI default, every 1 GW
possible) rather than inventing a separate cadence knob. Per-player selection
is cheap enough (each candidate is O(n) per player, not a heavy model fit) to
re-run every GW without the performance concern that governs the ML
ensemble's retrain cadence - but for consistency and to avoid selection
"flapping" week to week on marginal-history players, tie it to the same
`retrain_every` boundary the position ensembles already use, so both layers
update in lockstep and a single retrain log line covers both.

## 4. Where this plugs into the existing pipeline

Goal per the task: **`fpl/milp/optimize.py` must not need to change at all**
- it only ever reads a `predicted_total_points` column from a CSV.

Concrete new code paths, none replacing existing ones:

- **New module, e.g. `fpl/model/player_selection.py`** (does not exist yet):
  - `eligible_for_selection(player_history_df, min_total=20, min_played=10) ->
    bool` - the §1 threshold check.
  - `rolling_origin_scores(player_series, candidate_fns) ->
    dict[candidate_name, mase]` - runs the §3 CV loop for one player's
    history against the candidate pool, returns a MASE per candidate.
  - `select_best_method(scores) -> str` - argmin, with a tie-break rule
    (prefer `position_ensemble_pred` on ties, since it's the
    already-validated default and per-player selection should have to
    *earn* deviating from it, not merely tie).
  - `build_predictions_column(df, ensemble_preds, baseline_preds_by_col,
    eligibility_mask) -> pd.Series` - combines per-player selections into one
    output column, falling back to `ensemble_preds` wherever
    `eligibility_mask` is False.

- **`fpl/model/train.py`**: no change to `evaluate_static_split`'s existing
  reported columns (keep the existing MAE/MASE table exactly as-is for
  continuity with RESEARCH_LOG's historical comparisons). Add a new,
  separate reporting block (not folded into the existing loop) that computes
  the per-player-selected column's MAE/MASE on the same GW77-107 test window,
  printed alongside the existing table so it's directly comparable without
  disturbing the existing output other tooling/history may depend on.

- **`fpl/model/predict.py::walk_forward_predictions`**: after the existing
  per-position `models_cache[pos].predict(...)` call that fills
  `predicted_total_points`, add one new optional code path (e.g. gated by a
  new `--per-player-selection` CLI flag, default off) that, for each test GW,
  looks up each player's current best-method selection (recomputed at the
  same `retrain_every` cadence as the position models, per §3) and overwrites
  `predicted_total_points` for that player's row with the selected method's
  prediction - for ineligible players, the line is a no-op (the existing
  ensemble prediction already sits in that cell). This is the one piece that
  guarantees `optimize.py` needs zero changes: the output CSV's schema and
  column names stay identical; only how one column's values were sourced,
  per-row, changes.

- **`fpl/model/ensemble.py`**: no changes needed. `PositionEnsemble` keeps its
  current role as the cross-sectional layer; the new module treats it as an
  opaque candidate, not something to modify internally.

## 5. Evaluation plan

Same walk-forward backtest discipline as the rest of this project - no new
validation philosophy invented for this feature:

1. Run `fpl.model.predict` walk-forward over the same **GW77-107** window
   used for every prior comparison in this project (CLAUDE.md's backtest
   reference point), twice: once with `--per-player-selection` off (today's
   behavior, the control) and once on.
2. Feed both predictions CSVs into `fpl.milp.optimize` **unchanged**, same
   `--horizon`, same chip settings, so the only variable between runs is the
   forecasting layer.
3. Compare final `actual_total_points` sum against the existing reference
   points: old LSTM 1526, LightGBM-only 1811, 6-model position ensemble 1900.
   Per CLAUDE.md's explicit caution, this is the number that matters - not
   MAE/MASE in isolation, since "MAE improvements don't always translate 1:1
   into more actual points once the optimizer is in the loop."
4. Also report the per-player-selection MASE table from §4's new reporting
   block, split by eligible vs. ineligible players, so a result can be
   diagnosed (e.g. "helped for eligible players but total points didn't move
   because they're a small slice of the squad" vs. "made things worse even
   for eligible players").
5. Sanity-check candidate-selection distribution: what fraction of eligible
   players end up selecting the ensemble vs. each TS baseline. If the
   ensemble wins for ~95%+ of eligible players, the added machinery is
   producing a near-no-op at high engineering cost - a legitimate negative
   result worth reporting plainly rather than declaring victory on a MASE
   rounding difference.
6. Go/no-go threshold: only worth merging into production if it beats 1900
   actual points on the same window by a margin clearly outside run-to-run
   noise (the MILP is deterministic given its inputs and CBC's solver
   behavior, so re-running twice with identical inputs should already show
   how much run-to-run variance to expect before treating any gain as real).

## 6. Effort/risk estimate

**Size:** medium - new module (~150-250 lines), one new CLI flag, one new
reporting block in `train.py`. Not a rewrite of any existing production
class; `PositionEnsemble`, `models.py`, and `optimize.py` are all untouched.
Comparable in scope to the baseline-methods work already done in
`baselines.py` (all per-player forecasters this project needs already exist;
this is a selection layer on top, not new forecasting methods).

**What could go wrong:**

- **Overfitting the method choice itself.** Selecting "the best of 6+
  candidates" per player, from a rolling-origin CV with realistically only
  ~10-30 folds per eligible player (per §1's data), is itself a
  high-variance decision - Venter's paper had comparable per-player sample
  sizes and this is an inherent risk of the approach, not something unique
  to this codebase's implementation. §1's threshold reduces but does not
  eliminate this; even at 20+ rows, a per-player "SES wins by 0.02 MASE over
  the ensemble" result is likely noise, not a durable pattern. Mitigate with
  the tie-break-toward-ensemble rule (§4) and, if pursued, a stability check
  (does the same player's selection change every retrain cycle, or is it
  sticky? Flapping selection = noise, not signal).
- **Computational cost.** ~1,600 players x ~6-7 candidates x rolling-origin
  folds, repeated every `retrain_every` GWs across a 31-GW backtest window.
  The TS baselines are cheap (`baselines.py`'s functions are all O(n) numpy
  loops, not iterative optimizers) except per-player ARIMA
  (`fit_predict_arima_per_player`, statsmodels MLE fit) which is already
  flagged in that file as too slow to refit per-row - for a rolling-origin
  CV doing this *per fold per player* it would need to be fit once per player
  per retrain cycle (not per fold), or dropped from the per-player pool
  entirely if that's still too slow at ~1,600 players x 8 retrain cycles.
  Recommend timing a small pilot (e.g. just the MID position, one retrain
  cycle) before committing to the full backtest matrix.
- **A large fraction of the pool this document exists to help (new
  signings, promoted-team breakouts) is exactly the group excluded by the
  eligibility threshold** (§1: 35% of the current season's active players
  have zero prior history). The players a manager most wants better
  forecasts for - an unproven player who might break out - are structurally
  the ones this technique can't touch. This bounds the plausible upside: even
  a large per-player win for eligible players caps out affecting at most
  ~60-65% of the live pool, and probably less once the >=10-played-rows
  filter is applied on top.

**Recommendation: cautious go, small-scope pilot first, not a full commit.**
Given (a) every position-pooled TS baseline tried so far lost to the existing
ensemble (RESEARCH_LOG), (b) the explicit, reasonable hypothesis that pooling
was masking per-player heterogeneity remains untested, but (c) the feasibility
data in §1 shows a meaningful chunk of the live player pool can't support this
technique at all - the right next step is the small pilot described in §5/§6
(one position, e.g. MID - largest player pool, moderate history - over a
shorter GW window first) to get a real MASE/points signal cheaply, before
investing in the full 4-position x 31-GW backtest matrix. If the pilot shows
the ensemble wins for the overwhelming majority of eligible players (per §5.5),
that alone would be grounds to stop rather than scaling up - a well-reported
negative result at small scale is more valuable here than a slow, expensive
negative result at full scale, consistent with how every other baseline in
this project has been tested and honestly reported so far.
