# Research log

A running log of major decisions, experiments, and their results for this project - what was
tried, why, and what actually happened, so results are reproducible and don't need to be
re-derived from git history or re-litigated later. Newest entries at the top. See `CLAUDE.md`
for the current architecture; this file is the history of *why* it looks that way.

## 2026-07-20 - FT banking cap: site rule is 5, solver policy stays 2 (TODO 3.4 closed, branch `feature/ft-cap-5`)

FPL raised the bankable-free-transfer cap from 2 to 5 in 2024-25; the solver still used the
thesis-era `Q_bar=2`. Updating it looked like a one-line rule fix - the backtest said
otherwise. Full 2x2 on GW153-183, same production predictions CSV throughout:

| | horizon 3 | horizon 5 |
|---|---|---|
| FT cap 2 | **2086** (baseline) | 1977 |
| FT cap 5 | 1951 | 1886 |

Cap 5 costs ~100-135 points at BOTH horizons (each pairwise CI straddles zero, but sign
tests run 18-13 and 21-9 against it, and h5/cap5 vs the h3/cap2 baseline is a clear loss,
P=0.996). A longer lookahead does not rescue it. Mechanism: the solver banks whenever the
predicted gain from moving now looks smaller than the option value of waiting - but with
noisy forecasts a predicted-indifferent deferral is realized-costly on average, and cap 2's
use-it-or-lose-it pressure was accidentally protective. Third instance of the project's
regularization-beats-flexibility pattern (single CatBoost > blends; `MILP_GAP_REL=0` >
faster loose gap).

**Decision (PO, 2026-07-20): `Q_bar` is a banking POLICY, not rule compliance - banking
less than the site allows is legal play.** `config.MILP_MAX_FREE_TRANSFERS` stays 2 (the
constant and the evidence now live in config.py); the baseline stays 2086, no re-baselining.
One real code change shipped with this: a live squad arriving with MORE than 2 already
banked (legal since 2024-25) is now honored - `optimize.py` bounds the FT state at
`max(policy_cap, --initial-ft)` instead of going infeasible, guarded by a unit test.
Verified byte-identical squad selection vs the 2086 baseline run under the final config.
Also documented (CLAUDE.md known limitations): the sell-price simplification and the
chips-disabled backtest convention. Runs: `ft_cap5_h3_backtest`, `ft_cap5_h5_backtest`,
`ft_cap2_h5_backtest` in experiments/results.csv.

## 2026-07-18 - Tuned LightGBM/XGBoost: single:catboost survives tuned-vs-tuned (TODO 2.4 closed, branch `exp/tune-lgbm-xgb`)

TODO 2.4 / audit C4: the registry ranking had been tuned-CatBoost vs default-everything-else,
leaving open whether CatBoost's bake-off win was real or a tuning artifact. LightGBM and
XGBoost were tuned per position (Optuna, 50 trials each, expanding-window CV under the
`TUNING_TRAIN_MAX_GW=152` cap, cap recorded in `_meta`) and the full train.py comparison +
combination bake-off re-run tuned-vs-tuned.

**Verdict: production confirmed, nothing changes.** `single:catboost` wins the bake-off at
all four positions (eval-half MASE 0.724/0.638/0.711/0.736 for GK/DEF/MID/FWD); the best
blend (`top_k`) trails everywhere, and tuned LightGBM/XGBoost never get within 0.09 MASE of
CatBoost at any position (e.g. DEF: catboost 0.651, tuned lightgbm 0.747, tuned xgboost
0.744). The Clemen-1989 pattern holds even after leveling the tuning playing field. Since
the bake-off winner did not change, no MILP backtest was triggered - the production config
is byte-identical to what produced the standing 2086.

Params live in the gitignored `fpl/models/tuned_params_{POS}_{lightgbm,xgboost}.json`
(regenerate: `python -m fpl.model.tuning`). Run: `tuned_lgbm_xgb_bakeoff` in
experiments/results.csv. Cluster 2 is now exhausted except parked 2.5 (bookmaker odds).

## 2026-07-23 - P(haul) captaincy tilt is a wash, NOT wired in (negative result)

Question from the PO: the bucket distribution yields a P(haul >=10) per player; captaincy is a
pure right-tail bet, so does ranking captains by an upside *tilt* of the point forecast beat
ranking by expected points? Rule considered for production: `E[pts] x (1 + P(haul))`. Gate
agreed up front: wire it in ONLY if the tilt wins.

Walk-forward GW153-183 (retrain every 4, `coarse5`, tuned-vs-tuned), MID+FWD pool, per-GW
top1_capture (1.0 = always captained the week's true top scorer):

| model | cap_ev (rank by E[pts]) | cap_haul (rank by P(haul)) | cap_tilt (E[pts]x(1+P(haul))) |
|---|---|---|---|
| catboost_hurdle_bucket | 0.5262 | 0.5543 | 0.5562 |
| catboost_bucket | 0.5562 | 0.5693 | 0.5412 |
| catboost_regression (production point forecast) | 0.5431 | - | - |

**Verdict: wash, gate fails, NOT wired in.** The tilt does not win in any robust sense: it
*helps* the hurdle model (0.5262 -> 0.5562) but *hurts* the plain bucket model (0.5562 ->
0.5412), and the single best captaincy number (0.5562) is reached by plain-bucket E[pts] with no
tilt at all. A real upside signal would help regardless of which base model produced E[pts]; a
sign-flip across base models over a ~31-GW MID+FWD pool is the signature of noise, not signal.
Production captaincy stays on E[points].

Consistent with the earlier bucket verdicts: the framing wins forecast metrics (here the buckets
are far better calibrated, bias ~0.00 vs the regression's -0.53) but not *decisions* - the same
mean-vs-decision gap seen at 2026-07-08. Note that entry's 2059-vs-2107 MILP numbers predate the
2026-07-11 re-baseline and are superseded as absolute values; the direction of its verdict is
what carries, not the levels. Caveat on this run: one scheme (`coarse5`), one window, no CI -
enough to fail an "only if it wins" gate, not enough to call the tilt actively harmful. The
`cap_tilt`/`cap_haul` diagnostics stay in `probabilistic_buckets.py` as a documented dead end.
(The prototype branch `probability-of-loss-2026-27`, which first raised this idea, was archived
as tag `archive/probability-of-loss-2026-27` - superseded by that module.)

## 2026-07-18 - Origin-based anchor refreshed post-mask-merge: 1906 (supersedes 1916)

The 2026-07-16 xP zero-round mask merge left the origin-based (deploy-honest) anchor at its
pre-mask 1916 value. Re-ran it: `fpl.model.predict --origin-based --horizon 3` over GW153-183
(retrain-every 4, production `single:catboost`) into `fpl.milp.optimize --horizon 3`, same window.
Masked run scores **1906**.

**A reproduction note, not a regression.** Regenerating the pre-mask origin-based squad through
today's MILP from the unchanged 2026-07-11 predictions file (`preds_origin_gw153_183_retuned.csv`)
gives **1903**, not the originally-logged 1916 - a 13-point drift. Likely cause: the same-day
CBC -> HiGHS default swap (see the entry below) - both solvers prove optimality, but on tied
objectives they can select different, equally-optimal squads with different *realized* points
(the MILP's proven-optimal guarantee is on predicted E[pts], not on the realized total). Paired
`fpl.milp.compare_backtests` on the regenerated pair: 1903 vs 1906, CI [-135, +151], sign test
10-15 (6 ties), p=0.424 - a clear tie, same verdict the standard protocol reached (2086 vs 2060).

**New origin-based anchor: 1906** (retires 1916). Origin-based headline for reporting purposes
(~1500/31 GWs live expectation) is unaffected - that number comes from the separate frozen
GW191-221 confirmation, not this window.

Run logged as `xp_zero_round_mask_backtest_origin_based` in experiments/results.csv.

## 2026-07-16 - MILP solver swap CBC -> HiGHS: ~20% faster, identical squads; gap-tolerance shortcut rejected (branch `worktree-exp+milp-solver-speed`)

Question from the PO: can the MILP be made faster with a newer solver/library? Infrastructure
change only - no modeling content, no new baseline. All runs below are the standard GW153-183 /
horizon-3 window on the same predictions file (`preds_std_gw153_183_retuned.csv`, the post-DGW-fix
capped-retuned vintage, whose standing MILP total is 2060).

| Solver configuration | Wall time | Realized points | Squad decisions |
|---|---|---|---|
| CBC (old default, exact) | 153.0s | 2060 | reference |
| **HiGHS via highspy (new default, exact)** | **120.9s** | **2060** | byte-identical to CBC (diffs <= 1e-9, float noise) |
| HiGHS, 8 threads | no change | 2060 | identical |
| CBC, 8 threads | no change | 2060 | identical |
| HiGHS, gapRel=0.001 | 97.7s | **2020** | 18 squad cells differ |

Findings:
- **HiGHS is the win, and it is exact.** Both solvers prove optimality, so the squad output is
  the same by construction; HiGHS just proves it ~20% faster overall and, more usefully, caps
  the worst gameweek at 7.7s where CBC occasionally stalls (12.7s GW163, ~11s GW154 in repeat
  runs). Solver landscape check: HiGHS is the strongest open-source MILP engine reachable from
  PuLP as of 2026 (SCIP/CBC behind it, CP-SAT would need an integer-coefficient rewrite);
  commercial solvers (Gurobi, free academic licence) are the only step up, worth ~another
  order of magnitude if solve time ever actually matters.
- **Threads do nothing.** The per-GW problems (~700 players x 3 GWs, ~20k binaries) have too
  shallow a branch-and-bound tree to parallelize; measured no benefit at 8 threads on either
  solver. `--threads`/`config.MILP_THREADS` kept (default 0) since it cost nothing to expose.
- **Do not buy speed with the MIP gap (negative result).** gapRel=0.001 (objective within 0.1%
  of optimal per solve) saved a further ~20% wall time but changed real decisions: 2020 realized
  points, -40 vs exact. A 0.1% slack on a ~90-point objective is ~0.09 predicted points per
  solve - enough to flip marginal transfer/captaincy picks, and over 31 rolling solves those
  flips compound (the rolling horizon feeds each GW's squad into the next). `--gap-rel` exists
  for ad-hoc use but `config.MILP_GAP_REL` stays 0.
- Model build time is NOT the bottleneck (~0.3s/GW vs 2-7s solving), so PuLP-level rewrites
  (lpDot, variable pruning) were not pursued. Per-GW solve/build split now printed by
  `optimize.py` for free future diagnosis.

Changes: `--solver {cbc,highs}` / `--threads` / `--gap-rel` CLI args on `fpl.milp.optimize`
(defaults from `config.MILP_SOLVER="highs"` / `MILP_THREADS=0` / `MILP_GAP_REL=0.0`), `highspy`
added to requirements. `run_week.py` and the test suite inherit the new default through
`parse_args`; pytest green (59 passed) on HiGHS.

## 2026-07-18 - LambdaRank v1: clear loss (1825 vs 2086), ranking hypothesis weakened (branch `exp/lambdarank`, NOT merged)

TODO 2.3 / audit C3: three "better metrics, fewer points" episodes suggested the MILP consumes
within-GW RANKING rather than point levels. This tested that mechanism directly: new registry
member `lgbm_rank` (`models.LambdaRankScorer`) - LightGBM lambdarank with one query group per
GW_global round (models are per-position, so groups are effectively (GW, position)), points
clipped to integer relevance 0-15 with LINEAR label gains (default 2^rel gains would let one
haul dominate a round's gradients), and an isotonic regression mapping ranker scores back to
the points scale so the MILP's absolute-scale terms (transfer penalty, chip thresholds) stay
meaningful. Group label threaded through `models.fit_model(gw=...)` exactly like the hurdle's
`minutes=`.

**Realized points: a CLEAR LOSS - 1825 vs 2086** (-261, 95% CI [-421, -93],
P(baseline better)=0.998, sign test 20-10). Unlike most rejected candidates this is not a tie:
optimizing pure within-round ordering destroys ~12% of realized points even after the scale is
restored monotonically. The instructive read: the ranking hypothesis in its strong form is now
WEAKENED - if ranking were all the MILP consumed, this should have at worst tied. Level and
tail information a regressor preserves (how MUCH better a captaincy pick is, not just that it
is better) evidently carries real squad value; consistent with `top1_capture` improvements
alone never having translated into points either.

Verdict: negative result, logged per convention. Branch NOT merged (registry member + tests
live on `exp/lambdarank` only); production unchanged at `single:catboost`, baseline stays
2086. Run: `lambdarank_v1_backtest` in experiments/results.csv;
`preds_lambdarank_gw153_183.csv` + `squad_selection_W153-183_SHL3_lambdarank.csv` artifacts.

## 2026-07-16 - Current-GW xP is a confirmed post-match leak (TODO 2.2 closed NEGATIVE); zero-round mask kept (branch `exp/current-gw-xp`)

TODO 2.2 asked whether the current GW's raw `xP` could be used as a feature, gated on verifying
the vaastav stamping first. Verdict: **no, permanently - the column is stamped post-match.**

**The statistical sniff tests passed, which is the instructive part.** Per-season checks looked
pre-match-plausible everywhere: played-row corr(xP, points) ~= 0.52 in every season (nowhere near
deterministic), and among nailed starters (prev 3 games all 60+ min) who surprisingly got 0
minutes, only ~49% had xP < 1 - roughly what legitimate pre-deadline injury news could achieve.
On those checks the feature shipped... and scored an **impossible 2915 vs 2060** (+855, CI
[-1131, -588], 94 pts/GW - superhuman) on the standard GW153-183 backtest. Root cause, confirmed
in vaastav's own docs: `xP` is the FPL API's `ep_this` scraped AFTER each round completes, and
FPL revises that field post-match. A partially-revised pre-match estimate is exactly the kind of
leak that passes correlation checks and fails only the end-to-end points test - the strongest
vindication yet of the mandatory-backtest rule. Leaky prediction/squad artifacts deleted;
`add_xp_features` now documents the leak as confirmed (not just "conservative choice"), and a
unit test asserts nothing unshifted from xP can reach the feature list.

**By-catch, kept: the xP zero-round mask.** Verification found whole rounds where every row's
xP is exactly 0 - unfilled dump rounds, not forecasts (27 of 38 GWs in 2025-26; 1-3 per season
2020-25, including 2024-25 GW22 inside the backtest window). Those fake zeros were polluting the
existing lagged `xP_prev`/`xP_roll3`. They are now masked to NaN before the lagged forms are
built. Backtest: **2086 vs 2060 - a tie** (CI [-112, +187], P(mask better)=0.72, sign test
16-13); kept on data-correctness grounds, same logic as the DGW fix. If merged, 2086 becomes the
new comparison baseline (PO call at merge time).

**Live-only survivor idea (Cluster 3):** `ep_this` fetched from bootstrap-static BEFORE the
deadline in `run_week` is legitimately pre-match - it just can't be backtested from this dataset.
Logged as a possible companion to 3.1's availability filtering.

Runs: `xp_current_leak_probe` (artifacts deleted) and `xp_zero_round_mask_backtest`
(`preds_xpmask_gw153_183.csv`, `squad_selection_W153-183_SHL3_xpmask.csv`) in
experiments/results.csv.

## 2026-07-11 - Minutes hurdle v1: statistical tie on points (2085 vs 2060), sweeps every forecast diagnostic (branch `exp/minutes-hurdle`)

First Cluster 2 experiment against the repaired measurement system (TODO 2.1, audit C1 -
"probably the largest single gain"). New registry member `catboost_hurdle`
(`models.TwoStageHurdle`): E[pts] = P(minutes>0) x E[pts | played] - an EXACT decomposition,
since a player who never comes on scores exactly 0 and 59% of rows are 0-minute rows. The
participation classifier (CatBoost Logloss) absorbs the zero mass; the regression head
(CatBoost MAE, capped tuned params) trains on played rows only, so its median is no longer
dragged to 0 by benchwarmers. fit_model gained a `minutes=` training label for this
(participation is not recoverable from points: played rows can score 0 or negative).

**Realized points (the decision metric): TIE.** GW153-183 / horizon-3, standard protocol:
hurdle **2085** vs production catboost **2060** (+25, 95% CI [-128, +202], P(hurdle
better)=0.71, sign test 14-15 - dead even). Under the standing rule this does not clear the
promotion bar; `PRODUCTION_WEIGHT_STRATEGY` stays `single:catboost`.

**Forecast diagnostics (walk-forward, same CSVs): a clean sweep at every position.** Better
RMSE, less negative bias, better total calibration (e.g. DEF 0.565 vs 0.439), better
within-GW Spearman, better top1_capture (DEF 0.444 vs 0.351) - with slightly worse MAE,
exactly as theory predicts once a model stops fitting the zero-inflated median. Notably,
this is the THIRD model to sweep the diagnostics, and the first that did NOT lose realized
points doing it (level calibration lost 56, buckets lost 48). The mean-vs-median trap cost
appears to be shrinking as the measurement system improves.

**Verdict:** honest tie - kept in the registry per project convention (a real blend/bake-off
candidate; the bake-off will flag it if it starts winning), production unchanged. Follow-up
candidates if 2.1 is pushed further: a 3-class minutes stage (0 / cameo / 60+, capturing
appearance-point structure), or feeding P(played)/E[min] as FEATURES to the production
regressor (cross-fitted). Runs: `minutes_hurdle_v1_backtest` in experiments/results.csv;
`preds_hurdle_gw153_183.csv` + `squad_selection_W153-183_SHL3_hurdle.csv` artifacts.

## 2026-07-11 - Capped re-tuning (Q2 closed) + the one-shot GW191-221 confirmation: the honesty ladder 2060 / 1916 / 1705 / 1499

Final act of the audit's Cluster 1. Two things ran, in order: (1) the CatBoost tuning was
re-run under the new `TUNING_TRAIN_MAX_GW=152` cap (Optuna 50 trials/position, seed 0, cap
recorded in `_meta`), and (2) with the config thereby final, the frozen 2025-26 window
GW191-221 was spent - run exactly once, both protocols, per the standing one-shot rule.

**Q2 resolution (was the 2026-07-06 tuning capped?): unprovable, and now moot.** The capped
re-search found different params at every position, but the feature pipeline changed between
the two searches (DGW fix), so this cannot distinguish "the old search was uncapped" from
"the features moved". What CAN be said: on the GW153-183 backtest the old and new params are
a statistical tie (2041 vs 2060, +19, 95% CI [-139, +140], P=0.49) - so whatever the old
search saw, it bought no measurable window advantage. The retuned params are adopted as
production regardless, because only they are provably clean; every params JSON now carries
its own cap in `_meta`.

**The honesty ladder.** Each step removes one source of optimism from the headline number,
all with the identical production config (tuned single:catboost, horizon-3 MILP):

| # | Window | Protocol | Points | Optimism removed at this step |
|---|--------|----------|--------|-------------------------------|
| 1 | GW153-183 (selection window) | standard | **2060** | - (the comparison baseline) |
| 2 | GW153-183 | origin-based | 1916 | -144: lookahead (form after t in the horizon) |
| 3 | GW191-221 (never-selected) | standard | 1705 | -355 vs #1: selection/winner's curse |
| 4 | GW191-221 | origin-based | **1499** | both: the honest live expectation |

**The confirmation window says the winner's curse was real and large.** GW191-221 actually
OFFERED more raw points than GW153-183 (28,052 vs 25,374 summed over all players; top-100
concentration similar at 11,760 vs 11,571), so the ~355-point drop is not a thin season - it
is what dozens of selection decisions on one window cost in optimism, just as audit B1
predicted (garden of forking paths). The lookahead gap replicated on the fresh window (-206
vs -144; same direction, similar scale given CI widths ~+/-100), which independently
validates the origin-based protocol.

**Standing rules from here (supersedes this morning's 2041/1936 entry):**
- Comparison baseline for model/feature A/B: **2060** (standard protocol, GW153-183,
  retuned params), always with a `compare_backtests` CI.
- Deploy expectation to quote externally: **~1500 points per 31 GWs** (the selection-free,
  deploy-protocol number). 1916 remains the deploy-protocol figure on the selection window
  and the ladder as a whole is the thesis-grade result: it decomposes exactly how a
  research backtest overstates live performance (2060 -> 1499 = -27%).
- **GW191-221 is SPENT.** It must never be used to choose a model, feature, or
  hyperparameter. If a future confirmation is needed, freeze GW222+ or wait for 2026-27.

All four runs in `experiments/results.csv` (`retuned_rebaseline_*`,
`confirmation_oneshot_*`); squad CSVs under `fpl/squad_selections/` (`*_retuned.csv`,
`*_confirm.csv`). Params backup of the superseded set kept for the session only - the live
JSONs in `fpl/models/` are the clean ones.

## 2026-07-11 - Re-baseline after DGW-leak fix + origin-based protocol: 2107 -> 2041 (comparison) and 1936 (deploy expectation)

Audit items 1.3 + 1.4 landed together (commits `d4ebdc2`, `ac927c4`) and the GW153-183 /
horizon-3 window was re-run once under both protocols, exactly so there would be ONE new
standing baseline rather than two corrections in sequence. Same tuned single:catboost
configuration throughout; the only changes are the measurement system's.

| Configuration | Realized points |
|---|---|
| pre-fix standard protocol (the old standing baseline) | 2107 |
| **standard protocol, DGW-leak-free features (new comparison baseline)** | **2041** |
| **origin-based protocol, same features (deploy-honest expectation)** | **1936** |

**What the -66 says (A3).** Removing the double-gameweek same-round leakage cost the
standard backtest 66 points (-3.1%). That was never real skill: the second fixture of a DGW
was being predicted with the first fixture of the same round already in its form features,
and DGW players are exactly the ones the MILP loads up on. The audit called the effect
"small but systematic"; measured, it is 2/3 the size of the entire Optuna tuning gain -
worth remembering whenever a leak is dismissed as minor.

**What the -105 says (B2).** With form frozen at each origin's deadline (the live
information set, via the same `build_live_snapshot` path run_week uses), the identical
model + MILP scores 105 fewer points than the standard walk-forward (95% block-bootstrap CI
on the gap [+21, +206], P(standard better) = 0.991; sign test 20-11 GWs, p = 0.150 - the
first comparison in this project to carry an interval, per the new
`fpl.milp.compare_backtests`). The standard protocol lets the MILP's t+1/t+2 lookahead
terms see forms that include GW t's outcomes; live never can. The CI excluding zero says
the lookahead optimism is real, not window noise.

**Standing decision rule from here.**
- Model/feature comparisons: judge on the STANDARD protocol vs **2041**, with a
  compare_backtests CI - the protocol is identical on both sides of any comparison, so it
  stays fair, and it is half the compute of the origin-based run.
- Deployment claims ("what would this have scored live"): quote **1936** / the
  origin-based protocol. The +105 lookahead gap was measured once and does not need
  re-measuring per experiment; re-check it only if the horizon logic or freezing mechanics
  change.
- All pre-2026-07-11 realized-points numbers (1526/1811/1900/1966/1870/1880/2059/2107)
  were produced on leaking features and are not comparable with post-fix numbers. Their
  RELATIVE conclusions stand (both sides of each comparison leaked identically).

Both runs logged in `experiments/results.csv` (`audit_rebaseline_standard_protocol`,
`audit_rebaseline_origin_based_protocol`); prediction dumps in `experiments/predictions/`
(`preds_std_gw153_183_dgwfix.csv`, `preds_origin_gw153_183_dgwfix.csv`), squad CSVs in
`fpl/squad_selections/` (`..._dgwfix.csv`, `..._origin_dgwfix.csv`). Note: the planned
retroactive CI on the old 2107-vs-2059 bucket verdict is now moot - both sides of it ran on
pre-fix features and the baseline they compared against no longer stands; if the bucket
question is ever reopened it starts from a fresh 2041-baseline run.

## 2026-07-11 - Audit Cluster 1, first batch: live-path parity, tuning cap, comparison CIs, per-position MASE (branch `fix/audit-cluster1`)

First four measurement-system repairs from `AUDIT_2026-07-11.md` (items 1.1/1.2/1.6/1.7 in
TODO.md). No modeling change and no new backtest number in this batch - these fix HOW results
are produced and judged.

**A1 - the live path now runs the validated configuration.** `run_week.py` used to hand-roll
its model fit: members fit WITHOUT `position=` (so the Optuna-tuned params - the entire +251
gain behind 2107 - never loaded live) under a default `"nnls"` strategy (the 12-member blend
that lost the bake-off at every position). Live 2026-27 would have run a ~1856-level system
while every documented number said 2107. Now: `config.PRODUCTION_WEIGHT_STRATEGY`
(`single:catboost`) is the single definition of the production model, consumed as the default
by both `fpl.model.predict` and `fpl.run_week`, and both fit through one shared code path,
`train.fit_position_ensembles` (position-aware). The parallel never-loaded artifact path was
deleted outright (`train_final_ensembles`, `PositionEnsemble.save/load`, stale
`fpl/models/<POS>.*` files): three competing definitions of "the production model" became one.
`fpl.model.train`'s bake-off now prints a loud warning if its empirical winner ever disagrees
with the config constant. Note: all historical BACKTEST numbers (1966, 2107, ...) are
unaffected - they always ran through predict.py with the strategy passed explicitly; the skew
was live-only, which is why it never showed up in any logged experiment.

**A2 - the tuner can no longer validate on the backtest window.** `tune_position` folds are
now capped at `config.TUNING_TRAIN_MAX_GW` (152, the GW before the standing GW153-183 window)
by default, matching the discipline the bucket tuner already had; the bucket module's
hardcoded 152s now reference the same constant. The cap is recorded in the saved params JSON
under a `_meta` key (stripped by `models._tuned_params` before the constructor splat), so a
params file now documents its own provenance. OPEN: whether the 2026-07-06 tuning run
actually respected GW<153 (PO question Q2) - today's saved `tuned_params_*_catboost.json`
predate `_meta` and cannot prove it either way. If the answer is no/unknown, re-run tuning
under the cap and re-verify 2107.

**B3 - realized-points verdicts get an uncertainty interval.** New
`python -m fpl.milp.compare_backtests runA.csv runB.csv`: paired per-GW differences between
two squad_selection CSVs over the same window, moving-block bootstrap (default block 3, to
respect squad-carryover autocorrelation - an iid bootstrap understates the variance of the
total) 95% CI on the total difference, plus a binomial sign test. Unit-tested
(`tests/test_compare_backtests.py`), including that the block bootstrap is wider than iid
under autocorrelation. A synthetic demo with independent per-GW noise of realistic size shows
even an 86-point gap failing to clear the CI - the standing 48-point bucket verdict badly
needs this check. The retroactive 2107-vs-2059 comparison waits on regenerated per-GW CSVs
(the 1.3/1.4 re-baseline will produce them); from now on, any promote/demote decision on
realized points should quote this CI, not just the point difference.

**B4 - MASE now uses per-position scales everywhere.** `train.py` (static split, bake-off,
walk-forward) previously divided every position's MAE by one POOLED naive scale, while
`tuning.py` used per-position scales - so per-position MASE claims were really MAE rankings
in disguise ("FWD hardest, MASE 1.07 > 1" said FWD has high MAE against a shared denominator,
not that the model loses to FWD's own naive forecast), and numbers were not comparable across
modules. **MASE tables printed before 2026-07-11 are not comparable with those printed
after.** Realized-points numbers, MAE columns, and all relative model rankings WITHIN a table
are unaffected (a per-position scale is a constant within each row).

All 68 tests pass (4 new: params-JSON `_meta` round-trip, tuning-cap application, and the
compare_backtests suite). Remaining Cluster 1: 1.3 (DGW leakage GW-level shift), 1.4
(origin-based horizon backtest) - to be landed together with ONE re-baseline - then 1.5
(frozen 2025-26 confirmation window, needs PO approval Q3) and 1.8 (auto-subs footnote).

## 2026-07-10 - Bucket-count sweep, dedicated classification tuning, and ensemble tests: 8 buckets confirmed, tuning helps, every ensemble is a negative result

Branch `exp/bucket-scheme-sweep`. Three questions from the PO, all answered on the same GW153-183
walk-forward (retrain every 4, tuned-vs-tuned against the production CatBoost regression):
(1) how many probability buckets are best, and does moving the binary cutpoint from <=2/>2 to
<=6/>6 help; (2) does the winning scheme improve with hyperparameters tuned for classification
instead of borrowed from the regression; (3) do ensembles help - bucket x regression blend,
"binomial for captaincy + 8 buckets for forecasting", bucket-model self-ensembles.

**1. Bucket count: at least 5, and 8 is enough.** New parametric schemes (`binary2/6/9`, `tri3`,
`int13`) next to the existing `coarse5/fine8/fine10`, all swept in one shared retrain loop
(`--schemes` now takes a list; predictions dumpable via `--save-predictions` for post-processing):

| scheme (buckets) | spearman | ev_rmse | total_calibration | cap_ev |
|---|---|---|---|---|
| fine8 (8) | 0.7015 | **1.9177** | 1.0175 | **0.5655** |
| fine10 (10) | 0.7015 | 1.9178 | 1.0205 | 0.5655 |
| int13 (13) | 0.7015 | 1.9181 | 1.0222 | 0.5655 |
| coarse5 (5) | **0.7017** | 1.9200 | 1.0168 | 0.5562 |
| tri3 (3) | 0.6926 | 1.9721 | 0.9964 | 0.5300 |
| binary2 (<=2/>2) | 0.6939 | 1.9792 | 0.9918 | 0.4869 |
| binary6 (<=6/>6) | 0.6877 | 2.0839 | 1.0241 | 0.5412 |
| binary9 (<=9/>9) | 0.6772 | 2.1885 | 1.0436 | 0.5243 |
| catboost_regression | 0.6755 | 2.0726 | 0.5439 | 0.5431 |

5 through 13 buckets are identical to the third decimal - the distribution's value saturates
fast, and past 8 buckets there is nothing left to buy. Below 5 the loss is real at every
position. Moving the binary cut from 2 to 6 does NOT rescue the binomial framing for
forecasting: one cut anywhere throws away too much of the outcome scale. **fine8 declared the
scheme winner** (best ev_mae 1.0228, tied-best everything else, simplest of the tied group).
Every multiclass scheme beats the regression on ranking, RMSE, and calibration - consistent
with the 2026-07-08 entry, and still subject to its MILP caveat.

**2. Dedicated classification tuning helps (where it aimed).** New `--tune` mode Optuna-tunes
the CatBoost CLASSIFIER per position (multiclass logloss objective, expanding-window CV,
GW<=152 only so the eval window stays unseen; saved as `tuned_params_<POS>_bucket_fine8.json`,
loaded via `--bucket-tuned-params`). The tuner consistently chose shallower/slower-learning
trees than the regression params (depth 4-5, lr ~0.02-0.03 vs depth 5, lr 0.014). Walk-forward,
fine8, dedicated vs borrowed params: logloss 0.8985 vs 0.9035, ev_mae 0.9977 vs 1.0228, ev_rmse
1.9130 vs 1.9177, bias -0.008 vs +0.020, spearman 0.7020 vs 0.7015, haul AUC 0.8876 vs 0.8846 -
a small, uniform improvement on every proper/accuracy metric (exactly what tuning optimized).
Captaincy capture moved the other way (cap_ev 0.5000 vs 0.5655) - see 3c.

**3. Ensembles: all negative, reported per this project's practice.**
- **(a) Blend with the regression** (w*bucket + (1-w)*regression, w chosen on GW153-167 by
  spearman, scored only on GW168-183): chosen w=0.6 gets spearman 0.7071 vs 0.7062 for pure
  bucket - a tie - while calibration degrades (0.80 vs 0.98) and RMSE is worse than pure bucket
  (1.9076 vs 1.8742). The regression's conditional-median bias only dilutes the well-calibrated
  bucket mean. No reason to blend.
- **(b) Self-ensembles**: equal-weight E[pts] across schemes (fine8+coarse5+fine10+int13) is a
  no-op - their forecasts are near-perfectly correlated (spearman 0.7016, identical captaincy).
  CatBoost+LightGBM within fine8 gains noise-level ranking (0.7027 vs 0.7020) and loses
  calibration (0.90) and captaincy (0.4719), because default-params LightGBM is biased low
  (-0.22). Echoes the Clemen-1989 result already in CLAUDE.md: the best single model resists
  dilution.
- **(c) The "binomial for captaincy" hybrid is NOT confirmed.** PO hypothesis: rank captains by
  fine8 E[pts] * (1 + P(>threshold)) with the probability from a dedicated binary model. On
  borrowed-params fine8 the binary6 tilt looked good (top-1 capture 0.5749 vs 0.5655 base, and
  the 6-cut beat both the 2-cut and the haul-cut) - but the entire gain lives in GW153-167 and
  vanishes in the second half, and on the dedicated-tuned fine8 the same tilt HURTS (0.4963 vs
  0.5000). A dedicated binary9 haul model also has WORSE haul AUC (0.8779) than the P(>=10)
  implied by fine8's own distribution (0.8876), so the binomial model adds no tail information
  the multiclass doesn't already carry. Verdict: captaincy stays E[pts]-ranked; top-1 capture
  over 31 GWs is too noisy to certify any of these tilts, and nothing here replicates.

**Decision:** fine8 with dedicated tuned params replaces coarse5 as the standing bucket
configuration (forecasting-only view, per the 2026-07-08 decision); no ensemble layer is added.
All pooled rows are in `experiments/results.csv` (`bucket_scheme_sweep`,
`bucket_fine8_dedicated_tuning`, `bucket_regression_blend`, `bucket_captaincy_tilt`,
`bucket_selfensemble_cb_lgb`); prediction dumps regenerate via
`python -m fpl.model.probabilistic_buckets --walk-forward --schemes ... --save-predictions ...`
(new `fpl/model/bucket_ensembles.py` post-processes them). If the bucket view is ever to feed
the MILP again, re-run the 2026-07-08 backtest with the tuned fine8 first.

## 2026-07-08 - MILP backtest: bucket E[points] LOSES to the tuned regression on realized points (2059 vs 2107), despite winning every forecast metric

The decisive test the previous entry set up. Bucket E[points] won calibration, RMSE, ranking, and
captaincy in the walk-forward forecast eval - but the only thing that pays is realized MILP points, and
here it LOST. New `walk_forward_predictions_csv` in `fpl/model/probabilistic_buckets.py`
(`--export-predictions`) writes bucket E[points] in the exact predict.py CSV format; both models were run
through the *identical* GW153-183 / horizon-3 MILP in this one session, so the comparison is generated
under one setup with no cross-run drift:

| Configuration | Realized points |
|---|---|
| tuned CatBoost **regression** (`single:catboost`, production) | **2107** |
| tuned CatBoost **bucket** E[points] (`catboost_bucket`, `use_tuned`) | 2059 |

Both used tuned per-position params, both made exactly 1 transfer/GW, both used zero chips over all 31
GWs - so the 48-point gap (-2.3%) is purely *which players/captains* each model's numbers selected, not
transfer or chip behaviour.

**The baseline correction that matters.** The prior entry (and HANDOFF.md) said to compare against the
honest ~1870. That is the *untuned* baseline. The bucket model uses tuned params, so the honest
apples-to-apples baseline is the *tuned* regression, which I reproduced at exactly **2107** (matching the
2026-07-06 Optuna result to the point). Against 1870 the bucket model looks like a +189 triumph; against
the correct 2107 it is a 48-point loss. The ~1870 target in the handoff was the wrong yardstick for a
tuned model - any future bucket claim must beat 2107, not 1870.

**Why the forecast wins didn't convert (mean-vs-median trap, third instance).** The predictions confirm
the calibration story cleanly: bucket sum(pred)/sum(actual) = 1.017 (near-perfect mean), regression =
0.544 (MAE median-flattening roughly halves every total, per position: DEF pred 0.46 vs actual 1.02, FWD
0.65 vs 1.43). So the bucket forecast *is* the correctly-levelled mean the MILP nominally wants - yet
the MILP still scored fewer points with it. This is now the THIRD demonstration that better forecast
metrics don't buy squad points: (1) level-calibration scalar LOST 56 (2026-07-06), (2) CatBoost's 10%
MASE edge bought 0 (2026-07-06), (3) the bucket model's calibration+ranking+RMSE+captaincy sweep bought
-48 here. The MILP's transfer decisions are driven by *within-GW cross-player ranking of predicted
points*, and correct absolute levels don't help that ranking - the classification target (coarser signal
on the exact point value) evidently ranks the borderline transfer/captain candidates slightly worse than
the regression, even though it ranks the whole pool better on pooled Spearman.

**Verdict:** keep the bucket model as a *forecasting-only* research result and for its free P(blank)/P(haul)
distribution (the sibling branch's captaincy use), but do NOT route bucket E[points] into the production
MILP as a point-forecast replacement - it costs realized points. The conceptual claim ("one model yields
E[pts]+downside+upside") stands; the "and it also builds better squads" claim does not. Caveats as ever:
one window, one seed; but the direction agrees with two prior instances, so it is not treated as noise.
Reproduce: `python -m fpl.model.probabilistic_buckets --export-predictions --test-min-gw 153 --test-max-gw
183` then the horizon-3 MILP on that CSV vs `fpl.model.predict --weight-strategy single:catboost` on the
same window.

## 2026-07-08 - Walk-forward: bucket expected points BEAT the tuned production regression on every decision metric except raw MAE

The 2026-07-06 static split showed the bucket distribution is learnable; this answers the decision-grade
question: does bucket-derived E[points] hold up against the *tuned production* CatBoost regression under
honest conditions? New `evaluate_walk_forward` in `fpl/model/probabilistic_buckets.py`
(`--walk-forward`), same protocol as `fpl.model.predict`: every GW in 153-183 predicted using only
strictly-earlier data, retraining every 4 GWs. Fairness fix for the static split's caveat: the bucket
CatBoost reuses the tuned per-position regression hyperparams (loss swapped MAE -> MultiClass/Logloss,
`bootstrap_type=Bernoulli` pinned since CatBoost requires a sampling bootstrap when `subsample` is set),
so this is tuned-vs-tuned. Baseline is `fpl.model.models.fit_model("catboost", ...)` - the actual
production forecaster.

**Pooled results (GW153-183, coarse5 scheme):**

| metric | catboost_hurdle_bucket | catboost_bucket | catboost_regression |
|---|---:|---:|---:|
| ev_mae | 1.005 | 1.025 | **0.862** |
| ev_rmse | **1.915** | 1.920 | 2.073 |
| bias | **+0.004** | +0.020 | -0.532 |
| total_calibration | **1.004** | 1.017 | 0.544 |
| spearman_pos_gw | **0.703** | 0.702 | 0.676 |
| cap_ev (MID/FWD top1_capture) | 0.526 | 0.556 | 0.543 |
| cap_haul / cap_tilt | 0.554 / 0.556 | **0.569** / 0.541 | - |
| loss_auc / haul_auc | 0.867 / 0.889 | 0.866 / 0.886 | - |

**Reading it (mean-vs-median trap, again, but now in the buckets' favour):**
- The regression "wins" MAE only because MAE rewards predicting the conditional median - and it pays
  for that by under-predicting total points by 46% (total_calibration 0.544, bias -0.53). Bucket
  E[points] is a true probability-weighted mean: calibration 1.00-1.04 at every position, bias ~0.
  The MILP's absolute-scale transfer/chip logic consumes MEANS, so this is the scale it needs.
- On mean-aligned accuracy (RMSE) the buckets win: 1.915 vs 2.073.
- On ranking - the metric that survived every previous mean-vs-median dispute - buckets beat the
  regression at ALL FOUR positions (pooled Spearman 0.703 vs 0.676; DEF 0.640/0.602, FWD 0.758/0.744,
  GK 0.684/0.645, MID 0.730/0.711). Captaincy top1_capture also favours buckets (0.556 vs 0.543 on
  E[pts]; 0.569 ranking by P(haul) alone).
- The distribution itself (blank AUC 0.87, haul AUC 0.89) comes for free on top of that - no separate
  classifier needed, unlike the sibling `probability-of-loss-2026-27` branch's two-model setup.
- Hurdle vs plain bucket: effectively tied (hurdle marginally better distribution quality and
  calibration, plain marginally better captaincy). No reason to pay the hurdle's 2-model complexity yet.

**Verdict:** the bucket reframing is not just "extra information at equal accuracy" - it produces a
*better-ranked, correctly-levelled* expected-points forecast than production. Caveat: 80 quick-mode
iterations showed wildly different calibration (overshoot up to 2.3x), so these conclusions hold only
with the tuned params; don't judge this model family from `--quick` runs.

**Next question (the one that actually pays):** does a correctly-levelled mean change MILP realized
points? Route bucket E[points] into a predictions CSV, run the standard GW153-183/horizon-3 backtest,
and compare against the honest ~1870 baseline (per re-baseline commit `8081f20` on the sibling branch -
NOT the leaked 1966). Note the level-calibration experiment on the sibling branch (scaling the MEDIAN
forecast by a scalar) LOST points (1800); this is different - a genuine conditional mean, not a
rescaled median - but that history is why the MILP test must decide, not the forecast metrics.

## 2026-07-06 - Probabilistic bucket model bake-off

Implemented `fpl/model/probabilistic_buckets.py`: a standalone, forecasting-only bake-off that reuses the
normal feature pipeline but changes the target from one `total_points` number to a five-bucket distribution:
`<=0`, `1-2`, `3-5`, `6-9`, `>=10`. This is a real model-family change, not a CSV reshape: the estimators are
trained with multiclass/binary probability objectives and output `P(blank)`, `P(haul)`, and an implied
`expected_points` from the same distribution. It deliberately does not feed the saved production ensembles or
the MILP.

Models tested on the standard static split (train GW<=152, test GW153-183), per position: CatBoost multiclass
bucket, LightGBM multiclass bucket, XGBoost multiclass bucket, logistic multiclass bucket, CatBoost hurdle
bucket (`P(plays)` x bucket distribution conditional on playing), and LightGBM hurdle bucket. Scored with
multiclass log-loss/Brier, expected-points MAE/RMSE, blank/haul Brier + AUC, within position-GW Spearman, and
MID/FWD captaincy `top1_capture`.

**Result:** CatBoost owns the probabilistic quality. Pooled log-loss/Brier:
`catboost_hurdle_bucket` 0.6898/0.3605, `catboost_bucket` 0.6904/0.3607, then XGBoost 0.6980/0.3635; LightGBM
and logistic trail on distribution quality. CatBoost also has the best blank/haul ranking overall
(`haul_auc` ~0.887). The hurdle split helps very slightly on log-loss, but plain CatBoost bucket gives better
captaincy tilt (`E[points] * (1 + P(haul))`: 0.453 vs 0.416), so the pragmatic next candidate is probably
plain CatBoost bucket unless walk-forward says otherwise.

**Odd but useful signal:** logistic bucket has poor distribution quality but excellent captaincy capture
(`cap_ev`/`cap_haul`/`cap_tilt` all ~0.52). Treat this as a ranker anomaly to investigate, not as a reason to
prefer it as the calibrated probability model. Next step: add a walk-forward version and compare the implied
`expected_points` against production CatBoost regression before considering any optimizer integration.

## 2026-07-06 - Optuna tuning: +251 realized points. Tuned CatBoost scores 2107, the new record

The "most embarrassing gap" (untuned registry) is closed for the production model, and it was
worth more than every other intervention this week combined. Setup: 30 Optuna trials per
position for CatBoost, expanding-window time-ordered CV, **tuned on GW<153 only** so the
hyperparameters can't absorb anything from the backtest window; tuned params saved to
fpl/models/tuned_params_<POS>_catboost.json and picked up automatically by the position-aware
fit_model. Then the identical honest GW153-183 / horizon-3 backtest:

| Configuration | Realized points |
|---|---|
| honest CatBoost, hand-set defaults | 1856 |
| honest NNLS blend | 1869 |
| old leaky headline | 1966 |
| **honest CatBoost, tuned** | **2107** |

+13.5% over untuned, +141 past even the leaky old number. The tuned params tell a consistent
story at every position: much LOWER learning rate (~0.013 vs default 0.05), shallower trees
(depth 4-6), similar-or-more iterations - i.e. the defaults were learning too fast and too
deep, overfitting recent noise. CV MASE also improved (e.g. FWD best-trial 0.677 vs ~0.85
untuned full-window - different eval windows, so indicative not exact).

Caveats logged honestly: one window, one seed, 30 trials; and for once accuracy and realized
points moved TOGETHER - no mean-vs-median paradox this time. Follow-ups: re-run
`python -m fpl.model.train` so the saved production ensembles pick up the tuned params (the
bake-off will also re-verify single:catboost vs blends under tuned params); consider tuning
lightgbm/xgboost so the registry comparison is fair; consider more trials/seeds for stability.

## 2026-07-06 - Backtest re-baseline: honest weights erase most of the old lead; calibration HURTS; CatBoost ties NNLS on points

The long-pending re-baseline after the blend-weight leakage fix, all on the standard
GW153-183 / horizon-3 window, all with the current 115-feature set unless noted:

| Configuration | Realized points |
|---|---|
| leaky NNLS, pre-fixture/minutes features (old headline) | 1966 |
| leaky NNLS + fixture/minutes | 1880 |
| honest NNLS (weights fit GW137-152, strictly prior) | **1869** |
| honest single:catboost | **1856** |
| honest catboost + level calibration (scalars GK 1.39 / DEF 2.02 / MID 1.65 / FWD 1.83) | 1800 |

Three conclusions, all honest-negative or sobering:

1. **The 1966 headline was substantially flattered.** Under honest weight discipline the same
   architecture scores 1869. The old number mixed weight leakage with a different feature set, so
   the exact decomposition is unknowable, but ~1870 is the real current baseline. All docs/report
   comparisons should use it going forward.
2. **Level calibration HURT (-56 vs raw CatBoost).** The suppressed-transfers hypothesis was not
   supported - both runs made ~1 transfer/GW. Mechanism: a per-position scalar preserves
   within-position ranking but reshuffles CROSS-position budget allocation (DEF doubled, GK only
   +39%), and that reallocation cost points. `--calibrate-level` stays in predict.py as a
   documented negative result, off by default.
3. **CatBoost's 10% MASE advantage bought zero realized points** (1856 vs 1869, a 13-point / 0.7%
   difference on a single window - noise). The mean-vs-median trap is real but roughly OFFSETTING
   here: CatBoost ranks better, NNLS is better level-calibrated, and the MILP nets out the same.
   Decision: keep `single:catboost` as production anyway - identical points for 1/12th the
   training cost and a much simpler system - but the "best forecaster" claim must be stated as
   "best MASE, equal realized points", not as a squad-quality win. Corollary: future modeling
   changes should be judged on the realized-points backtest (or at minimum the top1_capture /
   calibration diagnostics), never on MASE movement alone - this is now demonstrated twice
   (fixture/minutes 1966->1880, and here).

## 2026-07-06 - EXPLORATION (branch `probability-of-loss-2026-27`): reframe forecasting as probability-of-loss

Exploratory branch, not merged, forecasting-only (no MILP work, per the standing priority). Question:
does modelling the *probability and magnitude of bad/good outcomes* - the finance downside-risk view
(Roy 1952 safety-first, VaR, CVaR, Sortino) - carry signal the mean point-forecast discards? FPL points
are an extreme case for it: 61% of player-GW rows are exactly 0, 59% didn't play at all, 68% of players
who DID play still returned <=2, and only 1.7% are hauls (>=10). Frequent small losses, rare large gains -
precisely where an expected-value forecast is least informative.

New parallel module `fpl/model/loss_probability.py` (like `probabilistic.py`, does NOT feed the optimizer).
Two per-position LightGBM binary classifiers on the same features / GW153-183 split as `train.py`:
BLANK (total_points<=2, the loss event) and HAUL (>=10, the captaincy upside event). Scored with proper
scoring rules (Brier, log-loss), ROC-AUC, and expected calibration error - NOT accuracy, since the ~86%
blank base rate makes "always predict blank" 86% accurate and useless (the intermittent-series trap that
motivated MASE, in classification form).

**Results - the events are highly predictable:**
- P(BLANK): AUC 0.84-0.88 every position, calibration error 0.01-0.05. The downside is very rankable
  (dominated by the minutes/nailedness signal - a P(loss) model is partly a re-expression of "will he play").
- P(HAUL): AUC 0.82-0.86 *despite* 0.8-2.6% base rates - the rare right tail is rankable well above chance.

**Results - it changes the captaincy decision (the point of the exercise).** Per-GW top1_capture over the
MID+FWD pool (actual points of your #1 pick / the week's true best):
| ranking | top1_capture |
|---|---|
| E[points] (what production does) | 0.365 |
| P(haul) alone | 0.360 |
| blend E[pts]x(1+P(haul)) | **0.429** |
P(haul) *alone* ties the mean forecast, but tilting the mean forecast by upside probability captures ~17%
more of the actual top-scorer points. Spearman(E[pts], P(haul)) = 0.937 - high but not 1.0, i.e. genuine
independent upside signal. This is direct evidence for the mechanism behind the unresolved 1966->1880
fixture/minutes regression (2026-07-04): the MILP maximises *expected* points and is blind to the upside
tail; a probability-of-haul term is exactly what a variance-aware decision would consume.

**Caveats (why this is a probe, not a verdict):** one static window, not walk-forward; one untuned blend
form; captaincy only; the E[pts] baseline here is a plain LightGBM (not production CatBoost), so 0.365 is
internally-consistent-but-not-the-production number. Promising enough to justify the fuller options below,
not settled. Next candidates mapped, none yet built: (a) a hurdle model P(plays)xE[pts|plays] to handle the
59% non-appearance zeros explicitly; (b) ordinal/multiclass over point buckets giving E[pts], P(loss) and
P(haul) from one model; (c) reading P(loss) off the existing quantile module's CDF; (d) a safety-first /
CVaR MILP objective (deferred - MILP is parked).

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
