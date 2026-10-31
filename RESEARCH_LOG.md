# Research log

A running log of major decisions, experiments, and their results for this project - what was
tried, why, and what actually happened, so results are reproducible and don't need to be
re-derived from git history or re-litigated later. Newest entries at the top. See `CLAUDE.md`
for the current architecture; this file is the history of *why* it looks that way.

## 2026-10-31 — Bounded Codex research workflow pilot

Registered one 180-second selection diagnostic on observed GW141–142, matched CatBoost MAE/RMSE 50-tree recipes, seed 0. The existing runner completed 16 measured fits and 8 native-optimal solves in 28.63s. RMSE improved forecast RMSE/calibration but reduced corrected net points: standard 125 versus 154 (-29); origin 120 versus 140 (-20). Requested bootstrap block 3 was clamped to sample length 2; the resulting zero-width intervals are degenerate and do not quantify generalization. Engineering success; scientific screening failed. No production change, final-holdout claim or repeat trial.

Hypothesis/configuration, preregistration link, primary-session skeptical review, reproduction commands, compact verified metrics and raw-byte manifest: [pilot report](experiments/fpl_workflow_pilot_2026-10-31/REPORT.md). Existing `fpl.experiment` rows are preserved in that directory; original negative evidence remains intact.

## 2026-08-11 - Optional PyTabKit dependency lane: RealMLP/TabM available; TabR incomplete

### Integration execution note (fresh Python 3.11.15 shared environment)

The Position-Specialist integration executor ran `pytest tests/ -q` (178 passed) and
generated the complete 56-command tournament plan. It also verified the exact generated
TabR selection command through its first trial: it fails closed with
`OptionalModelDependencyError` because this environment lacks both `faiss` and `skorch`.
An isolated `n_epochs=1`, seed-17 probe did not reproduce the earlier Python-3.14.6 neural
smokes: RealMLP exited after PyTorch initialization without returning predictions, while
TabM returned `ValueError: Training without validation set is currently not implemented`.
These are operational availability outcomes, not negative model results; no expert was
substituted, no tuned-artifact manifest was created, and no selection/backtest/promotion
evidence was produced. The historical macOS arm64/Python-3.14.6 evidence below remains
separate from this fresh-environment result.

### Supervisor follow-up: available-set protocol remains incomplete

The selection-stage CLI regression was added and the full test suite then passed with
179 tests. A reusable isolated Python 3.14.6 environment was revalidated with
PyTabKit 1.7.3, faiss-cpu 1.15.0, skorch 1.4.0, and declared Optuna 4.9.0. With an
explicit validation fraction, bounded CPU RealMLP and TabM fit/predict probes both
returned finite predictions. The single permitted bounded TabR attempt reached epoch-0
validation but returned no prediction, so TabR is recorded unavailable/incomplete and
is not substituted.

The explicit available research family is `catboost_mae`, `catboost_rmse`,
`lightgbm_l2`, `lightgbm_l1`, `lightgbm_huber`, `xgboost_squared_error`,
`xgboost_absolute_error`, `hist_gradient_boosting_absolute`,
`hist_gradient_boosting_squared`, `extra_trees_tuned`, `linear_svr_tuned`, `realmlp`,
and `tabm`. Its generated plan has 52 selection tune commands and is expressly
non-promotable because the registered TabR family member is unavailable. The first
unmodified command, `python -m fpl.model.tuning --position GK --model catboost_mae
--stage selection --seed 0 --train-max-gw 136`, completed two of 50 four-fold trials
(14.39s and 6.43s) before the execution session ceiling terminated it; no artifact was
written. At the observed pace it needs roughly 6--12 minutes. Do not reduce its trials
or timeout; resume it in a runtime that permits one complete registered study. No real
OOF selection, finalist/MILP artifact, CI, or promotion result exists.

### Persistent-session update: first real selection artifact completed

The exact unchanged `GK/catboost_mae` generated command subsequently completed all 50
four-fold trials in a persistent session. Its winning selection-CV MASE was
0.6775657717. The generated artifact is
`fpl/models/tuned_params_GK_catboost_mae.json`, with file SHA-256
`e7ffb57b6a9c85cf5c73f1b201d9b16a079fbd43e1af661228a23d881dc4f0ad` and embedded
provenance hash `81a581b8e5f9054c9846f4e6a3b6c6ed8e5cb8c2f9d457a6a5d016b132842086`.
Validation confirmed position GK, model catboost_mae, seed 0, stage selection, four
folds, `time_budget_seconds=3600`, and discovery cutoff 136. This is one tuning
artifact only; it is non-promotable and must not be used for selection until every
member of the declared available set has a matching artifact and the full causal chain
is completed.

This was a bounded dependency smoke, not a model comparison or promotion test. On macOS arm64 with
Python 3.14.6 and PyTabKit 1.7.3, RealMLP completed CPU fit/predict in 13.08s (`n_epochs=1`, seed 17)
and TabM completed in 1.41s. TabM's same-seed repeat produced maximum absolute prediction difference
0.0. These verify that RealMLP and TabM can be used as opt-in research candidates; they say nothing
about forecast accuracy, realized points, or production suitability.

`faiss-cpu==1.15.0` installs through pip, so the previous conda-only FAISS assumption is stale. TabR
then exposed an undeclared `skorch` requirement. With `skorch==1.4.0`, its bounded run reached epoch-0
validation but did not yield a prediction before it was stopped. **Verdict: TabR remains unavailable and
incomplete; no substitute is authorized.** `requirements-research.txt` now declares `skorch==1.4.0`;
repeat a bounded TabR fit/predict check before any study.

## 2026-08-11 - Position-specialist MoE methodology audit PASS; no tournament result or promotion

The remediated research-only interface passed its methodology audit. This is **not** an experimental
result: beyond the separately logged dependency smoke and synthetic audit validation, no model training,
real tuned artifacts, tournament, MILP backtest, confidence interval, or promotion decision is recorded.
Production remains `single:catboost` unless the full registered promotion gate passes.

The runtime season-aware workflow is selection-stage tuning capped at discovery -> fail-closed tuned-artifact
manifest -> causal OOF/frozen selection -> hash-bound finalist/control artifacts -> promotion. The current
chronology is GW<=136 discovery, GW137-152 selection, and GW153-183 chips-disabled horizon-3 standard/origin
evaluation; code structurally rejects the SPENT GW191-221 window. The final family is only a complete
four-position map and, optionally, a causal MID gate. The gate uses only deadline-known `mins60_rate_roll5`,
a training-only naive-lag-1 MASE scale, and validated OOF provenance.

Promotion requires positive standard and origin totals versus CatBoost, standard CI lower bound above
zero, origin CI lower bound at least -40, positive realized-point differences at seeds 0/1/2, and a
Holm-adjusted one-sided paired sign-test pass at alpha 0.05. If both permitted candidates pass, the
simpler one wins. Holm uses exact paired sign-test p-values (ties dropped); moving block bootstrap is
limited to realized-points CIs and is not a null p-value. RealMLP/TabM have passed optional-dependency
smokes; TabR is unavailable/incomplete despite declared `skorch==1.4.0` and needs a completed prediction
run. A rebuilt, ignored/untracked dataset has 162,981 rows through GW228. Integration must replace this
status with actual commands, artifacts, verification output, and results if a study runs.

## 2026-07-23 - Live readiness: first real 2026-27 end-to-end rehearsal; registered-player filter added

Modeling is PARKED (PO decision, same day - every candidate since the re-baselines tied). The 2026-27
season opened on the FPL API (GW1 deadline 2026-08-21, 555 elements), so `fpl.run_week` was exercised
end-to-end for REAL for the first time - fresh-build mode, horizon 3: dataset refetch, retrain,
live snapshot, API fixtures/availability/prices, MILP - all optimal solves.

**Finding (fixed):** the live pool contained 264 phantom players (of 710) with dataset history but no
2026-27 registration - outbound transfers/retirements the availability scaling structurally cannot
catch (the API has no status for a player it does not list). New `run_week.filter_to_registered`
restricts the pool to bootstrap-static's element list (the purchasable universe, matched by code ==
player_id); re-run confirms 446/446 pool players registered and a phantom-free squad.

**Documented, not fixed (structural):** the pool covers 17 of 20 teams - promoted-club players and
new signings have no dataset rows, and the models cannot forecast debutants. Their rows accumulate as
the season is played. Continue-mode (`--team-id`) remains untested until the PO registers a team.

## 2026-07-23 - Minutes-model v2 (3-class hurdle): 2053 vs 2057 - dead tie, production unchanged

**Hypothesis (TODO 2.1 v2):** v1's binary hurdle pools cameos (1-59') and full games (60'+) into one
E[pts | played] regressor, but those regimes have structurally different conditional distributions
(a cameo is capped near the 1-point appearance floor; the 60' boundary gates the second appearance
point and clean-sheet eligibility). A 3-class minutes stage - P(DNP)/P(cameo)/P(full), each playing
class with its own CatBoost regressor - should fit two homogeneous targets instead of one mixture.

**Result (standard protocol, GW153-183, single:catboost_hurdle3 vs single:catboost at 2057):**
**2053 - the deadest tie yet**: diff -4, 95% CI [-151, +135], sign test 15-15 (p=1.000). Same story
as v1 (2085 vs 2060, also a tie): minutes-aware decomposition changes nothing the MILP can monetize
on this window. The likely reason: the minutes-projection features (start_rate_roll5,
mins60_rate_roll5) already carry the participation signal into the plain regressor, so an explicit
probability stage is redundant rather than wrong.

**Decision:** production unchanged (single:catboost). `catboost_hurdle3` merged into the registry as
a documented negative result next to `catboost_hurdle` (project convention). The remaining v2 idea -
cross-fitted P(played)/E[min] as FEATURES - is now unlikely to pay: two hurdle variants in a row
suggest the signal is already in the features. Parked. Run: `minutes_hurdle3_backtest` in
experiments/results.csv; artifacts `experiments/preds_hurdle3_gw153_183.csv` +
`squad_selection_hurdle3_gw153_183.csv`.

## 2026-07-23 - Origin-based baseline refreshed post-identity-fix: 1880 (tie with retired 1906)

Follow-up to the element-code identity adoption (same day, above): the origin-based deploy-honest
number 1906 was computed under name-based identity, so it no longer described the shipped
configuration. Re-ran `fpl.model.predict --origin-based --horizon 3` GW153-183 (retrain-every 4,
single:catboost, capped-tuned params) through the MILP: **1880** vs the retired 1906 - a statistical
tie (diff -26, 95% CI [-138, +140], sign test 11-19, p=0.200; retired CSV recovered from git
history at b2de8f4 for the paired comparison). **1880 is now the standing origin-based baseline.**
The standard-vs-origin gap under the current config is 2057-1880 = 177 points of lookahead optimism,
consistent with the +105-to-144 range measured on earlier configs. Run:
`origin_baseline_refresh_codeid` in experiments/results.csv; artifacts
`experiments/preds_origin_codeid_gw153_183.csv` + `squad_selection_origin_codeid_gw153_183.csv`.

## 2026-07-23 - Element-code player identity + fetch join guards (TODO 4.8): tie, recommend adopt on correctness

**Hypothesis:** name-based identity (`pd.factorize(name)`) both merges distinct players sharing a
name and splits one player across spelling variants; joining vaastav's per-season `element` id to
the permanent FPL `code` (players_raw.csv, same field as bootstrap-static's `code`) fixes identity,
and letting the live path match API players by code removes the silent name-mismatch drops.

**Change:** `fetch.fetch_player_codes` + `assign_player_ids` (player_id == FPL code; name-factorize
fallback offset to 100M for rows without a code - zero such rows in practice); join guards in
`clean_season` (FDR merge `validate="many_to_one"`, coverage logging, hard failure if player_code or
opponent coverage < 0.99); `fetch_fixture_difficulty` vectorized (iterrows removed); `run_week`
matches availability/prices/squad picks by `el["code"]` directly (name normalization deleted).

**Identity impact (dataset rebuilt, 162,981 rows unchanged):** 2,083 name-identities -> 1,960 real
players. 125 players were SPLIT across spellings before (form history fragmented mid-career - their
rolling features restarted from NaN at every spelling change); 4 names wrongly MERGED two people
(Alvaro Fernandez, Josh King, Kaine Kesler Hayden, Yegor Yarmolyuk - beyond the hand-patched Ben
Davies). Code coverage 100%.

**Backtest (standard protocol, GW153-183, single:catboost, capped-tuned params):** **2057** vs
baseline **2086** - a statistical tie: diff -29, 95% block-bootstrap CI [-134, +55],
P(new better)=0.22, sign test 11-18 (p=0.265). Well inside the window's ~±140 noise band.

**Recommendation (pending PO):** adopt on data-correctness grounds, same precedent as the
2026-07-16 xP zero-round mask (also a tie, kept because the data is objectively right). The live
path additionally becomes strictly more robust (id-equality matching cannot silently drop players).
On adoption the standard-protocol baseline becomes **2057**; the origin-based number should be
refreshed at the next natural opportunity. Run: `element_code_identity_backtest` in
experiments/results.csv; artifacts `experiments/preds_codeid_gw153_183.csv` +
`squad_selection_codeid_gw153_183.csv` (worktree).

## 2026-07-23 - Subagent delegation infrastructure: pinned models, quality-gate rule, config drift-guard test

Not a modeling experiment - a development-infrastructure change (PO-initiated, cost control).
Subagents previously inherited the main session's model (Fable/Opus), so delegated mechanical
work was billed at top-tier rates. Landed in three commits on `main`:

- **`.claude/agents/implementer.md`** (Sonnet, medium effort): spec-driven code changes,
  refactors, test scaffolding. Its prompt embeds the repo's binding rules (English-only,
  shift(1) leakage guard, SimpleImputer wrapping, no inline magic numbers, pytest before done,
  never commits). **`.claude/agents/searcher.md`** (Haiku, low effort): read-only lookup and
  grep-and-summarize, with the global-GW-counter warning baked in. New CLAUDE.md section
  "Subagent delegation (cost control)" documents the split; `CLAUDE_CODE_SUBAGENT_MODEL` env
  var noted as an optional hard ceiling but deliberately not set (it would force the Haiku
  searcher up to Sonnet).
- **Quality-gate rule** added to CLAUDE.md: the main session is mission-giver AND verifier -
  it must review the implementer's actual diff and re-run `pytest tests/` itself, and
  spot-check searcher claims, never accepting a subagent summary as proof.
- **Validation:** both agents exercised on real tasks. Searcher correctly located all
  `PRODUCTION_WEIGHT_STRATEGY` usages (grep-verified, ~45k Haiku tokens). Implementer
  delivered `tests/test_config_strategy.py` from a written spec (~47k Sonnet tokens): a
  drift guard asserting the constant is a non-empty string, that `single:<name>` names a
  `FACTORIES` key, and that non-single strategies match `ensemble._WEIGHT_FITTERS` (the
  actual `fit_weights` dispatch table, asserted programmatically rather than hardcoded).
  Suite grew 105 -> 108, all green; diff independently reviewed before merge.

No modeling code touched; backtest baselines unaffected.

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


## 2026-10-02 - Recover causal exploratory screening across double gameweeks

Hypothesis: the previous exploratory run stopped because fixture rows were wrongly
required to be unique player-round rows, rather than because a model failed.
Recovered the four-model screen in `fix/branch-recovery`: fixture-level fitting and
prediction are preserved, while forecasts and actuals are summed per player round.
MASE now uses summed training rounds strictly before each refit. Ambiguous fixture
identities fail before fitting; combiner weights use only earlier OOF outcomes.

The original explicit parameters and seed 0 were reused for catboost_mae,
catboost_rmse, hist_gradient_boosting_squared and lightgbm_l2. The preserved GW231
snapshot SHA is `62f8f1d31d47a5f859f6d73819c9e631a59de346fed7de2fd5877b994a2ccc98`.
Rows above GW152 were removed before feature construction. OOF covers GW137-152;
after four warmup weeks, identical shared rows are scored on GW141-152.
The completed panel has 51,004 member rows, including 3,604 DGW member rows.

| Position | CatBoost MAE-loss MAE | MASE | RMSE | Bias | Total calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| GK | 0.522655 | 0.566999 | 1.552765 | -0.239078 | 0.675916 |
| DEF | 0.830619 | 0.589740 | 2.129114 | -0.517243 | 0.463817 |
| MID | 0.786539 | 0.605238 | 2.088439 | -0.455474 | 0.612536 |
| FWD | 0.819309 | 0.572396 | 2.291841 | -0.532962 | 0.536173 |

This member has the lowest MASE at every position among this screen's strategies,
but substantially underpredicts total points. The diagnostics demonstrate the
mean/median tradeoff; they do not establish squad quality. Full diagnostics, ranking
and top1_capture, weights, correlations, member OOF, predictions, and byte hashes are
under `experiments/branch_recovery_2026-10-02/screen/`. The result was logged through
`fpl.experiment.log_result` in this worktree's `experiments/results.csv`, with dirty
code explicitly identified and every source-file hash preserved.

Verdict: exploratory screen complete; realized MILP points and CI NOT ASSESSED.
No comparison to 2057/2088 is claimed. Production remains single:catboost.
Independent verification: 202 tests passed; dataset, code and output hashes,
shared rows, and causal cutoffs all checked. Existing imputer warnings reflect
historically absent defensive-contribution features; no missing stats were fabricated.


### 2026-10-02 — Engineering recovery and documentation reconciliation

Uncommitted `fix/branch-recovery` draft; production remains `single:catboost`.
Ported packaging, frozen provenance/cache and crash containment selectively, hardened
read/mutation admission, and corrected autosubs/vice/chip/Free Hit output behavior.
Fresh-build bank conservation uses equality. No model fitting or parameter promotion
was part of these replays; GW232+ remains outside research.

Independent validation: 243 passing tests; full frozen cached/uncached feature equality;
offline wheel and isolated import; 44 tuned JSON files and the provenance manifest
verified byte-identical inside the wheel. Same-input GW153-183 horizon-3 chip-disabled
MILP comparison used corrected scoring and optimal solves throughout. Standard control
and candidate both scored 2088; origin both scored 1901. Each candidate-minus-control
difference is 0 with block-bootstrap 95% CI [0,0] (10,000 draws, block 3, seed 0).
Some degenerate bench choices changed, while starting lineup/captain/vice did not.
Historical origin 1900 becomes 1901 solely through legal GW181 autosub scoring of the
same selected decisions; this is not predictive improvement. Inputs, hashes, source
control, commands, logs, CSVs, CIs and package/cache verification are preserved locally
in `experiments/engineering_recovery_2026-10-02/`.

Reconciled stale handoff/TODO, duplicate/conflicting operating rules and obsolete
neural-readiness advice. Preserved prior instructions, dated audits and paired report
PDF/source documents under `docs/archive/`; no historical research data was discarded.
Forecast-audit guards and provenance-bound Optuna resume remain separate open tasks.


Follow-up guard recovery: convex ensemble/prediction invariants, tuned cutoff admission
and chronological static partitions recovered from forecast-audit. Fixed an additional
mask-ordering error when baseline enrichment reorders rows; shuffled-index regression
verifies chronological blend membership. Final primary-session suite: 311 passed, two
non-failing warnings; offline wheel rebuilt and package contents/import reverified.
Tournament panel/DGW reconciliation remains open. These guards are not model results.


### 2026-10-02 — Tournament round semantics and opt-in persistent tuning

Recovered shared OOF panel validation and tournament player-round aggregation. Models
fit/predict fixture rows; comparisons sum actuals/predictions by player-position-GW,
with training-round MASE, observed training maxima and complete comparable expert
coverage. Future rows are excluded before preflight. Existing OOF/ranking/frozen
artifacts are preserved. Metadata records player_round_v1 and CSV byte identity.
MID training is aggregated consistently; historical/selection missing minutes history
retains the existing low-regime policy rather than removing player coverage.

Frozen preflight passed on 13,652 selection fixture rows, including 901 extra DGW
rows and 48 missing routing values. No research models were fitted in this preflight.
Feature frames now explicitly carry their verified frozen byte SHA after merges.

Persistent tuning is opt-in serial_trial_seed_v1, distinct from the unchanged default
legacy sampler. Identity binds actual capped feature/target frame, upstream byte SHA,
ordered features, folds/cutoff/model/seed/stage, source/search-space hashes and runtime
(including TabR dependencies and thread/device environment). A POSIX lock covers the
study operation; orphan trials, mismatched identity, reduced trial targets and incomplete
timeouts fail closed. FAIL/PRUNED count as terminal attempts and are recorded separately.
Tests verify identical parameter/value/state sequences for uninterrupted 15 trials versus
3 plus clean resume, beyond TPE startup; numerical determinism of native estimators is
not claimed. Time allowance resets per invocation. Exports go into an isolated namespace
and cannot overwrite canonical studies; legacy validators reject the new sampler unless
a caller explicitly registers that protocol. No registered studies or artifacts changed.

Primary-session verification: 391 passed, two existing non-failing warnings. Offline
wheel build, isolated import and byte-identical packaged sources/all 44 JSON artifacts
plus frozen manifest passed. Local evidence: layer2_preflight.json,
layer2_package_verification.json and layer2_sources.tar.gz under
experiments/engineering_recovery_2026-10-02/. No MILP/model-promotion claim follows from
these engineering tests; previous identical-predictions scoring controls remain unchanged.
Live API checked-round/season admission and forced-chip recovery remain the next layer.


### 2026-10-02 — Registered engineering check: live history and chip constraints

Recover exact checked-round API ingestion and active-season source matching, keeping
frozen research separate. Recover explicit positive chip forcing, WC/FH hit/carry
exemptions and first-period FH budget/permanent-state guards. Standing half boundaries,
FT cap/reset policy and corrected realized scoring remain fixed. Validate offline API
fixtures and synthetic chip cases, then same preserved standard/origin GW153-183
chip-disabled horizon-3 controls with proven optimal solves and paired bootstrap CI
(10,000 draws, block 3, seed 0). Pre-change optimizer bytes and exact plan are preserved
in experiments/engineering_recovery_2026-10-02/live_chip_recovery/. No live fetch,
model fitting or predictive improvement claim is authorized by these engineering tests.

Completed: 432 regression tests passed with two existing warnings. All 124 same-input
chip-disabled solves were optimal. Standard/control and candidate both scored 2088;
origin/control and candidate both scored 1901. Both paired differences were 0, 95% CI
[0,0] (10,000 block-bootstrap draws, block 3, seed 0). API tests use offline payloads;
no mutable live refresh or model fitting occurred. Current docs and stakeholder overview
were updated. The optimizer source review identifies modern-rule migration work without
claiming that an external solver is already replacement-ready. The PO subsequently
authorized commit, PR and merge after verification.

### 2026-10-02 — Registered optional Solio optimizer evaluation

Hypothesis: a pinned modern-rule solver can replace our optimization foundation while
preserving forecast provenance and improving decision quality or runtime. Registration,
source hashes, preserved predictions and evaluated sources are recorded under
experiments/optimizer_migration_2026-10-02/. No fetch, refit or prospective holdout use.
Both engines used identical static prices and predictions, GW153–183, horizon 3,
chips disabled, exact optimal solves. Modern FT cap 5 differs from legacy cap 2.

All 124 solves optimal. Standard legacy/Solio: 2144/2126, delta -18, paired 95% CI
[-113,58]; origin: 1902/1921, delta +19, CI [-64,82]. Bootstrap: 10,000 draws,
block 3, seed 0. Gross equals net in all four runs because no hits occurred. Legacy
runtimes 84.166/88.006 s; Solio 133.666/141.410 s. Solio was approximately 1.6 times
slower. Neither points comparison supports superiority; no default or model promotion.
Historical variable-price controls remain separate. Candidate native solver time is the
main bottleneck. Admission-only hardening subsequently yielded identical prepared inputs
for all 62 candidate windows, with unchanged registered vendor solver bytes.

Independent rule and regression suite: 556 passed, two existing warnings. Preserve the
optional integration and negative results; retain legacy default and single:catboost.

### 2026-10-17 — Solver admission and capped-carry correctness repair (#28)

Registered the capped-carry hypothesis and repair in issue #28 before changing the
formulation. Strict admission exposed a previously hidden infeasible no-transfer
scenario: carry >= unused FTs conflicted with carry <= banking cap minus one.
Fallback rows had allowed weak rollover assertions to pass despite failed solves.
The repair encodes capped carry with a saturation binary while retaining the
standing banking cap, chip-reset policy, objective and price assumptions. The
five-FT fixture also needed a club-legal initial squad; its original first three
forwards put four players in each of two clubs with no affordable repair.

Admission now requests zero native relative and absolute gaps, requires PuLP model
AND solution optimal status, finite feasible integer values, and a valid HiGHS
optimal primal/dual certificate within a predeclared 1e-7 numerical tolerance.
CBC evidence is explicitly only PuLP-reported optimality. Failed or partial runs
abort without successful recommendation artifacts; no fallback decision is scored.
Native certificates describe the encoded MILP, not independent FPL rule validation.

Focused synthetic regressions validate solver admission, FT rollover, chip and
corrected scoring behavior. These checks are not a historical backtest or evidence
of improved realized points. Prior CSVs and registered results remain preserved;
this defect requires qualifying historical solve evidence and targeted matched
replays under issue #23 before treating old totals as certified comparisons.
No forecast/model promotion or claim of modern-rule equivalence is made.

### 2026-10-17 — Training Ground paired forecast scorecards (#22/#23)

The owner reprioritized the Training Ground and experiments over deadline-current
live validation. The reporting interface now admits two separately pinned forecast
panels and reuses the established metrics for player-round MAE, RMSE, bias,
aggregate calibration, within-position/round Spearman and top-ranked capture.
Fixture inputs are summed before scoring; already aggregated inputs retain unknown
fixture counts. Full keys, positions and outcomes must match before aggregation,
without intersecting populations. Origin forecasts are stratified by lead rather
than pooling repeated target outcomes. Stable numeric player codes resolve ties.

Preserved optimizer forecasts describe a common pool including zero-filled rows,
not recovered original model coverage. Training-scaled MASE, refit chronology,
probabilistic quality and promotion remain unmeasured/unassessed. Comparison
contracts, when present, must bind these exact forecast bytes and protocol.
Reports and JSON share the same measured values; immutable receipts capture both
panels and implementation hashes. Focused scorecard/metric checks: 110 passed.

Registered a one-attempt standard legacy-optimizer replay under
experiments/training_ground_pilot_2026-10-17/plan.json and issue #23 before execution.
It reuses the original matched static-price forecasts, GW153–183, horizon3,
chips disabled, zero relative/absolute gaps, 31 solves maximum, 120s per solve and
600s subprocess wall limit. No fitting, live fetching, model selection or production
promotion. The preserved control lacks native proof and stable decision identities;
any contrast remains an unqualified engineering diagnostic. Raw artifacts and
failure/completion receipts are preserved separately from tracked summaries.

Pilot outcome: **FAILED QUALIFICATION** after 91.26 seconds. The worktree used
stale local origin/main cf3a43f, before the merged certificate/carry fix 646610b.
The registered source hashes therefore bound the older implementation; its output
proof was NOT ASSESSED and the completion check refused admission. All raw bytes
and failure evidence are preserved. The single-attempt plan was not retried, and
no corrected-carry result or qualified point comparison is claimed. The worktree
was then synced to verified main; a new registration is required for another replay.

A separate registration, plan-verified-main.json (SHA efc44b5060dc3ecb43dff670887a6822de3228a4036981dc687fb7d9beb8399d),
verified base 646610b and the corrected optimizer bytes before execution. This run
completed 31 native-proven optimal solves in 89.69 seconds: gross/net 2096, zero
paid transfers. The saved historical control has 2144 points; contrast -48, paired
95% interval [-128.0,21.0]. This is an **UNVERIFIED engineering contrast**,
not evidence for selecting or promoting a model: the historical control lacks native
proof and stable identities, and independent decision legality is not yet assessed.
No forecasts were refit. Both failed and completed attempts remain preserved with
source/input/output hashes and registry rows. Standard and origin historical reports
now include descriptive diagnostics of their shared preserved forecast pools.

### 2026-10-17 — Frozen admission for probabilistic research (#13)

Quantile static evaluation and bucket static/walk-forward evaluation, tuning and
prediction export previously loaded mutable weekly data and built features before
restricting the requested window. All five now admit the complete SHA-bound frozen
snapshot first, then retain raw history only through the requested evaluation end
or tuning cutoff before constructing features. Earlier training/form history stays
available. Static train/evaluation overlap, invalid windows, GW232+ requests and
bucket tuning beyond the standing selection cap fail before feature construction.
The subset retains the original byte hash but loses the complete-frame cache marker.

Verification: 51 focused probabilistic/admission/provenance tests passed, including
real hash-tamper rejection through all five consumers. A read-only smoke admission
of the actual preserved snapshot verified SHA62f8f1d31d47a5f859f6d73819c9e631a59de346fed7de2fd5877b994a2ccc98
and trimmed to global GW152 without fitting, fetching live data or changing production.
These are engineering checks, not new forecast or downstream performance results.
The reproducible runner, matched downstream loss comparison, ablations and opt-in
cold-start work in issue #13 remain outstanding.

### 2026-10-17 — Explicit shared forecast recipes (#13)

Standard and origin forecasting now have an opt-in per-position/member parameter
recipe and explicit seed, passed through the existing fit_position_ensembles path.
All active members must have a recipe before any fit. Registry defaults are resolved
and deep-copied first; otherwise a partial CatBoost MAE dictionary can silently
lose its named loss and use the estimator's RMSE default. Conflicting loss/objective
or seed overrides are rejected. Explicit parameters bypass ambient tuned files.

The explicit forecast mode permits single-member strategies only and excludes hidden
blend/calibration/MID-gate fitting paths. Ordered feature subsets are preserved;
missing/duplicate features and raw outcome labels fail admission. Existing standard
and frozen-origin temporal loops, minutes and fixture identities remain shared.
Omitting the new arguments retains the old production call path.

Verification:82 focused recipe, registry, expert-policy, origin and combination tests
passed (one existing runpy warning). Constructor checks establish that partial MAE
recipes retain MAE and the requested seed without fitting. Small real-fit checks
fail if any ambient tuned artifact is loaded. Integration spies verify
strictly earlier training, exact parameter/seed propagation and scoring identities.
These checks are not new real-data model results, downstream point evidence or
promotion. The manifest runner, actual matched comparisons, ablations and cold-start
scope remain open under #13.

## 2026-10-17 — Bounded Training Ground runner pilot (issue #13)

Added `fpl.research_runner`: immutable complete recipes and ordered features, frozen
data/source/runtime checks, bounded subprocess execution, measured fit receipts,
four captured forecast panels before any MILP, native-optimal corrected scoring,
paired scorecards and exact captured-file verification. Existing loops and metrics
are reused. Production remains `single:catboost`.

Registered before fitting in issue #13, comment 6239826152: plan SHA
`03d28b5ac9cfca9355ab7f11472831815577701fcc635e5b91a24f4f77343f45`,
frozen dataset SHA `62f8f1d31d47a5f859f6d73819c9e631a59de346fed7de2fd5877b994a2ccc98`.
One GW141–144 selection diagnostic, seed 0, horizon 3, retrain every 4, matched
CatBoost MAE/RMSE with 300 iterations, depth 6, learning rate 0.05, two threads
and full constructor defaults. Budget: 16 fits/16 solves, 120s per solve, 1200s
worker cap, zero gaps.

Completed 16 fits and 16 native-proven optimal encoded MILPs in 87.47s
(43.78s fitting, 28.07s solving). RMSE candidate versus MAE control: standard net
220 versus 218, delta +2, paired 95% block-bootstrap interval [-20, +15]; origin
217 versus 203, delta +14, interval [0, +31]. Four observed rounds with block 3,
10,000 draws and seed 0 are an execution diagnostic, not evidence of a reliable gain. Both
exact sign tests have two-sided p=1.0. Standard accuracy shows the expected tradeoff:
MAE .893 versus .738, RMSE 1.794 versus 1.903, bias +.059 versus -.418,
aggregate calibration 1.060 versus .576. Origin metrics are retained separately
by lead; its current-target population has 2843 player-rounds versus standard 2856.
No cross-protocol coverage equivalence or cold-start fix is claimed.

Evidence remains **UNVERIFIED**, unranked and promotion-ineligible. Independent
FPL legality and outcome-blind execution are NOT ASSESSED; this window is not a
fresh holdout. Preserved full bundle and receipt hash are referenced by tracked
`experiments/training_ground_runner_pilot_2026-10-17/{plan,result}.json` and its
`fpl.experiment` registry. Focused verification: 167 tests passed, including tiny
real recipe fits and explicitly synthetic runner integration tests. No live refresh.

## 2026-10-17 — Recover the earlier ablation family; seal runner receipts in parent

Recovered the existing October 3 seed-0 study rather than fitting its six variants
again. Read-only preservation copied 106 unchanged files (43.2 MB) from worktree
6048. Verified 37 archived evaluated Python sources against registration, all 12
prediction/squad result pairs and their exact admissions, final summary and the
pre-reference selection lock. Archived code was hashed, never imported or executed.
The dirty worktree's execution ownership remains unclassified; no authority was
transferred, files changed, locks cleared or worktree retired.

The locked opponent-feature removal scored 973 selection-origin net points versus
930 for the MAE full control. Current saved-artifact reports cover five selection
contrasts and four locked reference contrasts. Reference results reproduce the
original net totals: opponent removal standard 2040 versus 2082, delta -42,
95% interval [-141, +74]; origin 1932 versus 1905, delta +27, interval [-87, +136].
Separately tuned RMSE scored standard 1939 versus 2082, delta -143, interval
[-255, -17]; origin 1852 versus 1905, delta -53, interval [-227.025, +149].
These are loss-plus-capacity settings, not a matched loss-only experiment. Both
protocols use 10,000 block-bootstrap draws, block 3, seed 0; the historical window
is already examined and cannot establish a fresh gain.

All nine scorecards remain **UNVERIFIED and unranked**. Original exports lack native
certificates and stable-ID decisions; native optimality, independent legality and
causal publication-time validity remain NOT ASSESSED. Historical `actual_total_points`
stored net points: separate normalized exports restore gross from the preserved
gross column and retain explicit net only after checking every row against hits.
No outcome settlement, fitting, solving or decision replay was performed. All 106
input and 113 generated output hashes were independently rechecked. The script,
qualification receipt and `fpl.experiment` registry are retained under
`experiments/recovered_ablations_2026-10-17/`; the receipt locates preserved raw artifacts
and reports. Full qualification SHA:
`7cdf7245fef52f2c5093f06b983227e68e509e4ff208e758be439cc985d2add6`.

Runner completion now seals in the parent after successful child exit and log close.
The worker binds all non-log outputs first; parent verification rejects changed
artifacts, source/runtime or replacement plan bytes before writing final completion.
This prevents late native shutdown logging from invalidating a freshly completed
receipt. The earlier real pilot still verifies unchanged and was not rerun.
Focused tests: 135 passed, including late-log, late-artifact, source drift and
consistently rehashed replacement-plan regressions. These tests are synthetic
execution checks, not additional model evidence. Production remains unchanged.

## 2026-10-17 — Harden opt-in history-only cold-start priors (issue #13)

Recovered the existing pooled position/team fixture-prior calculation with fixed
strength 20, including DNP outcomes and mixed population second moments. Added
strict identity, fixture-duplicate, numeric-range and complete-row fallback checks.
Tables bind a global origin, latest admitted round, runtime-dependent history/table
fingerprints and the prior strength. Saved JSON round trips retain that metadata;
wrong-origin, changed or semantically invalid tables fail. Future labels/identities
are excluded before admitted-history validation or moment estimation.

Registered missing-player rows preserve unknown form/outcomes and bypass both normal
and MID-gate models through the optional future-prediction API. Default output stays
unchanged; the weekly CLI does not enable this option. Feature building and shared
ensemble/weight/calibration fits reject marked synthetic rows before fitting. Empirical
spread and participation frequencies are uncalibrated historical descriptors, not
live probabilities, confidence intervals or effective posterior sample sizes.

Pre-registered engineering smoke in issue #13 comment 6240323272, plan SHA
`e0e14f4274148d55bd350f2559d1ccb139ed4daff7f0655bc2505be00722d0ce`.
Canonical frozen bytes admitted; global origin 141 retained 95,543 real fixture rows
through GW140 and produced 108 prior rows across all four positions. Two prior-table
computations (original/future-perturbed) had identical values and metadata; JSON reload
and four fictitious unseen-team fallback probes passed in 1.05s. No GBM training,
optimizer solves, live fetches or player-eligibility/performance claims. All five output
hashes independently verified. Plan, exact smoke script, result and `fpl.experiment`
registry are tracked under `experiments/cold_start_prior_smoke_2026-10-17/`; raw prior
tables and probes are preserved in the corresponding external visualization artifact.

Focused verification: 131 passed with one existing runpy warning, including causal
cutoff invariance, hand-calculated moments, bool/string/overflow identities, JSON
round trip, DGW/blank routing, all-cold pools and early synthetic-training rejection.
Forecast accuracy, historical deadline registration, cold-start MILP benefit and
promotion remain NOT ASSESSED. The earlier preserved coverage-only study remains
separate evidence; it was not relabelled as a prediction or decision improvement.

## 2026-10-17 — Strict saved-decision settlement qualification (issues #21/#23)

Fixed shared corrected scoring to accept both normalized `TC` and preserved `TC_name`
labels; both now award the triple bonus, including vice activation after captain DNP.
The shared scorer offers strict selected-squad outcome coverage. Compatibility mode
preserves prior zero filling but exposes `LEGACY_ZERO_FILL` and the missing-player
count. Numeric outcomes, hits, gameweeks and within-round positions are checked.

Added `fpl.milp.replay` run/report around the existing scorer. It checks stable IDs,
15-player roster composition, club limit, XI formation, original bench partition/GK
slot and captain/vice membership. Every selected player requires current-round
outcomes. Corrected gross/net totals can be reconciled against a pinned reference.
Captured input/output bytes and failed attempts are preserved; reports reject changed
receipts. Financial/price/FT/chip-inventory legality, native solver proof and original
outcome-blind execution remain separate NOT ASSESSED fields. No second scorer or
optimizer was introduced. Legacy BB rows with 15 starters cannot supply a unique legal
XI/bench and are rejected. Both adapters use captain/vice lists; the earlier issue
lookup's scalar/list mismatch claim was corrected after checking the source.

Registered exactly four saved-pilot replay attempts in issue #23 comment 6240521098,
plan SHA `12181569b3c0c2c6340d8243bf1b0e573879576a1f86e14ab7604873a6f15e55`.
The original PR32 bundle and its 12 snapshot/forecast/reference input hashes verify.
All four attempts are **NOT QUALIFIED**: GW143 has selected-player IDs without
outcome rows (standard control/candidate: 3/4; origin control/candidate: 3/4).
These gaps may be blanks, but missing records alone do not establish that fact.
No zero rows or retrospective club identities were invented to force admission.
The earlier 218/220 and 203/217 pilot totals remain unchanged and UNVERIFIED.

All four failed receipts and 23 captured/generated artifacts are preserved and
independently hash-verified. Tracked plan, exact matrix script, result, registry and
all-round identity coverage audit are under
`experiments/decision_replay_pilot_2026-10-17/`. The original bundle still verifies
unchanged. There were zero new model fits, optimizer solves or live refreshes in
this qualification matrix. Next required evidence is explicit blank-fixture and
player-club coverage, followed by independent financial/rule-state validation.
Focused verification: 102 tests passed, including hand-calculated TC/vice, DGW,
negative points, strict missing-bench outcomes, malformed rosters, incompatible
references and corrupted receipts. Synthetic tests are distinct from this failed
real-artifact qualification. No model or optimizer promotion.


### 2026-10-17 — Explicit blank evidence resolves pilot settlement coverage

Registered four attempts in issue #23 comment 6240844501, plan SHA
`dc39135f05aea6a6a99d22b80fcaab1b580605dd9ccf2f09a27543c8631db68b`.
Pinned Vaastav 2023-24 fixtures/teams at commit
`204c134d8e90e2b6e69ec252aa3ef6f10f21e73c`. Fixture bytes SHA
`d8521bdbb80edb5141bc17df71bf6026275f2e56fd6a6dc6041b3ae23a39ce6a`;
team bytes SHA `7795f091ccea17fda0752cb07b935e1515b09a9d3f924147997cbf28139edd49`.
The complete 380-fixture ledger agrees with frozen data on all fixture IDs, events,
normalized clubs and scores. GW143/local29 has four fixtures. A manifest-bound
register captured at 2024-03-16T04:00:06Z (9.498 hours before deadline) maps all eight
missing selected codes to clubs without an actual fixture in that round.

The new replay admits only separate explicit zero outcomes with known club/position,
retaining original forecast bytes and all prior failed attempts. All four strict
settlements reconciled against their preserved gross/net references:

| Case | Explicit blank rows | Gross points | Net points | Settlement |
| --- | ---: | ---: | ---: | --- |
| Standard control | 3 | 218 | 218 | Verified |
| Standard candidate | 4 | 220 | 220 | Verified |
| Origin control | 3 | 203 | 203 | Verified |
| Origin candidate | 4 | 221 | 217 | Verified |

There were zero fits, optimizer solves or live refreshes. All 55 captured/generated
file hashes and all four original forecast byte identities were independently
verified after execution, along with the unchanged original pilot and failed
qualification result. Tracked plan/script/result/registry are under
`experiments/blank_outcome_replay_2026-10-17/`; full immutable output is preserved
in `TrainingGroundExplicitBlankReplay_20261017` in the task artifact folder.

This is settlement-only evidence, **not full experiment qualification**. Registration
is capture-time only, the fixture ledger is retrospective, and financial/rule
legality, outcome-blind execution and independent native proof remain unassessed.
Entries remain unranked; production stays `single:catboost`. Focused regression:
154 tests passed, including complete-ledger admission failures, capture hash/timing
rejection, original-byte preservation and a hand-calculated autosubstitution/hit case.


### 2026-10-17 — Explicit permanent-state exports and synthetic invariance control

New legacy snapshots and Solio rows capture fresh/continuing mode, permanent
holdings and bank before/after, and a separate temporary Free Hit selection residual.
Original budget/FT/selection fields remain unchanged. A first-round fresh Free Hit
is now rejected before loading inputs because there is no permanent squad to restore.
This does not alter the standing chips-disabled research controls.

Registered the bounded synthetic export control in issue #21 comment6240952603,
plan SHA `9fa64bb9f2a80d5905a0fd3b2824dc79cf5d5a6a581038acab131b8788909d23`.
Exactly four native-optimal solves compared the pre-change47736fc optimizer with
current source on identical synthetic two-round continuing/FH input. Every one of
21 original returned columns matched (exact list identities, numeric tolerance1e-9).
Six explicit state columns were added. No models were fitted. This is engineering
invariance evidence, not empirical FPL performance or independent financial legality.

Eight output and three input hashes independently verified; source copies, input,
logs and both machine snapshots remain in `DecisionStateExportInputs_20261017`
and `DecisionStateExportControl_20261017` under task artifacts. Tracked
plan/script/result/registry are in `experiments/decision_state_export_2026-10-17/`.
Focused regression:127 passed, including fresh/continuing state, transfers,
Free Hit restoration across rounds, temporary/permanent bank distinction, stable
ID JSON serialization, native certificates and existing corrected scoring.

Correction to earlier audit wording: legacy v/budget_end is permanent bank in a
continuing FH (e=u=0); temporary affordability is a separate constraint. The new
field exposes that separate residual without reinterpreting historical columns.
Old snapshots lacking explicit initial/permanent state are not upgraded by this
export change. Financial/rule audit and outcome-blind execution remain issue #23.


### 2026-10-17 — Strict initial-state and hit-cost comparison admission

Primary issue #21 completion audit found that matching nonempty dictionaries could
omit initial holdings and free transfers. Strict schema-1 contracts now require
explicit mode, squad, bank (`budget`, tenths) and free transfers. Fresh states have
empty holdings; continuing states have15 unique positive integer stable IDs.
Scalar/type/finite checks reject malformed state, including boolean numeric values.
Extra explicit state assumptions remain bound by pairwise equality. This validates
declarations, not monetary grids, transfer caps or independent FPL legality.

Contracts also declare the supported integer transfer-hit cost, with unsupported
costs rejected. Saved transfer-hit totals must agree with paid transfers; requested
and effective metric plus cost units are explicit in machine/human output. Invalid
season labels are rejected before global-week mapping. Prior incomplete contracts
and historical reports remain unchanged; no guessed defaults are inserted.

Focused verification:309 existing/extended contract, comparator, Playground and
saved-replay tests passed; one additional human/JSON receipt consistency test passed.
Hand-calculated gross/net controls and registered promotion conditions remain intact.
No empirical fits, optimizer replays, live fetch or production change. Full financial
rule arithmetic and original outcome-blind execution remain separate work under #23.


### 2026-10-17 — Source-derived pilot financial audit and selling-price sensitivity

Registered8attempts in issue #23 comment6241211559, plan SHA
`97abcc8676272f15551c1f286cf16c52d33ff792f34764ff9db78b0e71072f52`.
Original pilot source hashes match runner/optimizer/config bytes at e91852b
(the revision is not unique; bytes are decisive). Omitted runner overrides plus
bound defaults uniquely establish fresh empty holdings,1000tenths bank,1FT,
chips disabled, FTcap2/gain1 and current-price sales. These facts are source-derived,
not original explicit plan fields. Source copies/recovery evidence are preserved.

Independent no-chip arithmetic uses neither optimizer nor scorer. All four original
bank/hit/FT paths pass across GW141–144. A separate half-profit acquisition-basis
sensitivity on the same choices also completes with all choices affordable:

| Case | Reported/current-price closing bank (£m) | Half-profit closing bank (£m) | Difference (£m) |
| --- | ---: | ---: | ---: |
| Standard control | 1.2 | 1.1 | -0.1 |
| Standard candidate | 5.6 | 5.6 | 0.0 |
| Origin control | 0.7 | 0.7 | 0.0 |
| Origin candidate | 9.3 | 9.3 | 0.0 |

Used selected/sold price cells per case:64,64,64,65; forward-filled cells:5,6,5,5.
No used price cells depend on backward fill, missing-value default50 or disagreeing
copies. This does not prove historical publication times or forecast-horizon price
knowledge. The evaluated optimizer uses target-row values throughout its planning
window; this audit does not establish deployment-honest future-price inputs.

Zero fits, optimizer solves or mutable data refreshes; original forecasts, decisions,
settlement evidence and points remain unchanged.44 output hashes, eight case receipts
and original pilot independently verified after execution. Tracked plan/script,
result/registry and human report are under `experiments/financial_replay_2026-10-17/`;
full output/source recovery is preserved in `TrainingGroundFinancialAudit_20261017`
and `FinancialAuditSourceRecovery_20261017` under task artifacts.

Focused verification:210 tests passed, including price-rise/loss and rebuy arithmetic,
missing/contradictory prices, bank/FT/hit failures, original state and price-fill labels.
Full site-rule legality, supported chip audit, deadline-known price evidence and
outcome-blind execution remain NOT ASSESSED. No model ranking or promotion.
Selling-policy diagnostic reference: Premier League,2025-10-03,
https://www.premierleague.com/en/news/2858775/explanation-on-fpl-price-rises .

### 2026-10-17 — Origin-capture planning prices, modern-rule engineering replay

Issue #40 preregistration comment6241380947; plan SHA
`927ab9e2d1de49eaccc4144f2536e937901b5ab6c5b6d6e15ed7e27db2f76f0c`.
Preserved GW141–144 origin forecasts, horizon3, seed0; no refits; eight optimal
native Solio solves with zero reported gaps. Capture-known metadata/prices replace
raw target-round fields, acquisition basis supplies half-profit sales, FT5/no chips
and registered forecast pools match between arms. Decisions precede strict replay.

Control197 gross/net; candidate203 gross/net. Candidate-control+6, paired95%
block-bootstrap CI[-8,+11], block3/10000/seed0; two round wins each. Four already
examined selection rounds do not establish model superiority. **Recipe correction
(2026-10-18):** the preserved pilot matches explicit non-loss hyperparameters and
features; the previous mixed-capacity description was incorrect. CatBoost
loss-dependent evaluation/leaf-estimation defaults differ, as verified by all fit
receipts in `experiments/pilot_recipe_audit_2026-10-18/`. UNVERIFIED/unranked; no promotion or default change. Legacy-policy
scores remain separate because backend/rules also differ.

Recent predeadline capture prices pass pinned-byte/identity admission; origins142–144
each exclude one unregistered forecast row. Both arms retain identical fixture
coverage. Solver inputs have no outcome columns. Separate settlement covers every
selected player, adding four evidence-backed retrospective blank rows per arm.
Exact-deadline prices, fixture-information causality, complete outcome-blind execution,
exported native primal/dual certificates and full independent site-rule legality
remain NOT ASSESSED. No wholly absent players are added. Source review confirms
the inherited Solio converter zero-fills missing player-round forecasts; their
planning validity remains NOT ASSESSED and was not a separate preregistered field.
Strict settlement blank evidence does not retroactively validate that assumption.

Tracked plan/script/result/registry/report: `experiments/origin_price_replay_2026-10-17/`.
Full output bundle `TrainingGroundOriginPriceReplay_20261017/first` and capture inputs
`OriginPriceReplayInputs_20261017` are externally preserved. Input/source/runtime
hashes were verified before/after, then all output hashes and both settlement
receipts rechecked. Focused origin/Solio regression:199 tests passed.

### 2026-10-18 — Unified preserved-pilot evidence scorecard

Extended existing Training Ground reports with optional hash-pinned strict
settlement receipts (issue #22, comment6246615073; related #40). No new scorer,
model fits or optimizer solves. Reporting spec SHA
`079b12d14442f680be3f831887d6cbd56f4a6a67ab1e738dec7767bffade9e8c`.
Both source replay bundles and compared CSV identities are verified. The scorecard
combines existing player-round/DGW forecast metrics, origin leads, paired net
comparison, gross/hits/net, recorded transfers/chips and solver fields.

The preserved modern-price pilot remains203 candidate versus197 control, +6,
paired95% CI[-8,+11]. UNVERIFIED/unranked and not promotion-eligible. Forecast
diagnostics cover the original forecast export, including rows later excluded
from the registered decision pool; this limitation is explicit in the population.
Recorded solver status/gap and price/pool claims do not upgrade price-information,
forecast-blank planning, fixture causality, complete outcome-blind execution, native
certificate or independent-legality gates. All six remain NOT ASSESSED.
Attempted-solve denominators/failure rate and original fit runtime remain unknown.

Full captured reporting attempt is in task artifacts under
`TrainingGroundUnifiedScorecard_20261018/runs/origin-price-pilot-evidence/readable-report`.
The spec, same-data JSON/Markdown, comparison and external-bundle completion
manifest are tracked under `experiments/unified_scorecard_2026-10-18/`; copied
completion hashes refer to the full external bundle, not just tracked summaries.
Verified report re-render equality, receipt hashes, immutable attempt reuse and
unranked league listing. Existing historical outputs remain unchanged. Focused
verification:133 tests passed, including malformed/corrupt receipts, missing
timings, zero/negative totals and prevention of solver-claim evidence upgrades.

The first reporting attempt is preserved externally. A separate `readable-report`
attempt replaces raw recorded-field JSON prose with a compact comparison table;
underlying forecasts, decisions and settled point totals remain unchanged.

### 2026-10-18 — Correct matched-pilot recipe description

The origin-price replay's plan/result/evaluated script incorrectly repeated a
capacity-confounding caveat belonging to the older separately tuned October3 study.
The original four-round runner plan and all16 start/completion receipt pairs prove
identical ordered features, depth6/300 iterations/learning-rate0.05, seed0, threads2
and all other explicit non-loss parameters. All eight protocol/position pairs match
cutoffs and training-row counts. Fitted parameter differences are loss_function,
eval_metric and loss-dependent leaf_estimation_method (Exact versus Newton).

Preserved original bundles and registered plan/result/script bytes unchanged.
Corrected the human interpretation and published a reproducible no-fit/no-solve
audit under `experiments/pilot_recipe_audit_2026-10-18/`. Existing point totals,
paired intervals, UNVERIFIED/unranked status and all information-time/promotion
gaps remain unchanged. Older separately tuned study caveats remain valid.

### 2026-10-18 — Registered ownership-feature ablation, retain full control

Issue13 comment6246841017 preregistered plan SHA
`0daba4410faaa6d8e8ba761ed432b9f5d3a47a8578d1887977f6113828ba379d`
and companion SHA e092cdd515d1e9497942f021f7563255201afc017b5292f0dfba71b3d0333d71.
SelectionGW137–152, seed0, H3/retrain4; same CatBoost MAE depth6/300 iterations/
learning rate0.05/two threads. Candidate drops only five shifted ownership
features (115→110); all32 paired fitted recipes/support cells match otherwise.

| Protocol | Full net | No-ownership net | Difference | Paired95% CI |
| --- | ---: | ---: | ---: | --- |
| Standard | 1030 | 1037 | +7 | [-55,+61] |
| Origin | 987 | 939 | -48 | [-113,+4] |

Block3/10000/seed0. Candidate incurs16 extra hit points in both protocols; origin
also loses32 gross points. Small lead0/standard MAE reductions do not establish
better decisions. Retain full-feature control; no general ownership-effect,
equivalence or promotion claim. Both comparisons UNVERIFIED/unranked.

One attempt,64 fits/64 native-optimal decisions,551.2s total (194.3 fit-call,295.1
optimizer-stage wall time). All170 output hashes,64 fit-start/completion pairs,
32 matched cells,64 stable-ID certificates, CSV identities and gross/net/hit
arithmetic rechecked. The exact floating-zero post-check was stricter than the
existing certificate tolerance: largest reported gap1.53e-16, primal/dual
difference4.27e-14. Rechecked against unchanged native certificate policy; no
solver reruns or relaxed stopping gaps.

Tracked plan, registration, receipts, registry, summaries, verification and report
under experiments/ownership_ablation_2026-10-18; full immutable raw bundle is
TrainingGroundOwnershipAblation_20261018/first in task artifacts. Future target
prices, fixture/registration knowledge, missing-outcome completeness, independent
site legality and complete outcome-blind execution remain unqualified. No
reference/final evaluation, prospective selection or production change.

### 2026-10-18 — Ownership ablation qualification correction

Supersedes the preceding retain-control inference: selected-row coverage audit
found11/9 missing standard control/candidate player-round pairs and23/11 origin
pairs. Origin control selected code568791 (Callan McKenna) in GW137, absent from
all supplied capture-time registers GW137–148 and forecast rows before GW149.
This fails the capture-time known-pool policy; exact-deadline eligibility remains
NOT ASSESSED. Legacy scores cannot identify an ownership-feature effect.
Raw plans, outputs and170-file manifests remain unchanged. Derived coverage_audit.json
and updated report/verification preserve the counterexample and input hashes.
All16 recovered captures passed the existing loader; register-aware saved-forecast
replay requires a separate registration, zero refits and equal policies across arms.
Issue13 remains open. Production remains single:catboost.

### 2026-10-18 — Strict multi-event settlement evidence

Refs13/23/40. Extend optional blank-evidence input with a pinned collection of
unchanged single-event descriptors. Reuse the existing register/complete-ledger
validator for every event, reject duplicate identities/unsafe paths/changed hashes,
and capture collision-free nested assets for offline receipt verification.
Independent focused replay, collection, blank-outcome and scorecard checks:141 passed, including
a synthetic complete-ledger two-event settlement and source-removal readback.
These checks establish engineering behavior, not an experimental FPL point gain.
Capture-time registration remains distinct from exact-deadline proof and advance
fixture knowledge. Original single-event interface and scoring remain unchanged.

### 2026-10-18 — Captured-register ownership replay failed closed

Plan b560336e14eaa122d628bccb702bb64f80d45b8720b97962dcdcbe6536ad9400,
registered issue13/comment6247100879. Saved origin forecasts, GW137–152, zero fits,
maximum32 Solio solves/60s each, equal capture-pool/price/modernFT5/nochip policies.
Candidate timed out; parent exit1 after180.97s. No retry, candidate score or pairedCI.
Control16 exported decisions all matched corresponding capture registers; strict
16-event settlement passed with12 explicit retrospective blanks:983 gross/24hits/959net.
Single-arm engineering evidence only, not comparable with mixed-policy legacy totals.
Actual attempted-solve count and candidate partial decisions were not recorded.
All214 sealed external files reverified; tracked package under
experiments/registered_pool_replay_2026-10-18. Issue13 stays open; issue6 owns
per-origin checkpoints/timing follow-up. Broader causality/legality/outcome-blind
gates remain unassessed; production single:catboost unchanged.

### 2026-10-18 — Opt-in Solio wrapper checkpoints and boundary timings

Refs6/13/22. The registered-pool candidate timeout left no failing origin or
partial candidate decisions. Add opt-in checkpoint_dir/--checkpoint-dir to preserve
immutable origin starts, prepared identities, vendor call starts/returns/failures,
normalization, pre-settlement first decisions and permanent next state. Terminal
receipts bind all earlier event files and record attempted wrapper calls, returned
vendor calls and completed decision origins separately. Interrupted/unsealed runs
remain incomplete. No automatic retry/resume or changed optimality policy.

Preparation/admission/vendor wall/native/normalization/state timings have explicit
boundaries. Successful native runtime comes from the vendor return; vendor-minus-
native is non-native overhead, not isolated build time. Failed native/build timing
remains NOT MEASURED. Wrapper/config/vendor source hashes and execution versions
are recorded; complete outcome-blindness remains NOT ASSESSED. Vendor source bytes
and production strategy remain unchanged. This is instrumentation, not a speed or
model-improvement result and not full issue6 acceptance.

Focused checks:211 passed across checkpoint, origin-price, input/output and native
solver interfaces. New cases retain the first decision after a second-origin failure,
reject timeout/nonoptimal output, preserve permanent Free Hit state, keep interrupted
runs unsealed, reject directory reuse and changed source identity, and retain decisions
before a later scoring failure. A Free Hit fixture initially used forbidden GW1;
corrected the test to GW2/3 without changing the existing chip admission rules.

### 2026-10-18 — Checkpointed captured-pool replay failed at candidate GW139

Registered plan80abc0bc04662cebacb5e29faf7be6f94afbd208227bf35f18abd20167578a6d,
issue13/comment6249794554; source027ec0e. Same saved ownership ablation forecasts,
GW137–152/H3/seed0, equal captured pool/price/modernFT5/nochip policies; zero fits.
Separate120s-per-solve attempt, max32 wrapper calls/4200s parent, no retries.
Control16 calls/16 saved decisions; candidate3 calls/2 returns/2 saved decisions,
then native timeout atGW139. Parent691.77s; failed vendor wall122.83s, native/build
split unknown. No paired points/CI, shortened-window scoring or model verdict.
All181 raw hashes and receipt bindings/order/counters verified. Capture-register
membership passed18 saved origins; independent no-chip financial arithmetic passed
16 control/2 candidate rounds after explicit fresh-acquisition normalization only
in derived audit copies. No new settlement because the candidate did not complete.
Initial derivative's unknown timing totals0/count0 are superseded by null/count0;
both derivative sources/results are preserved, and no solves/fits were repeated.
Evidence: experiments/checkpointed_pool_replay_2026-10-18 and external raw/postcheck
bundles named in its verification.json. Issue6 now has a concreteGW139 model/native
failure-profile case; issue22 failed-attempt adapter and issue13 remain open.
Production and broader deadline/fixture/legality/outcome-blind gates are unchanged.

### 2026-10-18 — Opt-in native model and failure diagnostics

Refs issue6. Added a single-window diagnostic hook preserving the exact linear model,
effective options, construction/native-call timings and available native status,
bounds/gap/counters before ordinary admission. Immutable artifacts distinguish
recorded diagnostics from accepted results. JSON is authoritative; MPS is review-only
and exported on a disposable copy after tests exposed live-name mutation in HiGHS.
Default mathematical model, optimality requirements and production are unchanged.
No research solve, point comparison or runtime-improvement claim in this change.
Primary focused verification:129 passed across serialization, real tiny-FPL solves,
timeout and normalization rejection, vendor regeneration, checkpoints and origin
prices. Round-trip tests distinguish exact serialized storage from HiGHS' equivalent
sparse-layout conversion. Existing-directory failure emits no vendor-start receipt.
The next separately registered case is candidateGW139/H3 at the existing120s limit;
issue6 remains open until measured diagnosis and a justified improvement are complete.

### 2026-10-18 — Registered GW139 native diagnostic reached optimality

Plan c7bf258693c568197114e4db473d47aca6fe90b393570eed791aa944d51d7bf6,
issue6/comment6250305359, source4ea4d62. One preserved candidateGW139/H3 solve,
seed0/120s native/300s parent, zero fits/retries. Full prior181+214 file seals and
admitted/prepared/options identities verified. Returned Optimal at68.140s native,
0.590s construction/73.314s parent;32,646 columns/57,360 rows/224,825 nonzeros,
objective=bound125.39588942007455,gap0,399 nodes. Normalization accepted; model JSON
round-trip verified without another solve. Prior native matrix was not exported;
input equality alone is not independent across-revision matrix equality. No claimed
speed improvement: original replay still failed and parallel/process contexts differ.
No points/settlement/CI/promotion. Raw external bundle GW139NativeDiagnostics_20261018;
tracked experiments/gw139_native_diagnostics_2026-10-18. Issue6 remains open; next
registered work should test exact-runtime repeatability/thread policy before change.

### 2026-10-18 — Verified operational entries in the Training Ground league

Refs issue22. Separate operational importer/reader verifies pinned schedule/native
receipts, exhaustive file coverage, safe paths, original source/runtime identity,
event ordering, boundary counters and recorded native acceptance consistency.
Captured evidence, JSON card and readable report are immutable; offline reads
revalidate the captured copy. Registration links bind bytes, not chronology.
Failed/incomplete imports and uncommitted export artifacts remain distinguishable
from completed operational reporting; no inferred process termination or outcomes.
All operational entries have null rank/points/delta/CI and no promotion eligibility.
Primary affected checks304 passed. Preserved real control:16 attempted/returned/
saved origins; candidate:3 attempts/2 returns/1 failure/2 saved, failedGW139;
separate native case:1 start/1 return/acceptedTrue. Generated human/JSON agreement,
verified reuse and all-unranked league checks passed. Original raw bytes retained.
No fits, solves or settlement during this reporting exercise. Evidence recipe and
summary copies:experiments/operational_league_2026-10-18; full external bundle
OperationalLeagueReporting_20261018 is sealed and named in verification-links.json.
Historical registry values were checked unchanged. Broader scorecard, training-only
MASE/probabilistic adapters and qualified promotion evidence remain open under22.

### 2026-10-03 — Registered reproducible forecast priorities, seed 0

PO authorized the proposed research plan. First package covers reproducible controls,
existing MAE/RMSE decision comparisons, bounded feature ablations and opt-in registered
cold-start coverage (issue #13). Production remains single:catboost. Before fitting,
experiments/research_priorities_2026-10-03/selection_seed0/registration.json registered
exact parameters, all source hashes, ordered features, software versions and frozen
input SHA 62f8f1d31d47a5f859f6d73819c9e631a59de346fed7de2fd5877b994a2ccc98.
The controls are newly fitted validated seed-0 selection artifacts, with five-thread
runtime overrides; they are not a replay of historical 2088/1901 controls.

Hypothesis: loss and existing feature families may alter realized decisions differently
from forecast error. Six predeclared selection-origin variants: full MAE/RMSE and MAE
without fixture, minutes, opponent or xP features. Selection GW137–152 chooses at most
one ablation, ties retaining full features; RMSE is predeclared. Finalists are locked
before standard/origin GW153–183 evaluation. Horizon 3, retrain every 4, chips disabled,
legacy HiGHS gap 0, proven optimal solves, corrected autosubs_v2, net transfer hits
with gross points retained. Paired bootstrap 10,000 draws, block 3, seed 0.

This is exploratory, seed 0 only, not promotion evidence. GW153–183 is repeatedly
examined historical reference; spent GW191–221 and prospective GW232+ are excluded.
Archived prices/fixtures lack complete publication-time provenance. Cold-start cohort
coverage uses first-recorded player rounds; it cannot prove historical registration
or a realized MILP gain. New defensive-contribution scoring is outside this reference.

Admission and cold-start regression checks passed; full suite initially 654 tests,
with four additional explicit parameter-forwarding/exact-member tests subsequently
passing. First-recorded cohort diagnostic: selection 58 players/58 player rounds,
68 fixture rows, 50 DNP rounds; reference 185 players/185 player rounds, 187 fixture
rows, 127 DNP rounds. Pooled-prior player-round MAE/RMSE are 1.333/1.495 selection
and 1.206/1.591 reference. These forecast diagnostics do not establish a decision
gain. Preserved cohort hashes and limitations are in cold_start_diagnostic.json.

Completed seed-0 evidence: newly fitted full MAE 2082 standard / 1905 origin net
points; tuned RMSE 1939 / 1852; locked MAE-without-opponent 2040 / 1932. Selection
origin totals: MAE 930, RMSE 1000, without-fixture 930, without-minutes 963,
without-opponent 973, without-xP 918. Only without-opponent advanced, locked before
reference evaluation. All 282 solves optimal. RMSE paid 32 selection hit points
and 8 origin-reference hit points (gross 1860, net 1852); every other run paid zero.

Candidate-minus-MAE paired 95% CIs (10,000 draws, block 3, seed 0): RMSE standard
-143 [-255,-17], origin -53 [-227.025,149]; without-opponent standard -42
[-141,74], origin +27 [-87,136]. The standard RMSE loss is supported on this
historical window; the origin RMSE gap and both ablation gaps are inconclusive.
Neither challenger establishes a production improvement. Retain single:catboost
and full features. This comparison assesses complete tuned configurations, whose
loss and tree capacity differ; it does not isolate a causal loss-function effect.
Recorded forecast generation plus MILP summed to 52.6 minutes on the local CPU.
Forecast-generation timers include refits, predictions and origin snapshot rebuilds.

Independent artifact verification recomputed every prediction/squad hash, checked
paired pools/outcomes/minutes/prices/fixture metadata, net-hit arithmetic and all
confidence intervals. verification.json records PASS and the final summary file
SHA 9d421af2cea36abaa33a8339c7bd981927fea4e117c75587a80bda14f8bbc5d7.
The original selection-summary bytes were preserved before reference output and
match the pre-reference finalist lock. The original completion registry hash is
of canonical JSON content, not pretty file bytes; the later VERIFIED entry records
the actual file SHA explicitly without altering that historical record.

After evaluation, the original evaluated sources were preserved under
selection_seed0/evaluated_sources/ and checked against registration. Subsequent
hardening retains immutable selection_summary.json automatically, logs final
summary file bytes, validates paired forecast inputs, checks source hashes around
every reference pass and at completion, and derives the probabilistic training
ceiling at runtime. The quantile module's captaincy explanation now states that
equal means give equal expected doubled points. These changes did not refit or
re-score the preserved experiment. Final full suite: 671 passed, two existing
warnings; one additional final-pass source-drift regression passed separately.
External-data, neural and scenario priorities remain conditional subsequent work;
no GW232+ admission, production promotion, commit, push or merge.

### 2026-10-03 — External-source admission audit (issue #14)

Pinned Vaastav 9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88,
Core Insights bcea139cd777b4f16e21887db22c90e5b498999e and OpenFPL
ad055ba7270145ce8ecc1f19581423493b1357e7. Preserved compact directory inventories,
Git blob verification, sample byte hashes and conclusions in
experiments/external_source_audit_2026-10-03/audit.json, SHA256
9a1c0b52c2cdf1f1920b5d595e52ea5a2699e2609c0fdf3b55c7a62b85f3dd1a.
No source is admitted to fitting. Individual Understat player-history file counts
are 0/712/554/774 for 2020/21–2023/24; filenames are not full-cohort coverage.
The 2020/21 aggregate has 524 players but no event dates. The sampled 2021/22
identity dictionary has 609 unique exact pairs, all joining the 737-player roster
with unique stable codes. Those sample bytes match the pinned revision; this does
not backdate the final dictionary to earlier deadlines. The sampled player history
contains 264 matches across seasons 2014–2021, with date-only event timestamps
and no publication/capture fields. Directory season labels are insufficient filters.

Core Insights begins at 2024/25 (GW153), after discovery/selection; historical GW
folders are event partitions and dynamic statistics are rewritten after finished
rounds. Cup coverage is uneven; capture/publication timing and Elo effective time
remain unproven. OpenFPL's pretrained inference artifacts and later-season imputation
do not provide an admissible temporal retraining corpus. Understat RF/XGB, its blend,
and cup/rest/Elo/availability candidates remain blocked on these prerequisites.
No GW232+ player outcomes accessed. Missing external collection stays missing.

Implemented research-only fpl.data.external contract: full-byte SHA before parsing,
explicit publication claims and timezone-aware timestamps, exact season-scoped
identity translation, composed mapping/roster availability, latest pre-deadline
versions and explicit current-fixture exclusion. All 39 focused synthetic tests
passed in the primary session. Tests validate consistency, not factual publication
proof. No external feature or production integration and no model-gain claim.

### 2026-10-03 — Earlier-only quantile interval calibration, seed 0

PO authorized continuation through the usage limit. The numerical protocol was
registered before fitting under the research backlog (#11); publication of its
prepared focused issue text remains pending explicit approval after automatic
review rejected disclosure of that payload. Local registration and implementation
are authorized. Artifacts: experiments/interval_calibration_2026-10-03/seed0/.
Every registered source is archived and hash-verified. Frozen input unchanged.

Fixed LightGBM p10/p50/p90, 300 trees, seed 0, five CPU threads, existing full
features, no tuning. Fit fixture targets; sum distinct fixtures into player rounds
before calibration. Selection: fit <=GW128, calibrate GW129–136, evaluate GW137–152.
Reference: fit <=GW144, calibrate GW145–152, evaluate GW153–183. Signed CQR scores
max(lower-y,y-upper), exact ceil((n+1)*0.8) order statistic; adjusted intervals
bracket the unchanged median. Summed fixture quantiles approximate round intervals,
not exact quantiles of a sum. Dependent temporal/player observations do not satisfy
an established exchangeability guarantee. This is static one-step distribution
validation; no horizon-three origin distribution or expected-points optimization
change, production promotion or realized MILP gain is claimed.

All eight position/phase adjustments were exactly zero. Raw/calibrated coverage,
width and interval score are therefore identical. Selection pooled coverage 0.91679,
width 2.28498, interval score 4.25432, n = 12,751 player rounds. Reference coverage 0.89984,
width 2.52208, interval score 4.73221, n = 21,606. Large zero-score atoms in earlier
calibration data (roughly 59–78% by position/phase) prevent smooth calibration toward
80% inclusive coverage. Descriptive reference coverage is0.98868 for 12,279 DNP
rounds and0.78289 for 9,327 played rounds; participation is known after the match,
so these subgroup results cannot be used as deadline-known selection/calibration
labels. No retuning based on reference diagnostics.

Local total 44.73 s, including 4.48 s feature preprocessing and 39.12 s across 24 fits.
Independent verification recomputed archived-source hashes, calibration order
statistics, CSV hashes and raw interval metrics, and confirmed identical endpoints
and preserved medians. Null result retained; no model or optimizer defaults changed.
Numerical/runner regressions and external admission: 98 focused tests passed. The
full suite before the final eight external edge-case additions passed 762 tests,
with two existing warnings; the eight new regressions passed separately.

### 2026-10-03 — CNN native-runtime containment before historical fitting

The bounded CNN's separate synthetic CPU estimator tests passed, and a standalone
Torch 2.13.0 Conv1d microprobe passed. The primary combined registry/origin/CNN
suite nevertheless exited 139 with a native segmentation fault in Torch Conv1d.
Root cause is not established; the failing process also loaded the tree libraries.
Preserved command, output fragment, source bytes/hash and prior microprobe context
in experiments/cnn_runtime_2026-10-03/crash.json. No historical CNN fit occurred.
The combined in-process path is blocked, with no retry. Existing PyTabKit/MPS crash
guards remain unchanged. A fresh CPU process worker is being validated separately
before any registered historical experiment; synthetic success will not constitute
forecast or MILP gain evidence. Native estimator contract tests execute only in a
fresh subprocess, so the combined regression suite cannot retry the failing path.

### 2026-10-03 — Bounded CNN selection, isolated CPU worker

After the in-process native failure, the process-isolated backend passed synthetic
validation with GBMs in the parent and Torch confined to a fresh child. Direct CNN
fitting now fails before Torch import when GBMs are loaded. Crash, timeout and partial
protocol responses fail closed without retries. The full suite passed 847 tests,
with four direct native tests skipped in the parent and passed separately in a fresh
subprocess; two existing warnings. Two later selection-input regressions also passed.
No historical fit attempted on the blocked mixed-runtime path.

Registered exact sources, parameters, feature order, runtime, frozen input and
fixed-fit calendar before evaluation in experiments/cnn_research_2026-10-03/seed0/.
Eight prior recorded player rounds (not calendar weeks), eight statistics, fixture
sums retaining NaN, 72 sequence/mask columns added to the 115 base features.
Two Conv1d layers with 16 channels, kernel 3/padding 1, masked pooling, context and
missingness, 32-unit head; seed 0, 15 fixed epochs, batch 1024, Adam 0.001,
weight decay 0.0001, gradient clipping 1, training-only preprocessing/MSE scaling.
No tuning. All variants fit once through GW136 and issue horizon-three origin
forecasts on GW137–152; these are different controls from the earlier rolling-refit
package. Every added sequence column is rebuilt from origin-truncated history and
frozen over the full horizon. Pools, scoring inputs and prices match exactly.

All 64 MILP solves proven optimal. Net/gross/hits: base MAE 926/926/0;
MAE plus sequences 893/893/0; RMSE plus sequences 1024/1056/32;
CNN plus sequences 982/1022/40. CNN failed the preregistered strict point gate
against the strongest tree, so reference features were not built or evaluated.
This rejects advancement of this bounded candidate, not CNNs in general.
No production change or MoE promotion. Verification independently recomputed
source archives, prediction/squad hashes, identical-input comparisons, net-hit
arithmetic, optimal counts and immutable selection lock.

Completed wall time 491.71 seconds (8.2 minutes). Forecast-generation timers
include training, worker startup, predictions and origin rebuilds: MAE 29.04 s,
sequence MAE 33.07 s, sequence RMSE 111.33 s, CNN 82.38 s. These are not pure
single-fit benchmarks. The earlier commentary's 9.9-minute estimate was too high;
the completed timer above is authoritative. Final summary byte SHA256
ceff744b26b2ffccf968b0384c8bfae1067b20499a9ca0362ad968bc09a0e616.

Interpretation correction for uncertainty: inclusive p10–p90 coverage above 80%
can arise from discrete point masses even with correct quantiles. It does not by
itself prove excessive width or under-confidence. The earlier historical log's
continuous-outcome interpretation should not be reused. Report both tail miss rates,
width and proper interval score; the signed earlier-only corrections remained zero.

### 2026-10-03 — Runner artifact preservation hardened before seed repeats

Primary review found that the package rechecked current Python hashes but did not
rehash every preserved input, lock and scored output throughout its run. New package
runs now archive exact registered sources before evaluation and repeatedly check
full dataset bytes, pinned tuned JSONs, registration, archive, immutable selection
summary/lock, prediction/squad CSVs and result JSONs. CNN and temporal-blend runners
also bind and recheck their registered archives, locks and scored CSVs before
ranking, paired CIs and completion. A mutation fails closed with an incomplete
checkpoint; no automatic retry. Existing verified historical artifacts unchanged.

Mutation regressions cover dataset, parameter, archive, registration, frozen
selection/weight/pilot locks and prior scored outputs, including drift during CI
calculation. Full suite: 915 passed, four direct native CNN tests skipped in the
mixed parent and passed in an isolated subprocess, two existing warnings; 26.06 s.
Production remains single:catboost; no commit, push, merge or promotion.

### 2026-10-03 — Earlier-only CNN/RMSE temporal blend, seed 0

Registered and archived all 47 current Python sources before scoring in
experiments/temporal_blend_2026-10-03/seed0/. Cached member source inputs were
verified against their original 45-source archive, immutable selection summary
and lock, full prediction CSV hashes and verification record. Frozen data unchanged.
Weight-fitting current-origin player rounds GW137–144 only, summing DGW fixtures.
Closed-form no-intercept convex MSE weights: CNN DEF 0.3816267815, FWD 0.0343532380,
GK 0.4283043450, MID 0.3756859836. No intercept, tuning or reference weight update.

Late GW145–152 replay resets the same fresh MILP state for both members. Blend
579 net / 587 gross / 8 hit points; RMSE 504 / 520 / 16. Delta +75, block-3 paired
bootstrap CI [9, 141]; two-sided sign test p = 0.125. This was already examined
selection data, including earlier selection of the tree member. The pilot is
exploratory continuation, not independent unseen validation. Strict points gate
passed; all settings and pilot bytes locked before reference feature construction.

Reference fixed fits <=GW152; GW153–183 standard/origin net points:
full MAE 2032/1906; sequence RMSE 2056/1944; blend 2112/1956.
Blend origin has four hit points, all other reference variants have zero.
Blend vs RMSE delta standard +56 CI [-22, 161], origin +12 CI [-148, 147.025].
Blend vs MAE standard +80 CI [-48, 226], origin +50 CI [-172, 251]. Every reference
CI includes zero. There is no established repeatable gain, no promotion, and no
comparison against historical 2088/1901 or the rolling-refit package controls.

All 202 solves optimal, gap zero, chips disabled, corrected autosubs_v2 with hits
paid. Independent verification recomputed earlier fixture-summed weights, cached
blend predictions, archive/cached-input/dataset hashes, unchanged weight locks,
exact paired reference metadata, net/gross/hit arithmetic, optimal counts and CI
point differences. Verification and its exact script bytes are preserved locally.
No joint distributions or risk objective are established by a mean blend.

Completed time 1460.28 s (24.3 minutes), including cached pilot and six reference
passes. Forecast timers (include fitting, snapshots, worker startup/predictions):
blend standard158.07 s/origin212.70 s; RMSE92.3/146.9 s; MAE7.6/60.3 s.
Final summary byte SHA256
0c7788908b8d362c53d15b52761ab0cf5924c9174ee194f5a102dfcfcefb5e5f.

### 2026-10-03 — Tree refit seed robustness registered

Registered seeds 1 and 2 together before either evaluation in
experiments/research_priorities_2026-10-03/robustness_protocol.json; serial launcher
and its bytes are bound. Uses the exact seed-zero tuned studies, same frozen input,
feature variants and four-round refit calendar as the completed first package.
Only refit random seeds change; no tuning or production promotion. Each seed selects
its ablation on earlier origin points, ties retaining full features, and freezes
finalists before reference. Different finalists cannot be pooled as one treatment;
all seeds share outcomes and do not constitute independent replications.

Seed 1 started in exec session 84806; seed 2 follows only after seed 1 completes
and registered source/input checks pass. Child runs preserve every scored artifact
and checkpoints; parent summary is written after each completed seed. Current
status is registered/running, not a result. Runtime metadata records the session
for classification on continuation. Do not overwrite, rerun or transfer execution
authority if the existing process cannot be classified. No commit/push/merge.

### 2026-10-03 — Historical future-price limitation made explicit

Read-only audit confirmed origin forecasts retain each target row's `value`.
milp.optimize constructs value_matrix_df by player/GW and uses its horizon columns
in future purchase/sale constraints. Origin form freezing therefore does not freeze
the realized future price path. Same-input comparisons remain registered historical
replays; their absolute points do not establish fully point-in-time deployment.
Documented in FORECAST_RESEARCH. No running settings or score artifacts changed.
A distinct origin-known price protocol is needed before a strict deployment claim;
fixture ratings and availability publication timing remain separate evidence gaps.

### 2026-10-03 — Tree refit robustness completed and independently verified

Serial exec session 84806 completed with exit_code 0; no training remains running.
Parent wall time 7502.96 s (125.05 minutes), seeds 1/2 preregistered together using
unchanged seed-zero tuning studies. Each seed: 282 optimal solves, 47 archived
source hashes, 85 immutable output hashes and eight tuning hashes verified.
Primary verifier additionally recomputed paired metadata, all net/gross/hit
arithmetic, earlier selection ranking and complete bootstrap/sign-test reports.
Exact verification script and per-seed/parent verification JSON are preserved in
experiments/research_priorities_2026-10-03/. Parent protocol byte SHA256
1a00af8f14da7a1add769693aa83bcd25ea0f603c185341a169cb0b6594d965c;
parent summary 5241e506476243c777334b9715bdd84b633b90bfb42fa4eebf2f320d2e6b10d4.

Seed 1 selected without-fixture (1031) over MAE full (991); RMSE selection1130.
Reference standard/origin: MAE1937/1965; RMSE1979/1913; ablation1958/1917.
RMSE standard delta+42 CI[-144,204], origin-52 CI[-221,86]. Ablation standard+21
CI[-119,139], origin-48 CI[-149,22]. RMSE hits standard4/origin16; other arms0.
Seed 1 summary cd5198d055cf1a740cd1c4c2b9cb824882c00bb1f499abb098e9297946042b5c.

Seed 2 selected without-minutes (1010) over MAE full (969); RMSE selection1084.
Reference standard/origin: MAE1979/1987; RMSE1882/1881; ablation2101/1893.
RMSE standard delta-97 CI[-321.025,168.025], origin-106 CI[-296,138]. Ablation
standard+122 CI[-58,340], origin-94 CI[-277,100.025]. RMSE hits standard0/origin28;
other arms0. Seed 2 summary 7bf58bb439d88190c94ff70f5b5545de5778c8d0647f45392d31a03be18f00ae.

With seed0, RMSE lost origin net points in all three refits (-53,-52,-106), while
selected ablations differ (opponent/fixture/minutes). No replicated feature gain.
All seeds share outcomes; different ablations are different treatments, and these
are diagnostic historical replays with known future-price limitations. Retain
full-feature MAE research control and production single:catboost. No promotion,
commit, push or merge. The earlier running entry remains historical status only.

### 2026-10-03 — Cached origin price sensitivity registered

Registered experiments/price_provenance_2026-10-03/seed0/ before any scoring;
49 current pipeline source files archived. Uses completed seed0 MAE/RMSE origin
forecasts, no model fitting or tuning. Four predeclared arms in GW137–152 and
GW153–183 compare target replay against latest strictly prior observed prices
held constant over each horizon. Both policies share price-known eligibility and
fresh identical initial state. All 188 solves must be optimal with gap0, chips0,
autosubs_v2 and net transfer-hit accounting. Paired block3/bootstrap10000 seed0
within-model price comparisons and RMSE/MAE within each policy are predeclared.

Catalogs for both phases sealed before first solve; current and original archived
sources, cached input bytes, data, catalog locks, scored files and metadata rehashed
throughout. Known-price union at initial selection origin:807 eligible/55 unknown;
reference origin:587 eligible/184 unknown. Recorded price ages reach 23 and115
rounds respectively at these initial origins. A historical observation is a proxy,
not deadline price/publication proof; stale observations are not silently labelled
current. No retrospective age filter is selected from reference outcomes.

Controls differ from unrestricted historical controls due to common eligibility.
Legacy selling-price and transfer rules remain; roster/fixture/availability timing
still needs evidence. This is price sensitivity, not full point-in-time validation
or promotion. Full suite before registration:989 passed, four native CNN tests
skipped in mixed parent (isolated tests cover them),16 warnings,32.84s. Tiny CBC
synthetic tests prove future-target-price perturbation cannot change asof budgets
or decisions and allknown replay catalog retains legacy outputs. Runtime is owned
execsession11710, serial MILP only. No training, commit, push or production change.

### 2026-10-03 — Price proxy sensitivity completed: model ranking depends on prices

Exec session 11710 completed with exit code 0: 817.75 s (13.6 minutes), 188 optimal
zero-gap solves, no model fits. All 49 archived source files, 47 cached input hashes
and 95 immutable output hashes verified against registered bytes. An independent
catalog reconstruction checked 13,454 selection and 22,094 reference observations,
each using the latest recorded GW strictly before its origin. Primary verification
decoded named squad exports through unique names/stable codes, reconstructed all 188
current budgets, checked known-price eligibility and net/gross/hit arithmetic, and
recomputed all paired bootstrap/sign-test reports. Exact verifier, verification.json
and completed runtime classification are preserved in
experiments/price_provenance_2026-10-03/seed0/.

Selection net points: MAE replay 930/asof 942; RMSE replay 1000/asof 975. RMSE paid 32
hit points in each policy; MAE paid 0. MAE policy delta +12 CI [-46, 80]; RMSE policy
-25 CI [-41, -2]. Reference MAE replay 1905/asof 1889; RMSE replay 1678/asof 2110.
RMSE replay gross 1702/hits 24; all other reference arms paid 0 hits. Net differences:
MAE asof versus replay -16 CI [-174, 90]; RMSE asof versus replay +432 CI [326, 569.025].
RMSE versus MAE under replay -227 CI [-427, -18], under asof +221 CI [36, 483].

This reverses the single-seed ranking under a changed price proxy. It does not
establish production gain, actual deadline prices or independent holdout evidence.
Asof changes current price proxies as well as future horizon paths; their effects
are not isolated. Conditional horizon-only freezing is the next registered
diagnostic, without new model fitting. Multiple descriptive comparisons are not a
registered promotion assessment.

The restricted RMSE replay differs from the original unrestricted 1852. Elyh
Harrison, stable code 588555, first appears in frozen raw history at GW176/price 40;
his first origin forecast is 177. The original RMSE squad nevertheless held him
from GW153 through 176 using the optimizer's future union pool. The original MAE
squad never held a player without prior observed prices. Missing price/history is
not proof of missing historical registration: a real registered debutant can lack
this history. The stricter proxy catalog therefore changes eligibility deliberately.

All selected asof observations were at most two rounds old; catalogs also include
much older unselected observations. One selected RMSE replay player-round has
price age 39 and no horizon fixture, exposing remaining roster/availability timing
uncertainty. No age filter was chosen from reference outcomes.

Summary byte SHA256 abd278ab943deda2ed3434bb30408cf81a48f14e0beb77eb725a7ec1734f41f7.
Production single:catboost unchanged. No commit, push, merge or public publication.

### 2026-10-03 — Conditional horizon-price freezing completed

Exec session1528 completed exit0. Four new evaluations reused verified price
controls, identical prediction CSV bytes and known-price catalogs.50 current
source files archived;74 immutable output hashes verified.94 optimal solves and
94 independent current-budget reconstructions passed. Complete source/input/
lock checks, forecast equality, net/gross/hits and all10 paired reports were
recomputed. Source/verification/runtime/wheel evidence preserved in
experiments/price_horizon_2026-10-03/seed0/. No model fits or tuning.

Elapsed408.31s (6.8 minutes). Selection MAE930/RMSE993; reference MAE1938/RMSE1856.
Reference current-horizon versus replay: MAE+33 CI[-24,91], RMSE+178 CI[42,312].
RMSE versus MAE current-horizon-82 CI[-259,98]. Versus lagged-asof, MAE+49
CI[-68,216], RMSE-254 CI[-414.025,-128]. Reference ranking flip disappears when
only horizon prices change; intervals do not establish a model promotion.

Separate independent initial-squad price lookup exposes a season reset issue:
lagged-asof MAE initial squad costs1035 and RMSE1080 under reconstructed current
prices, vs1000 fresh budget. RMSE costs979 under the lagged proxy and retains21
bank units. Target/current vectors are not verified deadline quotes, but price
repricing makes the prior-price proxy unreliable as a practical budget input.
The diagnostic is preserved as price_provenance/seed0/initial_price_sensitivity.json.
Do not interpret the asof2110 score as deployable gain. Subsequent registered
seed1/2 price comparisons exclude that proxy because of this budget mismatch,
not because of its favorable model outcome. Preserve examined-window limitations.

Summary bytes SHA256 fcd53056c55473c727cae6cef84f0a196b243b6eaca795ece725f3b77aa736b9.
Full suite before scoring1013 passed/four native parent skips/22 warnings,33.10s.
Offline wheel built without network: all50 Python sources and45 model/data JSON
artifacts match bytes; installed-package import preserves lazy Torch and production
single:catboost. Wheel SHA256 dee6d0c866e6326f3d7e27bee2d7fbe1d7ecc90884da01b8ebbd954d2dc8a199.
No production changes, commit, push, merge or public publication.

## 2026-10-03 — Cached price-policy robustness across refit seeds

Registered both seeds 1/2 before scoring, reference GW153–183 only, cached MAE/RMSE
forecasts and matching known-price eligibility. Target replay versus a frozen
reconstructed current-origin price vector; the lagged-price budget proxy was
excluded because season repricing changed starting-squad feasibility. No new
model fits, hyperparameter search, finalist selection or untouched holdout.

Owned exec session 1453 returned exit code 0. Elapsed 1344.60 seconds (22.4 minutes).
Independent per-seed checks and the parent verifier passed: 248 optimal solves,
248 reconstructed budgets, 432 sealed files, 50 archived source files. Recomputed
all paired bootstrap/sign-test reports, net/gross/hit arithmetic and cached
prediction equality. A separate read-only audit found no material defect.

| Refit seed | Replay MAE / RMSE | Fixed-current MAE / RMSE | Fixed-current RMSE minus MAE, 95% CI |
|---:|---:|---:|---|
| 0 (previously verified) | 1905 / 1678 | 1938 / 1856 | -82 [-259, 98] |
| 1 | 1965 / 1891 | 1958 / 1872 | -86 [-209, 27.025] |
| 2 | 1995 / 1881 | 1999 / 2016 | +17 [-116.025, 181.025] |

No repeated RMSE advantage is established. RMSE's within-model horizon-policy
changes are +178, -19 and +135; MAE's are +33, -7 and +4. Seeds share outcomes and
seed-zero tuning studies; no independent-sample pooling. These eligible-player
controls differ from the original unrestricted controls. Current vectors retain
historical reconstruction and possible filling, rather than proven deadline
quotes. Selling-price, roster and publication-time limitations remain.

Preserve production single:catboost and the full-feature MAE research control.
Prioritize deadline-bound prices, season repricing and historical rosters before
further model promotion claims. Completed evidence and independent verification:
experiments/price_provenance_2026-10-03/seed_robustness_*.json,
reference_seed1/verification.json and reference_seed2/verification.json.
Parent summary SHA256 c2040ffba0eedbf15b0b4df1691e5924caffe566910d27bdb1dbf181da894a31.
Latest full suite: 1050 passed, four direct native-parent skips covered by separate
fresh-process tests, 22 warnings, 32.99 seconds. No active training/solver jobs.
No production changes, commit, push, merge or new public publication.

### Subsequent final engineering verification

Final review found two admission issues and fixed them without rerunning or
rewriting completed studies. The live driver and direct legacy optimizer now
reject owned IDs absent from the forecast pool before solver construction; the
live guard covers Solio too. A registered player without a fixture across the
horizon can otherwise lose all exported rows. No synthetic fixtures are created.

New interval runs immediately hash registration, source archives and each CSV,
then rehash them around phases and completion. The final summary includes an
immutable manifest. Eight mutation/manifest regressions cover archive, registration,
dataset and earlier output changes. Bucket walk-forward diagnostics no longer
claim to constitute realized-decision evidence. Original study source archives
remain preserved and the original numerical evidence is unchanged.

Primary full suite after both changes: 1064 passed, four direct native-parent skips
covered separately in fresh-process tests, 17 warnings, 45.24 seconds. Aggregate
scientific verification binds ten completed studies and independent reports,
1642 optimal solves and 487 unique immutable outputs. No repeated scoring.

Final offline wheel build and installed imports passed; all 50 Python sources and
45 model/data JSON files match source bytes. Torch remains lazy and production
single:catboost is unchanged. Wheel SHA256
772f3c618cd1a740fb9e990c58206623bbd861b59d0dd455a37dbb6d7be08d5b.
Package verification and reproducible scripts are under
experiments/research_priorities_2026-10-03/package*_verification.json.
No active training/solver jobs, production promotion, commit, push or merge.


## 2026-10-03 — Independently dated historical bootstrap admission

Related source prerequisite: issue #14; availability/cold-start research remains
exploratory under issue #13 and does not change production. Public pinned sources
are Randdalf/fplcache de78720734e8c24954433fbf91bc356188cfc669 and
unsaltedpw/fpl-armband 4cbbe8b206686a6def6751966c640fbc673333a7.
Candidate folder names and Git commit clocks were not accepted as capture proof.
Internet Archive CDX timestamps and original response SHA1 digests independently
verify 47 selection/reference and 96 earlier training captures. Decoded body
SHA256/length also match the pinned source manifests. Original gzip response
bytes, not decoded JSON, carry the CDX digest. Transport failures remain recorded;
40 earlier captures were not verified and are explicitly missing, not substituted.

Offline admission requires selected official endpoint, exact timestamp/digest,
body deadline, target is_next event, stable codes, team/position identity and age
strictly greater than zero and at most 24 hours. Corruption and schema ambiguity
fail; only absent, over-age or earlier-event captures can become missing signals.
This admits 78/136 discovery rounds, 16/16 selection rounds and 28/31 reference
rounds. Reference GW160/161/162 captures are 225.849/377.349/562.849 hours old
and therefore rejected. All target outcome rows remain present. Validated type-5
assistant managers are excluded and counted, never converted to a player position.

The new pure availability transform adds six status indicators, published next-
round playing chance and capture age. Missing source/player coverage stays NaN.
An exact-origin snapshot callback freezes these values over the forecast horizon;
no current-round outcome or later availability is used. This is information at
the selected crawl time, not proof that prices/status/registration were unchanged
until the actual deadline. Latest eligible capture selection is not claimed.
fplcache filename clocks/near-deadline comparisons are corroboration only.
The armband tree has no LICENSE; upstream implementation is not vendored or
executed, its source documents and local payload corpus remain ignored.

An outcome-free pool audit reproduces the live snapshot's active-code rule:
selection missing registered codes range from 0–9; fresh reference rounds from
0–133. At GW153, 133 registered players are absent, including 72 with no prior
dataset row. These are crawl-time coverage counts, not independent repeated
observations or a cold-prior calibration/realized-points result.

Evidence: experiments/deadline_snapshot_audit_2026-10-03/{wayback_verification,
discovery_wayback_verification,admission_results,forecast_pool_audit}.json, with
repeatable offline admission and roster audit scripts. Primary full regression
suite after the adapters: 1194 passed, four skips, 17 warnings, 46.99 seconds.
No new model fitting or MILP scoring had occurred when this entry was recorded.


### Registered prior coverage follow-up and sealed availability study

The crawl-time registered-cohort check supplied finite empirical priors for all
191 missing stable codes across 44 admitted selection/reference origins. Each
prior uses only history strictly before its origin; total fitting took 10.259
seconds. There is no target accuracy or realized-decision assessment. The saved
registration's target_outcomes_read flag was overbroad: the frozen CSV parser
loads outcome columns, while prediction uses only strictly earlier outcomes.
registered_priors_metadata_correction.json preserves this correction without
altering the original registration or numerical output.

A later admission-verifier run changed only a source hash for the now-finished
availability runner and inadvertently replaced the report bound by the coverage
registration. The exact original bytes were reconstructed using the earlier
recorded source hash and accepted only after their complete SHA256 matched the
original registration (72224c545152dcab91a9f0f4f9cb31b725469a3fa8c209170aa18e729e892ab2).
The newer report is preserved separately as admission_verification_current.json;
raw capture evidence and admission counts were unchanged. Subsequent verification
commands write a separate current report to preserve completed-study inputs.

The seed-zero availability experiment registers identical tracked MAE capacity,
115 control features versus 123 challenger features, before fitting. Selection
origin net points are 930 control / 952 candidate; paired block CI for +22 is
[-25, 110], with 32 optimal solves. Both arms freeze reconstructed current-origin
prices across the horizon and share strictly-prior-price eligibility. Forecasting
took 51.2/55.4 seconds; MILP took 56.9/108.6 seconds. The strict registered point-
estimate gate advances the fixed candidate; no gain or promotion conclusion
follows from this uncertain selection result. Reference settings are locked
before scoring. Reference standard control is 2082 (same as the earlier fresh
control), forecasting 95.9 seconds and MILP 159.0 seconds. Reference evaluation
was still running when this interim scientific evidence entry was recorded.

Primary full suite with the sealed runner: 1213 passed, four skips, 17 warnings,
33.65 seconds. Offline wheel verification matches all 53 Python sources and 45
model/data JSON artifacts, installed imports pass, and Torch remains lazy.
Wheel SHA256 fd7509f5e49a851d56d87b9036c56889c8fd3e2c86223165adfaba5fb9ba2f3d.
The ten previously completed studies still verify 1642 optimal solves and 487
unique immutable outputs. New availability solves are not yet in that aggregate.


### Availability seed-zero completed — no improvement

Registered reference comparisons completed in 1503.98 seconds (25.1 minutes):
MAE control standard/origin 2082/1938; availability challenger 1917/1922.
Candidate minus control standard -165, paired CI [-357, 0.025]; origin -16,
CI [-122.025, 51.025]. Forecasting/MILP seconds: control standard95.9/159.0,
control origin194.0/126.4, challenger standard107.4/160.3, origin210.9/158.4.
These forecasting timings include fitting, inference and snapshot reconstruction,
not a single training-pass benchmark. No hyperparameter search was performed.

Independent primary verification rehashes original source archives, frozen input,
all admitted capture inputs and 91 immutable outputs; reconstructs both strict
prior-price catalogs and confirms unchanged outcome-fixture multisets; reproduces
all three bootstrap/sign reports, net/gross/hit arithmetic, disabled chips and
156 exact-optimal statuses; reconstructs 94 origin current-price budgets and
checks selected-player eligibility. Summary SHA256
e8db77bd7152eeb05d90b96cfcf3d4fad9ea2db1f8286645958f34f9fdff2174.

The new information did not establish a decision gain. The feature family remains
research-only, with selected-crawl-time coverage and reconstructed-price/roster/
FT/selling limitations explicit. No production promotion, commit, push or merge.

Seed1/2 replication was registered before those fits and while seed0 reference
evaluation was still running. Same tracked seed-zero parameter capacities,
eight additions and coverage mask, with seed overrides only. Each seed retains
the strict earlier origin-net gate; a loss/tie has no reference evaluation. Seeds
share historical outcomes and are not pooled independent samples. The staged
implementation was applied only after seed0 completed and passed verification;
current source hashes match the parent protocol exactly. Focused primary checks:
168 passed. Parent registration and extension are retained under
experiments/availability_research_2026-10-03/robustness_*.


### MAE availability replication rejected; RMSE information pilot registered

Seed1 selection control921 / candidate881, delta-40, CI[-181,69]. Seed2
control969 / candidate925, delta-44, CI[-162,95]. Both fail the predeclared
strict net-point gate and correctly have no reference evaluation. The two-seed
parent completes in506.33 seconds (8.44 minutes). Each study has32 optimal
solves and32 independently reconstructed origin budgets. Primary recomputation
of all three individual studies plus parent byte/archive checks passed; the
family's220 optimal solves are bound by robustness_verification.json. The
seed0 positive selection result is not stable and its reference lost on both
standard/origin comparisons. No availability gain or production change.

A separate one-seed RMSE information pilot was registered before fitting, after
the fixed MAE replications. The same eight additions, data/coverage and pricing
are used; both arms use their own tracked RMSE capacity. No new HPO, new feature
selection, pure MAE-versus-RMSE claim or promotion protocol. Each loss shares the
same repeatedly examined historical outcomes. A strict earlier origin gain is
required before reference scoring. Its proposed source extension hashes were
registered, applied only after the MAE parent ended and passed verification, and
matched exactly before fitting. The driver seals its own bytes and verification
script before execution. Local scientific artifacts are under
experiments/availability_research_2026-10-03/rmse_*.

Primary focused checks183 passed. Full suite1243 passed, four skips,17 warnings,
34.69 seconds. Latest offline wheel sources53 andJSON artifacts45 all match;
installed imports/defaultproduction/lazyTorch pass. Wheel SHA256
85a94bcce5bd188cb5bace492b5d65b8ee0e7f325f902d2e9642e7cdf892e5af.
The RMSE pilot was still running when this entry was recorded.

### 2026-10-04: RMSE availability selection gate

The registered RMSE pilot completed selection with control 993 / candidate 996
net points, delta +3 and paired block-bootstrap CI [-96, 119]. The exact paired
sign-test one-sided candidate-better p-value is 0.84912109375 (six wins,
nine losses, one tie). These diagnostics do not establish an improvement; the
predeclared strict point-estimate gate nevertheless advances the unchanged
candidate. Both arms together have 32 optimal selection solves. Forecasting
seconds are 499.19 control / 488.49 candidate; MILP seconds 87.41 / 75.59.
The forecasting interval includes repeated fits, inference and snapshot work,
not one four-position training pass. Reference settings are locked before
scoring and evaluation remains in progress. No production change or promotion.

The first reference arm reproduced the RMSE standard control at 1939 net points.
Forecast generation took 751.4 seconds and MILP 179.7 seconds. The remaining
reference arms were still running at this entry; partial controls are not a
candidate result.

A separate verified compute ledger rehashes all 13 completed studies' summary
and independent report bytes. Primary numeric reconstruction checked seven
study timing tuples and exclusion of reused price controls. The aggregate
verifier now admits both PASS and verified report labels; five synthetic
status cases passed in the primary session. No pipeline sources were changed.
The RMSE completion finalizer seals its scripts before waiting for the parent
verification report. It rechecks all pilot input/source locks, then creates new
aggregate/timing/finalization reports without repeating fits or solves. It
fails on an explicit study error, changed script bytes or a three-hour timeout.
The prior 1862-optimal-solve package remains preserved pending completion.

Independent finalizer fixtures exposed two postprocessing weaknesses: Python
optimization could bypass assertions, and inputs were not rechecked after child
report generation. Original bytes remain in finalize_rmse_evidence_v1.py. The
initial waiting process classified exit1 on changed script SHA without creating
reports; the model process remained untouched. Hardened finalization rejects
optimized execution and rechecks parent/protocol/input/source/summary/report
bytes after each child, plus aggregate bytes after ledger generation. Its v2
script lock is rmse_finalization_v2_input_lock.json. Primary regression checks
cover eight tests and 15 subtests, including mutations after either child and
existing-output preservation. Full primary suite: 1251 passed, four skips,
17 warnings and15 subtests in55.23 seconds. Pipeline sources are unchanged from
the sealed pilot and the verified 53-source wheel. Reference scoring remains
in progress; no final RMSE availability result is claimed here.

Primary byte reproduction checks confirm the RMSE standard reference control
matches the earlier registered tree-package control exactly: prediction CSV
SHA, squad CSV SHA, net/gross/hit totals all identical. Both actual CSV files
were rehashed. The proof is rmse_control_replay_verification.json. This verifies
the standard control only; remaining origin/candidate arms are still running.

### 2026-10-04: RMSE availability pilot completed and verified

The registered seed-zero pilot completed in 5473.19 seconds (91.2 minutes).
Reference standard control/candidate: 1939/2055 net points, delta +116,
paired block-bootstrap CI [-16, 237]. Reference origin control/candidate:
1856/1943, delta +87, CI [-65, 207]. Exact paired sign-test one-sided p-values
are 0.23656483 standard and 0.22912916 origin. Both confidence intervals include
zero. This is encouraging single-seed evidence relative to the registered RMSE
control; it does not establish a gain or production-promotion case. Earlier
selection was +3, CI [-96, 119]. The same repeatedly examined reference and
selected crawl-time/reconstructed-price limitations apply.

Forecast-generation/MILP seconds: control standard 751.4/179.7, control origin
1031.6/185.5, candidate standard 903.3/141.0, candidate origin 938.9/174.0.
These are repeated-fit forecast-arm timings, not one training pass. All 156
solves were optimal. Independent verification reconstructed 94 current-price
budgets, checked unchanged fixture outcomes and all paired CI/sign reports,
and bound 91 immutable outputs. Summary SHA256:
ed4bb9809814222552303594dbeefee416fe25708a3d07ee7a6297826b53fb91.

The hardened waiting finalizer completed successfully. The aggregate now binds
2018 optimal solves and 811 unique immutable outputs across 14 completed
studies. A separate RMSE-inclusive compute ledger preserves the earlier ledger.
Primary rechecking of 395 pilot/input/output artifact hashes passed and matched
the finalization comparisons exactly. Both owned background processes exited
zero; no outstanding job remains from this pilot. Full suite remains 1251 passed,
four skips, 17 warnings and 15 finalizer subtests. Production remains
single:catboost; no commit, push, merge or model promotion.

### 2026-10-04: RMSE refit-seed replication registered

After the completed seed-zero pilot was independently reverified in the primary
session, fixed refit seeds 1 and 2 were registered before either new fit. Both
retain the identical data, archive mask, eight additions, tracked seed-zero
RMSE capacities, pricing and strict earlier-origin selection gate. Model and
paired-bootstrap seeds change together; no new HPO or feature selection.
All baseline outputs and archived sources are bound alongside raw-input and
current-source hashes. Child settings/versions are checked against the parent,
and each completed child receives a fresh independent numerical verifier.
Negative/tied selection stops that seed before reference scoring. Outcomes are
shared and the historical reference remains repeatedly examined; this is a
replication protocol, not production promotion.

Preflight regression tests reject invalid seeds/models, changed baseline bytes,
wrong driver/verifier hashes, invalid prior status and existing output paths
before fitting. An exact integer identity guard rejects False as seed zero;
the real seed-zero metadata is integer 0. Primary full suite: 1260 passed,
four skips, 17 warnings and 34 subtests in 43.42 seconds. No pipeline sources
changed. Driver SHA256:
a07469cc04b11f5fd551a736fb706fe534b7e4da9a4b79d8d7a5d4ce74a7fa27.
The serial replication was launched after registration. Outputs are
rmse_seed1/rmse_seed2 and rmse_robustness_* under the availability study.

### 2026-10-04: RMSE replications completed and independently verified

Fixed refit seeds 1 and 2 completed normally. Selection-origin control/candidate
points were 1081/1062 for seed 1 (difference -19, paired CI [-122, 57]) and
1116/1050 for seed 2 (-66, CI [-181, 23]). Both failed the registered strict
positive-selection gate and have no reference evaluation. Together with
seed-zero selection +3 [-96, 119], this family does not establish a stable
benefit. The previously reported positive seed-zero reference intervals still
include zero. Keep production single:catboost.

The serial parent took 1430.96 seconds (23.85 minutes); children recorded
692.39 and 730.39 seconds. No HPO was repeated. The new replication verifier
binds 64 additional optimal solves and 220 for the RMSE availability family.
A separate 16-study package now binds 2082 optimal solves and 953 unique
immutable outputs. Earlier aggregate reports remain unchanged. Primary
independent child numerical verification and parent hash rechecks passed.
The additional read-only recheck binds 1351 unique evidence artifacts; its
current-source observation distinguishes later config additions from archived
study code. There is no fresh holdout or production promotion claim.

New evidence: rmse_robustness_verification.json and
rmse_replications_recheck_2026-10-04.json under the availability study;
package_with_availability_replications_verification.json and
compute_timings_with_replications_verified.json under research_priorities.

### 2026-10-04: Grouped uncertainty diagnostics registered and completed

Added a research-only grouped conformal calibration core and runner. Fixed
exposure groups use strictly shifted mins60_rate_roll5, boundaries 0.2/0.8,
an explicit missing group and position fallback for fewer than 100 earlier
calibration rows. Settings, frozen input, original quantile outputs and source
archives were sealed before preprocessing. No quantile model was refitted.
The original position-only control is numerically reconstructed before scoring.
Player-round interval-score deltas use whole-round moving blocks of length
three, 10000 draws, seed zero and player-round weighting. Both fixed phases
are descriptive diagnostics, regardless of selection direction.

The run completed normally in 5.36 seconds. Independent numerical verification
is pending; do not interpret its diagnostic result as model promotion evidence.
Root full suite after the core/runner/config additions: 1345 passed, four skipped,
17 warnings and 34 subtests in 35.98 seconds. Production remains unchanged.

### 2026-10-04: Grouped interval diagnostic independently verified

Primary execution of the independent NumPy/Pandas verifier rederived signed
order-statistic cells, sparse fallbacks, original control and grouped bounds,
all subgroup metrics and weighted noncircular block intervals. All 169 bound
artifacts passed; original CSV fields and capped shifted-feature/DGW joins match.
Selection proper interval-score delta was -0.0062463, CI [-0.0189959, 0.0034776];
reference -0.0029823, CI [-0.0108672, 0.0048672] (candidate minus control, lower
better). Both include zero. Reference coverage rose 0.899843 to 0.902620 and
width 2.522082 to 2.571142. Played-only score worsened slightly in both phases;
that subgroup uses observed minutes and is diagnostic only. No clear benefit.

A new wheel built without dependency installation; all 55 pipeline Python files
match current source bytes, with frozen manifest and vendor license included.
Wheel SHA256: 70123381a7a3701683558cc731be46b43b88f5971c1483ed76be296c7205bbf8.
Evidence lives in grouped_interval_calibration_2026-10-04; no fit or MILP was
run by its numerical verifier or packaging check.

### 2026-10-04: Registered missing-pool outcome coverage audit

A fixed coverage-only linkage was registered before frozen outcome parsing,
reusing the 191 sealed registered player-origin prior rows. Same-round fixture
labels are joined by stable code and origin; absent labels remain missing,
observed DNP zero is distinct, and no retrospective fixture count changes any
prior or forecast. The run completed normally: all 191 rows have observed labels,
192 fixtures, 93 with positive minutes, 88 with positive points and 180 summed
observed points. No registered-position/team mismatches. These are repeated
player-origin observations, not an attainable optimizer gain or independent
sample. Independent reconstruction is pending. Eight synthetic regression tests
passed, including pre-parse registration, drift, DGW and absence contracts.

### 2026-10-04: FOSS interface issue and historical Understat feasibility

Created GitHub issue #15, Explore reusable FOSS interfaces for the FPL pipeline.
Five pinned source repositories were inspected; three have actual license files,
two need licensing clarification. Detailed local findings are in
FOSS_INTERFACE_AUDIT.md. Automatic approval review rejected publishing that
specific detailed comment/effort estimate without explicit payload approval;
the requested issue itself was created successfully. No upstream application
code was executed or adopted.

One Understat Liverpool/2020 HTML archive capture is now locally preserved and
its decoded SHA1 independently matches the Archive CDX digest. Decoded SHA256:
c185d7a7c599af3bf7d9c2d6ea7caf1af757b66ef730349eaeceba8626aba7da.
The 2020-11-01 capture contains team fixture data plus player season-to-capture
aggregates, not player-match histories. This proves a sparse acquisition path;
it does not establish full deadline coverage, timezone/identity mapping or
admissibility for model fitting. No external observations entered training.

### 2026-10-04: Cold coverage proof and safe Understat extraction

Independent primary reconstruction of registered missing-pool outcomes passed:
original 191 prior rows/fields remain unchanged; fixture sums, every phase/position/
status count and all preserved bytes match. No position/team mismatches or missing
labels occurred in this bounded cohort. The eight added audit tests brought the
full suite to 1353 passed, four skips, 17 warnings and 42 subtests (39.69 seconds).
No predictive score or optimizer benefit was assessed.

The original Understat replay and independent CDX metadata are now both preserved,
with matching timestamp/URL/digest records. A static restricted-string/JSON parser
extracts the three named collections without executing JavaScript. It rejects
ambiguous keys/IDs, nonfinite values, malformed literals and oversized HTML;
65 synthetic tests passed. Actual extraction found seven completed matches and
31 upcoming fixtures, plus 21 player season aggregates and seven statistics
objects. Source numeric strings and naive event datetimes remain unchanged.
The parsed collections are ignored local evidence, with compact hashes retained.
No clock inference, player identity mapping, model fit or external-data admission.

### 2026-10-04: Verified diagnostic package and league-calendar preparation

A separate eighteen-study evidence extension preserves the earlier package and
adds the independently verified grouped intervals and missing-pool coverage
diagnostic. It rechecks 1116 bound artifacts and 1074 unique generated immutable
outputs. Optimal solves remain 2082: neither diagnostic ran an optimizer. This
aggregate is bookkeeping, not independent-sample pooling or promotion evidence.

Root full suite after the static source parser: 1418 passed, four skips,
17 warnings and 42 subtests in 39.95 seconds. Diff whitespace checks passed.

A partial priority-seven candidate is being prepared from unused existing EPL
kickoff fields: prior league rest days and counts over (0,7]/(0,14] days. Entire
current DGWs are excluded, lower-gameweek matches must also precede the clock,
missing league history stays NaN, and origin values must freeze over the horizon.
The historical fixture calendar remains a schedule proxy, not proof of deadline
publication; cup/European load and Elo are not supplied. Discovery quality checks
found no missing kickoff/fixture fields, conflicting team-fixture clocks or
fixtures with more than two teams among 92289 admitted rows. Nine synthetic
workload tests passed. No workload model has been fitted or scored yet.

### 2026-10-04: League workload comparison registered and launched

After primary review of the fixed helper/runner and synthetic gate/source tests,
the full suite passed 1449 tests, four skips, 17 warnings and 55 subtests in
41.05 seconds. The seed-zero workload study was then registered before any
feature construction or model fit. It binds the exact driver/helper bytes,
55 pipeline sources, frozen input, manifest and tracked parameters. Both arms
use identical seed-zero MAE capacity, five threads, four-round refitting and
horizon three. Exact ordered 115/118 schemas, context and price catalogs lock
before each phase's first fit. Selection-origin current-price proxy is common;
reference standard is target-price replay and origin retains the shared proxy.

The candidate adds only league_rest_days, league_matches_last7_days and
league_matches_last14_days. All current DGW fixture rows share context; only
strictly lower-round fixtures before the earliest current kickoff count. Origin
context freezes over the horizon and missing team/round context remains NaN.
The candidate must strictly exceed earlier selection net points before reference
features are admitted. This is a historical schedule-proxy experiment, not full
all-competition load, exact deadline publication evidence or model promotion.
The serial job is active; no result or gain claim yet.

Independent read-only Understat reconstruction separately matched all original
transfer/decoded/index/parser hashes and reconstructed the same three collections
without the production parser or JavaScript execution. Capture clocks and exact
CDX row identity agree; all training-admitted flags remain false.

### 2026-10-04: Workload candidate passes registered selection gate

Selection-origin control/candidate scored 930/1008 net points: +78. Recorded
control forecast/MILP times were 53.9/59.7 seconds; candidate 57.5/53.3 seconds.
Primary byte comparison confirms control predictions and squads exactly match
availability seed-zero's previously verified control. A separate replay proof
preserves both pairs of hashes. This control identity guards against accidental
baseline drift; the candidate result is not a production-promotion assessment.

The positive gate admitted reference evaluation. The local serial study remains
active. A reviewed independent verifier is queued behind terminal completion and
will reconstruct contexts, ordered schemas, budgets, realized scoring, forecast
diagnostics, paired confidence intervals and exact sign tests without refitting
or solving. The finalizer accepts only terminal complete/rejected evidence,
refuses incomplete runs or changed verifier bytes, writes a new verification file
once and does not retry failures. Reference results and final verification are
pending. Do not include this active study in the eighteen-study completed package.


## 2026-10-04 — Setup/logic audit and historical optimizer qualification (issues #6/#13)

The audit checkpoint is `experiments/setup_logic_audit_2026-10-04/pause_checkpoint.json`;
source, test-log and historical-audit byte hashes were rechecked on resume. Final
engineering validation: **1,624 passed, 4 skipped, 55 subtests**, 19 warnings,
41.45 seconds. `pip check` found no broken requirements; `git diff --check` passed.

Confirmed optimizer defects: a requested zero relative gap was omitted from the
HiGHS constructor, retaining its 0.0001 default; PuLP can translate time-limited
feasible incumbents to problem status `Optimal` while solution status remains
`IntegerFeasible`. The wrapper now passes explicit zero relative/absolute gaps,
requires optimal problem and solution statuses, saves native HiGHS termination,
primal/dual bounds and achieved gap, and aborts instead of exporting fallback
squads. Free-transfer lookahead now enforces exact capped rollover; the previous
lower-bound-only equation could invent future free transfers or become infeasible
at the banking cap. Existing policy cap and legacy chip-reset semantics remain.

Forecast admission now rejects conflicting player/fixture identities and invalid
IDs/prices/cross-origin metadata. Selection pools are paired before ranking;
admissions, logs, frozen manifest and final summaries are hash-bound. Reference
features load after the finalist lock. Comparisons reject incompatible scoring/
point-accounting and nonfinite data. Owned-squad retrieval no longer drops unknown
assets; incomplete/ambiguous state and pre-GW1 previous-round retrieval fail clearly.
Explicit parameter/seed overrides are refused for auxiliary fits that cannot honor
them. The isolated CNN has a separate PyTorch 2.13.0 requirements file.

Read-only historical validation found **94 valid forecast exports and 15 matching
selection candidate/control pools**. The corresponding 94 logs contain **2,238
reported `Optimal` statuses**, but no retained solution-status/native-gap certificate.
Those counts do **not** prove exact optimality. Original squads, scores, negative
findings and verification bytes are preserved. New paired saved-prediction replays
are required before interpreting comparisons under the corrected optimizer; this
is an optimizer audit, not model promotion or a new untouched holdout.

The previously pending EPL workload study completed before the audit: reference
control/candidate were 2082/2019 standard and 1938/1897 origin, differences
-63 [95% CI -159,54] and -41 [-152,55]. `verify_v2.py` independently reconstructed
156 scored rounds and 114 retained artifacts; it discloses 78 absent whole-squad
player-round outcomes using existing scorer omission semantics. The original
verifier/finalizer failure is preserved. These are historical-wrapper results
subject to the same optimality and free-transfer qualification, not corrected
optimizer evidence.

## 2026-10-04 — Corrected optimizer: paired byte-preserving MAE/RMSE replay

The registered replay completed all 12 cases (three seeds, two full registered configurations, standard/origin) over GW153–183, horizon 3, chips disabled. Original forecast CSV bytes were copied exactly; no model was refitted. All 372 solves requested zero relative and absolute solver gaps, retained optimal problem/solution/native status and primal/dual certificates, and passed independent accounting reconstruction. Native gaps were at most 3.391e-16; the predeclared verification tolerance is 64 machine epsilons for floating-point certificate roundoff, not a nonzero solver setting.

| Seed | MAE standard | RMSE standard | Difference (95% paired CI) | MAE origin | RMSE origin | Difference (95% paired CI) |
|---:|---:|---:|---|---:|---:|---|
| 0 | 2091 | 2036 | -55 [-192, 49] | 1825 | 1840 | +15 [-147, 191.025] |
| 1 | 2080 | 1997 | -83 [-224, 75] | 1945 | 1811 | -134 [-307.025, -1] |
| 2 | 2059 | 1877 | -182 [-366, -11] | 2007 | 1824 | -183 [-354, 0] |

Differences are RMSE minus MAE; paired moving-block bootstrap uses 10,000 samples, block length 3 and the corresponding seed. The RMSE configuration loses all three standard comparisons and two origin comparisons; it does not demonstrate an improvement and remains research-only. This compares complete configurations with different tree capacities, not loss function alone. Production remains `single:catboost`.

Local replay took 25.34 minutes (25.29 minutes in MILP; zero fitting time). The independent verifier rehashed 137 artifacts, reconstructed 372 rounds and reproduced all six comparisons. It discloses 88 missing all-origin whole-squad outcome rows under the existing scorer omission semantics; these are not invented zero labels. Full regression: **1638 passed, 8 skipped, 19 warnings, 55 subtests**, 39.98s; `git diff --check` passed.

Evidence: `experiments/corrected_optimizer_replay_2026-10-04/replay_v3/{registration,summary,verification}.json`; sealed pipeline sources and original-byte forecast/squad/log artifacts are retained locally. The stopped v1 literal-zero roundoff check and v2 byte-roundtrip check are preserved separately. Neither was a native failure; v3 registered byte copying before new scoring.

These are corrected-optimizer replays of repeatedly examined historical forecasts, not a new untouched holdout or production promotion. Earlier score tables remain preserved and qualified: changed FT formulation and solver handling mean they are not matching corrected controls. Other feature/model research still needs its own corrected paired replay before interpreting those historical scores under this optimizer. Historical availability/publication timing, accepted buy-price accounting and legacy chip/FT policy limitations remain. No commit, push or merge was performed.


## 2026-10-31 — PR #98 extraction and evidence preservation

Runtime correctness was extracted separately in PR #100; main already contained
the solver certificate and exact FT rollover fixes. The research integration
preserves the source PR's historical experiments byte-for-byte, with its original
registry at `experiments/pr98_split/source_results.csv` and a per-path split
manifest. The combined registry retains both branches' fields and records.
Active interfaces retain main's explicit recipe, cold-prior origin, solver and
scoring admission, with compatibility for labelled historical net exports.
Optional CNN drivers skip when their separate Torch runtime is unavailable.
This is code integration and synthetic regression verification, not new forecast
or realized-points evidence, a rerun certificate, or production promotion.
Licensing is separated into draft PR #101; these changes retain main's terms.
