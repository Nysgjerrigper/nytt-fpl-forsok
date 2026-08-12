# HANDOFF

Last updated: 2026-08-11. Branch/status must be refreshed by the integration executor before landing.

## Repo state (start here)

- **Position-specialist MoE research patch (2026-08-11, methodology-audited/unpromoted):** adds an explicit
  research registry, complete-map overrides for `predict`/`run_week`, an optional frozen MID gate,
  tournament-manifest writer, and promotion-gate CLI. Production remains `single:catboost`; no
  specialist tournament, tuned artifacts, backtest, CI, or promotion result is claimed. The methodology
  audit **PASSED**: runtime season-aware cutoffs, causal OOF/training-only MID MASE provenance,
  fail-closed tuned-parameter and finalist/control lineage, and structural spent-window rejection now exist.
  **Dependency check (verified):** PyTabKit 1.7.3
  on macOS arm64/Python 3.14.6 completed CPU fit/predict for RealMLP (13.08s, `n_epochs=1`, seed 17)
  and TabM (1.41s; same-seed max absolute prediction difference 0.0); `faiss-cpu==1.15.0` installs
  through pip. TabR additionally requires `skorch==1.4.0`; after installation it reached epoch-0
  validation but produced no bounded-run prediction. Keep TabR unavailable/incomplete, without a
  substitute. `requirements-research.txt` records the optional dependency set. The current derived
  chronology is GW<=136/GW137-152/GW153-183. Audit PASS is synthetic readiness validation only.
  **Fresh integration execution:** `pytest tests/ -q` passed (178 tests); standard and origin CLI
  smoke exports completed for GW153 using an explicit all-CatBoost map; frozen-artifact promotion
  smoke correctly retained CatBoost. The full generated plan has 56 tuning commands, but cannot
  complete in the current Python-3.11.15 environment: its exact TabR selection command fails at
  trial 0 because `faiss` and `skorch` are absent. Fresh bounded RealMLP/TabM probes also produced
  no usable predictions. No real selection, final backtest, CI, or promotion evidence exists.
  **Supervisor follow-up:** the selection CLI regression is covered and `pytest tests/ -q`
  now passes 179 tests. In the isolated Python-3.14.6 environment, RealMLP and TabM
  have bounded validation-aware fit/predict evidence; TabR again reached epoch-0 only
  and remains unavailable/incomplete. An explicit 13-expert available-set plan produces
  52 tune commands but is non-promotable because it excludes TabR. Its first unchanged
  `GK/catboost_mae` command completed two trials before the execution session ceiling;
  no tuned artifact exists. Resume that exact command in a runtime allowing its estimated
  6--12 minutes; do not reduce trials/timeouts or treat synthetic CLI smoke as evidence.
  **Persistent-session update:** the exact first command has now completed all 50 trials.
  Its validated GK/catboost_mae selection artifact is gitignored at
  `fpl/models/tuned_params_GK_catboost_mae.json` (file SHA-256
  `e7ffb57b6a9c85cf5c73f1b201d9b16a079fbd43e1af661228a23d881dc4f0ad`; embedded
  provenance hash `81a581b8e5f9054c9846f4e6a3b6c6ed8e5cb8c2f9d457a6a5d016b132842086`).
  It is one partial tuning result, not selection or promotion evidence.

- **Dataset provenance (2026-08-11):** rebuilt `Datasett/master_dataset.csv` has 162,981 rows through
  global GW228. It is ignored/untracked generated data; no dataset artifact was added to this patch.

- **`main`** holds the full production pipeline. The probabilistic-buckets experiment was
  fast-forward-merged in and its branch (`probabilistic-buckets-2026-27`) has been **deleted**
  local + remote — it lives on in `main`'s history, nothing lost. Before this merge `main` was
  27 commits stale (still the old R/LSTM code); it is now current.
- **`probability-of-loss-2026-27`** has been closed and archived as tag
  `archive/probability-of-loss-2026-27` (on origin): its captaincy idea was tested against the
  bucket module and came up a wash — see RESEARCH_LOG.md 2026-07-11 and "Settled" below. The tag
  still carries commit `8081f20` (the untuned backtest re-baseline note) which `main` does not have.
- Worktrees pruned to just the main checkout; stale agent branches deleted (all preserved on
  `origin/experimental/*`). Junk (caches, `.DS_Store`, a stray R-output txt, old regeneratable
  prediction/squad CSVs) cleaned out. `git status` is clean.
- **Subagent delegation infra (2026-07-23):** `.claude/agents/implementer.md` (Sonnet) and
  `.claude/agents/searcher.md` (Haiku) pin delegated mechanical work to cheaper models; CLAUDE.md's
  "Subagent delegation (cost control)" section defines the split and the quality-gate rule (main
  session reviews the diff and re-runs pytest itself). Validated on real tasks; produced
  `tests/test_config_strategy.py`, a PRODUCTION_WEIGHT_STRATEGY drift guard.
- `fpl/models/` is gitignored but **do not delete it**: it holds the tuned-params JSONs (expensive
  Optuna output) that produce the standing baseline. (The `.members.joblib`/`.weights.json` ensemble
  artifacts that used to sit alongside them were deleted 2026-07-11 with their never-called
  save/load path - audit finding A1; nothing ever loaded them.)

## The numbers that anchor everything: the honesty ladder (2026-07-23 update)

Re-baselined twice on 2026-07-11 (DGW-leak fix, then capped re-tuning), certified once on the
frozen 2025-26 window, re-baselined 2026-07-16 after the xP zero-round mask (statistical tie
with 2060, kept on data-correctness grounds), then again 2026-07-23 after the element-code
player-identity fix (TODO 4.8; another statistical tie, 2057 vs 2086, adopted on
data-correctness grounds - name-based identity had split 125 players across spellings and
merged 4 name-collisions). The origin-based cell was refreshed the same day post-identity-fix
(1880, tie with retired 1906). Full lineage in RESEARCH_LOG; the standing numbers, identical
config (capped-tuned single:catboost, horizon-3 MILP):

| Window | standard protocol | origin-based (deploy) protocol |
|---|---|---|
| GW153-183 (selection window) | **2057** = the COMPARISON baseline | **1880** |
| GW191-221 (one-shot, now SPENT) | 1705 | **1499** = the honest live expectation |

- Judge every model/feature claim against **2057**, standard protocol, same window, with a
  `fpl.milp.compare_backtests` CI. (Ties within ~+/-140 points are not distinguishable on
  this window - the old-vs-retuned params comparison measured exactly that.)
- Quote **~1500 per 31 GWs** for "what would this score live": selection-free window AND
  live information set. That comes from the separate frozen GW191-221 confirmation
  (1705 -> 1499, winner's-curse gap), not the GW153-183 window above.
- **GW191-221 must never be used for selection again.** Next confirmation: GW222+ / 2026-27.

Retired anchors: 2086 (pre-identity-fix; statistical tie with 2057), 1906 (origin-based,
pre-identity-fix; tie with 1880), 2107 (DGW-leaking features), 2060 (pre-xP-mask; statistical
tie with 2086), 2041 (pre-cap params; statistical tie with 2060), 1966/1870/1900/1811/1526
(earlier eras). Relative conclusions from those eras stand;
their absolute levels do not. Reproduce:

```bash
python -m fpl.model.predict --start-gw 153 --end-gw 183 --retrain-every 4 --output <preds.csv>
python -m fpl.milp.optimize --predictions-csv <preds.csv> \
    --start-gw 153 --max-gw 183 --horizon 3     # prints "Total actual points over horizon"
# deploy-honest variant: add --origin-based --horizon 3 to the predict call
```

## Settled this cycle: probabilistic buckets — forecasting-only, NOT a point-forecast replacement

`fpl/model/probabilistic_buckets.py` reframes the target from one `total_points` number into a
distribution over ordered buckets (`<=0, 1-2, 3-5, 6-9, >=10`), one model per position, from which
`E[points]`, `P(blank)=P(<=2)`, and `P(haul)=P(>=10)` all fall out of a single model.

The decision-grade MILP backtest ran (full detail: RESEARCH_LOG.md 2026-07-08):

| Configuration | Realized points |
|---|---|
| tuned CatBoost regression (production) | **2107** |
| tuned CatBoost bucket E[points] | 2059 |

Both tuned, both 1 transfer/GW, zero chips — a clean **−48 pts (−2.3%)** for the buckets. This
held **despite** the bucket model winning the forecast eval outright (level calibration 1.02 vs
0.54, Spearman 0.703 vs 0.676 at all four positions, better RMSE and captaincy). It is the third
demonstration that better forecast metrics don't buy squad points (the other two: the
level-calibration scalar lost 56; CatBoost's MASE edge bought zero — both 2026-07-06).

**Decision:** keep the bucket model as a forecasting-only research result and for its *free*
P(blank)/P(haul) distribution. **Do NOT route bucket E[points] into `fpl.milp.optimize` as a
point-forecast replacement** — it costs realized points. Reproduce the export/backtest:

```bash
python -m fpl.model.probabilistic_buckets --export-predictions \
    --test-min-gw 153 --test-max-gw 183     # writes bucket E[pts] in predict.py CSV format
# then the same fpl.milp.optimize horizon-3 command as above
```

Caveat as always: one window, one seed — but the direction agrees with two prior instances, so
it is not treated as noise.

## Open threads / candidate next steps (direction is the PO's call)

- **Position-specialist MoE tournament — methodology ready, no promotion decision.** Run only the
  registered workflow: selection-stage tuning at discovery cutoff -> fail-closed tuned manifest -> causal
  OOF/frozen selection -> hash-bound finalist/control artifacts -> promotion. Final assessment still needs
  real standard/origin MILP artifacts, seeds 0/1/2, and Holm-adjusted exact sign-test evidence; production
  stays `single:catboost`. TabR remains unavailable/incomplete and is not substituted.

- **Full-repo audit (2026-07-11):** `AUDIT_2026-07-11.md` holds a complete methodological/
  engineering review; its follow-ups live as the dependency-ordered "Audit follow-ups" clusters
  at the top of `TODO.md`. Headline findings: run_week does NOT run the validated production
  config (untuned params + NNLS — fix first), the tuning CLI lacks the GW<153 cap the log
  claims, DGW rows leak same-GW info into shifted features, and GW153-183 is overused as a
  decision window (2025-26 proposed as a one-shot confirmation holdout). Blocked on three PO
  answers listed in the audit's §9.

- **Captaincy via P(haul) — RESOLVED NEGATIVE (2026-07-11), not wired in.** The E[pts]×(1+P(haul))
  tilt was re-tested walk-forward tuned-vs-tuned: it helps one base model, hurts the other
  (sign-flip = noise), and the best captaincy number comes from plain bucket E[pts] with no tilt.
  Gate ("wire in only if it wins") failed; production captaincy stays on E[points]. See
  RESEARCH_LOG.md 2026-07-11. The sibling branch's 0.365→0.429 lift did not replicate.
- **Risk-aware bench/starter use of P(blank)** — untested; would need its own backtest, and the
  MILP consumes E[pts] only, so any use is downstream of the optimizer.
- **Fair registry tuning — RESOLVED (2026-07-18, TODO 2.4):** LightGBM/XGBoost were Optuna-tuned
  and `single:catboost` survived the tuned-vs-tuned comparison; the bake-off is now honest. See
  RESEARCH_LOG.md 2026-07-18.

## Standing rules for any modeling change (don't relearn these the hard way)

- Judge on the realized-points MILP backtest vs the standing baseline (**2057** as of
  2026-07-23), or at minimum top1_capture / calibration diagnostics — **never on MASE/MAE
  movement alone** (mean-vs-median trap, demonstrated 3×).
- Fit any combination weights / calibration on a window strictly BEFORE what you predict (the
  leakage bug, RESEARCH_LOG.md 2026-07-04).
- Log every experiment to `experiments/results.csv` + a RESEARCH_LOG.md note, negatives included.
- `pytest tests/` must be green before proposing a commit (currently 108 passing).
- Delegate mechanical work to the pinned subagents in `.claude/agents/` (implementer=Sonnet,
  searcher=Haiku) and verify their output in the main session — see CLAUDE.md "Subagent
  delegation (cost control)".
