# HANDOFF

Last updated: 2026-07-11. Current branch: `main` (clean, synced with `origin/main`).

## Repo state (start here)

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
- `fpl/models/` is gitignored but **do not delete it**: it holds the tuned-params JSONs (expensive
  Optuna output) that produce the 2107 baseline. (The `.members.joblib`/`.weights.json` ensemble
  artifacts that used to sit alongside them were deleted 2026-07-11 with their never-called
  save/load path - audit finding A1; nothing ever loaded them.)

## The numbers that anchor everything: the honesty ladder (2026-07-18 update)

Re-baselined twice on 2026-07-11 - first after the DGW-leak fix, then again after the capped
re-tuning - certified once on the frozen 2025-26 window, re-baselined 2026-07-16 after the xP
zero-round mask (statistical tie with 2060, kept on data-correctness grounds), then the
origin-based cell refreshed 2026-07-18 (same tie verdict; see RESEARCH_LOG for the reproduction
note on a 13pt solver tie-break drift found along the way). Full lineage in RESEARCH_LOG; the
standing numbers, identical config (capped-tuned single:catboost, horizon-3 MILP):

| Window | standard protocol | origin-based (deploy) protocol |
|---|---|---|
| GW153-183 (selection window) | **2086** = the COMPARISON baseline | **1906** |
| GW191-221 (one-shot, now SPENT) | 1705 | **1499** = the honest live expectation |

- Judge every model/feature claim against **2086**, standard protocol, same window, with a
  `fpl.milp.compare_backtests` CI. (Ties within ~+/-140 points are not distinguishable on
  this window - the old-vs-retuned params comparison measured exactly that.)
- Quote **~1500 per 31 GWs** for "what would this score live": selection-free window AND
  live information set. That comes from the separate frozen GW191-221 confirmation
  (1705 -> 1499, winner's-curse gap), not the GW153-183 window above.
- **GW191-221 must never be used for selection again.** Next confirmation: GW222+ / 2026-27.

Retired anchors: 2107 (DGW-leaking features), 2060 (pre-xP-mask; statistical tie with 2086),
2041 (pre-cap params; statistical tie with 2060), 1966/1870/1900/1811/1526 (earlier eras). Relative conclusions from those eras stand;
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
- **Fair registry tuning** — only CatBoost is Optuna-tuned; LightGBM/XGBoost run on defaults, so
  the model bake-off is tuned-vs-defaults. Tuning them would make the comparison honest (noted as
  a follow-up in the 2026-07-06 Optuna log entry).

## Standing rules for any modeling change (don't relearn these the hard way)

- Judge on the realized-points MILP backtest vs 2107, or at minimum top1_capture / calibration
  diagnostics — **never on MASE/MAE movement alone** (mean-vs-median trap, demonstrated 3×).
- Fit any combination weights / calibration on a window strictly BEFORE what you predict (the
  leakage bug, RESEARCH_LOG.md 2026-07-04).
- Log every experiment to `experiments/results.csv` + a RESEARCH_LOG.md note, negatives included.
- `pytest tests/` must be green before proposing a commit (currently 55 passing).
