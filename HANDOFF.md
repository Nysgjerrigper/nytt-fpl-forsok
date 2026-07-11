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
  Optuna output) that produce the 2107 baseline. The `.members.joblib`/`.weights.json` ensembles
  there are cheap to regenerate (`python -m fpl.model.train`); the tuned params are not.

## The number that anchors everything: 2107

The honest production baseline on the standard **GW153-183 / horizon-3** window is **2107**
realized points — tuned single:catboost regression through the MILP (2026-07-06 Optuna result,
reproduced 2026-07-08). **Not** the ~1870 figure that older notes cite: 1870 is the *untuned*
baseline. Any tuned model's realized-points claim must be compared against **2107**. Reproduce:

```bash
python -m fpl.model.predict --start-gw 153 --end-gw 183 --retrain-every 4 \
    --weight-strategy single:catboost --output <preds.csv>
python -m fpl.milp.optimize --predictions-csv <preds.csv> \
    --start-gw 153 --max-gw 183 --horizon 3     # prints "Total actual points over horizon"
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
