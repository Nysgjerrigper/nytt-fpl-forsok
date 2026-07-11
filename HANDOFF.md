# HANDOFF

Last updated: 2026-07-11. Current branch: `main` (clean, synced with `origin/main`).

## Repo state (start here)

- **`main`** holds the full production pipeline. The probabilistic-buckets experiment was
  fast-forward-merged in and its branch (`probabilistic-buckets-2026-27`) has been **deleted**
  local + remote — it lives on in `main`'s history, nothing lost. Before this merge `main` was
  27 commits stale (still the old R/LSTM code); it is now current.
- **`probability-of-loss-2026-27`** is the one remaining side branch: binary P(blank)/P(haul)
  classifiers (`fpl/model/loss_probability.py`) aimed at CAPTAINCY. Still open. It also carries
  commit `8081f20` (the untuned backtest re-baseline note) which `main` does not have.
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

- **Captaincy via P(haul)** — the sibling branch's E[pts]×(1+P(haul)) tilt lifted top1_capture
  0.365→0.429. The buckets give P(haul) for free. This is the most promising *live* use of the
  probabilistic work: a captain-selection tweak in `run_week.py`, not a MILP point-forecast swap.
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
