# HANDOFF

Current branch: `probabilistic-buckets-2026-27` (renamed from `codex/probabilistic-bucket-models`)

Division of labour between the two probabilistic branches (decided 2026-07-08): this branch's
full 5-bucket distribution targets TEAM SELECTION (which players to buy/hold), while the sibling
`probability-of-loss-2026-27` branch's binary P(blank)/P(haul) view targets CAPTAINCY. Both are
kept; active work continues here. That sibling branch also carries the backtest re-baseline
(commit `8081f20`): the old 1966-point headline contained ~100 points of weight leakage, so the
honest realized-points baseline is ~1870. Any future realized-points claim from this branch must
be compared against ~1870, not 1966. The commit is deliberately NOT cherry-picked here to keep
the branches independent.

## Current objective

Explore whether FPL player points should be modelled probabilistically instead of only as a
single regression target. The current implementation is deliberately forecasting-only: it does
not replace the production CatBoost point model, does not write saved model artifacts, and does
not feed the MILP optimizer.

## What changed

- Added `fpl/model/probabilistic_buckets.py`
  - Reuses `Datasett/master_dataset.csv` -> `fpl.features.build_feature_frame(...)`.
  - Uses the same feature columns as the normal model pipeline.
  - Changes the target from one numeric `total_points` value to five ordered buckets:
    `<=0`, `1-2`, `3-5`, `6-9`, `>=10`.
  - Trains one model per position (`GK`, `DEF`, `MID`, `FWD`), matching the existing architecture.
  - Compares:
    - `catboost_bucket`
    - `lightgbm_bucket`
    - `xgboost_bucket`
    - `logistic_bucket`
    - `catboost_hurdle_bucket`
    - `lightgbm_hurdle_bucket`
  - Reports probabilistic metrics and decision-facing metrics:
    - multiclass log-loss and Brier
    - expected-points MAE/RMSE
    - blank and haul Brier/AUC
    - position-gameweek Spearman
    - MID/FWD captaincy `top1_capture`

- Added `tests/test_probabilistic_buckets.py`
  - Tests bucket mapping, probability normalization, expected-points derivation, and Brier behavior.

- Updated `RESEARCH_LOG.md`
  - Added the static-split bake-off result and interpretation.

- `AGENTS.md` now has a Codex-specific addendum for this branch.

## Commands run

```bash
.venv/bin/python -m fpl.model.probabilistic_buckets --quick
.venv/bin/python -m fpl.model.probabilistic_buckets
.venv/bin/python -m pytest tests/test_probabilistic_buckets.py tests/test_probabilistic.py
git diff --check
```

The full bake-off used the standard static split:

```text
train: GW_global <= 152
test:  GW_global 153-183
```

## Verification

```text
tests/test_probabilistic_buckets.py ....  [4 passed]
tests/test_probabilistic.py ............  [4 passed]
total: 8 passed
```

`git diff --check` passed.

## Main result

CatBoost is currently the strongest probabilistic family on this static split.

Pooled distribution quality:

| Model | bucket_logloss | bucket_brier | haul_auc |
|---|---:|---:|---:|
| `catboost_hurdle_bucket` | 0.6898 | 0.3605 | 0.8871 |
| `catboost_bucket` | 0.6904 | 0.3607 | 0.8875 |
| `xgboost_bucket` | 0.6980 | 0.3635 | 0.8797 |

The hurdle version is marginally best on log-loss. Plain `catboost_bucket` is nearly identical
and had better captaincy tilt:

```text
catboost_bucket        cap_tilt = 0.4532
catboost_hurdle_bucket cap_tilt = 0.4157
```

Unexpected signal: `logistic_bucket` has weaker probability quality but excellent captaincy capture
around `0.52`. Treat that as an anomaly/ranker signal to investigate, not as the calibrated probability
model to prefer.

## Important interpretation

This is not just a data-formatting change. It is a separate model family:

```text
existing regression:
features -> total_points

probabilistic bucket model:
features -> P(<=0), P(1-2), P(3-5), P(6-9), P(>=10)
```

From the predicted distribution we can derive:

```text
expected_points
P(blank) = P(<=0) + P(1-2)
P(haul)  = P(>=10)
risk/upside profile
```

## Walk-forward head-to-head (added 2026-07-08)

The static split above answered "can the distribution be learned?". The decision-grade question is
whether bucket-derived expected points hold up against the tuned production CatBoost regression under
honest conditions. `evaluate_walk_forward` now does that:

```bash
python -m fpl.model.probabilistic_buckets --walk-forward \
    --test-min-gw 153 --test-max-gw 183 --retrain-every 4
```

- Same protocol as `fpl.model.predict`: each GW predicted using only strictly-earlier data,
  retraining every 4 GWs.
- Baseline `catboost_regression` = `fpl.model.models.fit_model("catboost", ...)`, which auto-loads
  the tuned per-position params — the real production forecaster, not a strawman.
- Fairness fix for the old caveat: the bucket CatBoost now reuses those same tuned per-position
  hyperparameters (loss swapped MAE -> MultiClass/Logloss, `bootstrap_type=Bernoulli` pinned because
  CatBoost requires a sampling bootstrap when `subsample` is set). Disable with `--no-tuned-params`.
- Metrics now include `bias` and `total_calibration` next to MAE/Spearman/top1_capture. This is the
  mean-vs-median trap made visible: the production regression is MAE-trained, so it predicts a
  conditional MEDIAN and systematically under-predicts totals (total_calibration well below 1); bucket
  expected points are a true probability-weighted MEAN. Consequence: `ev_mae` structurally favours the
  regression and must NOT be the deciding metric — judge on ranking (spearman, cap_ev) and calibration.

### How the distribution maps to team selection (the conceptual claim of this branch)

One bucket model yields every quantity a selection decision needs, where production needs
separate models (and still lacks some):

- `E[points]` (probability-weighted bucket means) — what the MILP consumes.
- `P(blank)` = P(<=2) — the downside. Two players with equal E[pts] are not equal picks: a steady
  4-point defender and a 50/50 blank-or-8 midfielder differ exactly here (Roy's safety-first view).
  Candidate uses: bench ordering, and risk-profiling the 11 starters.
- `P(haul)` = P(>=10) — the upside tail; the sibling branch showed tilting by it lifts captaincy
  top1_capture 0.365 -> 0.429.

## Known caveats

- Do not wire this into `fpl.milp.optimize` yet. The MILP remains intentionally untouched.
- The full walk-forward run is expensive (~128 CatBoost fits over GW153-183 at retrain-every=4);
  use `--quick` and a short GW window to smoke-test changes first.
- LightGBM import can require writable temp/cache access because it imports matplotlib; in restricted sandbox
  contexts, run the bake-off with normal workspace write permissions.

## Recommended next step

The walk-forward ran and the buckets WON (full numbers + interpretation: RESEARCH_LOG.md 2026-07-08):
better ranking at all four positions (Spearman 0.703 vs 0.676), near-perfect level calibration
(1.00 vs the regression's 0.54), better RMSE and captaincy capture; regression keeps only raw MAE,
which is the median artifact. Next: route bucket E[points] into a predictions CSV
(`fpl.model.predict`-compatible format) and run the standard GW153-183/horizon-3 MILP backtest against
the honest ~1870 baseline. Realized points decide - remember the sibling branch's level-calibration
scalar also looked sensible and LOST 56 points.

## Handoff (2026-07-08, MILP backtest ran - RESULT IN)

**What:** Ran the recommended MILP backtest. Added `walk_forward_predictions_csv` /
`--export-predictions` to `fpl/model/probabilistic_buckets.py`, exported bucket E[points] in predict.py
CSV format, and ran the GW153-183 / horizon-3 MILP on it vs the tuned production regression. All 55 tests
pass.

**Result - the bucket model LOST on realized points:**

| Configuration | Realized points |
|---|---|
| tuned CatBoost regression (production) | **2107** |
| tuned CatBoost bucket E[points] | 2059 |

Both tuned, both 1 transfer/GW, both zero chips - clean comparison, -48 pts (-2.3%) for buckets.

**CRITICAL baseline correction:** the ~1870 target repeated above and throughout this handoff is the
*untuned* baseline. The bucket model uses tuned params, so the honest comparison is the *tuned* regression
= **2107** (reproduced exactly this session, matches 2026-07-06 Optuna). Against 1870 the bucket looked
like +189; against the correct 2107 it is -48. Any future bucket realized-points claim must beat 2107.

**Consequence:** Despite winning calibration (1.02 vs 0.54), Spearman, RMSE, and captaincy in the forecast
eval, the bucket E[points] built worse squads - the third instance of "better forecast metric ≠ more MILP
points" (see RESEARCH_LOG.md 2026-07-08). Keep the bucket model as forecasting-only + for its free
P(blank)/P(haul); do NOT route it into the production MILP as a point-forecast replacement.

**Status:** Backtest complete, RESEARCH_LOG.md updated with the decisive entry. Export code added but NOT
yet committed. Sibling `probability-of-loss-2026-27` still carries the untuned re-baseline commit.
