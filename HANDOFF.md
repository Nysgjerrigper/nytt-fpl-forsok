# HANDOFF

Current branch: `codex/probabilistic-bucket-models`

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

## Known caveats

- Only a static split has been tested so far. Do not call this settled until there is a walk-forward comparison.
- The production model has tuned CatBoost params saved in `fpl/models/tuned_params_<POS>_catboost.json`; the
  bucket CatBoost currently uses the hand-set classifier defaults adapted from the regressor defaults.
- The probabilistic expected-points output has not yet been compared directly against the tuned production
  CatBoost regression on the honest backtest.
- Do not wire this into `fpl.milp.optimize` yet. The MILP remains intentionally untouched.
- LightGBM import can require writable temp/cache access because it imports matplotlib; in restricted sandbox
  contexts, run the bake-off with normal workspace write permissions.

## Recommended next step

Implement a walk-forward evaluator for `catboost_bucket`, `catboost_hurdle_bucket`, and the current tuned
CatBoost regression baseline. Compare:

- expected-points MAE/RMSE
- `bias` and `total_calibration`
- position-GW Spearman
- MID/FWD captaincy `top1_capture`
- blank/haul AUC and Brier

Only after that should this branch consider producing a predictions CSV or feeding any probabilistic signal
to the optimizer.
