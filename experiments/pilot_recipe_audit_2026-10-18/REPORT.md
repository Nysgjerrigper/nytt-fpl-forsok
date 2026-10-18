# Correction: the four-round pilot has matched capacity

The preserved `TrainingGroundRunnerPilot_20261017/first` experiment uses identical
ordered features and explicit non-loss hyperparameters for MAE/control and
RMSE/candidate: depth6,300 iterations, learning rate0.05, seed0 and two threads.
All16 fit-start and16 fit-completion receipts match the original plan. The eight
protocol/position pairs also match training cutoff, row counts and feature count.

Requested recipe differences contain only `loss_function`. Effective fitted
CatBoost parameters differ in `loss_function`, `eval_metric` and the loss-dependent
`leaf_estimation_method` (MAE Exact versus RMSE Newton). This is a matched-capacity
loss-family contrast, not identical internal training algorithms and not a
separately tuned capacity comparison.

Later origin-price replay notes incorrectly applied the older recovered study's
capacity-confounding caveat to this pilot. This audit corrects that description.
The original origin-price plan/result/evaluated script retain their original bytes,
including the incorrect limitation, to preserve their recorded identities; this
addendum supersedes that recipe interpretation. Do not use the old script's
limitation text for a new comparison without updating and registering a new plan.

The older October3 recovered study really did tune MAE/RMSE separately; its
capacity caveat remains valid and unchanged.

No models were fitted and no optimizer was called. All original forecasts, decisions,
settlements and point totals remain unchanged. The modern-price pilot remains
203 candidate versus197 control, +6 with paired95% CI[-8,+11], four previously
inspected selection rounds and one seed. It remains inconclusive, UNVERIFIED and
unranked. Price/blank information, fixture causality, complete outcome-blind
execution and promotion limitations remain unchanged.

Reproduce this receipt audit with a new output directory:

```sh
PYTHONPATH=. python experiments/pilot_recipe_audit_2026-10-18/audit.py \
  /path/to/TrainingGroundRunnerPilot_20261017/first /path/to/new-audit
```

`result.json` pins the original plan, completion and32 fit receipts plus this audit
script. The original completion verifier also checks all preserved output bytes.
The external `PilotRecipeAudit_20261018` bundle preserves the audit result/registry
and hashes; its receipt was checked against the tracked copies.
