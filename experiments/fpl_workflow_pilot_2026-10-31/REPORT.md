# Bounded FPL workflow pilot — 2026-10-31

The workflow passed its engineering pilot. The RMSE candidate failed the registered points screen. Keep production unchanged; reject this pilot as evidence of improvement.

## Registration and configuration

Hypothesis: with matched 50-tree recipes, RMSE loss improves mean calibration and origin net points relative to MAE. Control: catboost_mae; candidate: catboost_rmse. Global GW 141–142; seed 0; depth 6, learning_rate 0.05, thread_count 2, retrain_every 4, horizon 3; all canonical features retained. One attempt, worker cap 180s, solve cap 30s; no retry. These are previously observed selection rounds, not a final holdout or tuned-production comparison.

[Preregistration](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/104#issuecomment-6545844725) predates execution. Exact criteria are in [question.json](question.json), full resolved recipes and source/data/runtime hashes in [plan.json](plan.json).

## Measured forecasts

| Protocol / lead | Arm | MAE | RMSE | Bias | Total calibration | Spearman | Top1 capture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard / None | control | 0.778 | 2.000 | -0.525 | 0.489 | 0.688 | 0.509 |
| standard / None | candidate | 0.969 | 1.812 | 0.064 | 1.062 | 0.680 | 0.370 |
| origin / 0 | control | 0.783 | 2.006 | -0.528 | 0.489 | 0.688 | 0.509 |
| origin / 0 | candidate | 0.971 | 1.817 | 0.060 | 1.059 | 0.682 | 0.370 |
| origin / 1 | control | 0.794 | 1.978 | -0.497 | 0.516 | 0.639 | 0.389 |
| origin / 1 | candidate | 1.014 | 1.841 | 0.107 | 1.104 | 0.655 | 0.481 |

Calibration is predicted/actual aggregate points (ideal 1); bias is predicted minus actual. Fixtures are aggregated to player rounds, and origin leads stay separate. MASE is not measured without training-only scale receipts.

## Corrected MILP net points

| Protocol | MAE control | RMSE candidate | Candidate minus control | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| origin | 140 | 120 | -20 | [-20, -20] |
| standard | 154 | 125 | -29 | [-29, -29] |

The comparator clamps requested block length 3 to the two-round sample length 2. Only one full-length moving block is available, so every resample has the same total, producing degenerate intervals. Their zero width is a resampling artifact, not precise uncertainty or evidence of generalization. The one-sided candidate-superiority sign-test p-value is 1 in both protocols; two observations provide little inferential information.

## Skeptical research review

This is a primary-session critique; a separate result reviewer was unavailable. A separate agent reviewed the workflow implementation and skill before merge.

RMSE and mean calibration improved, while MAE and realized squad net points worsened. This demonstrates why forecast error alone cannot determine promotion. The points screen fails regardless of the calibration gains. One seed and two previously observed rounds cannot establish loss-family superiority, cross-period robustness or live performance. Standard forecasts contain documented lookahead. Origin form is frozen, but fixture/price information limits remain; proven optimality applies to the encoded MILP and does not independently certify legality or deadline information. The scorecards remain UNVERIFIED/unranked. Final-evaluation, spent and prospective outcomes were not selected by this workflow.

## Engineering outcome and recommendation

Completed 16 measured fits and 8 native-optimal solves. Worker measured elapsed 28.63s (fit 8.40s; solver 10.13s), within the 180s worker cap. All captured outputs verified through read_result. A separate registration-only reproduction passed without launching compute.

Keep the workflow; reject this pilot as evidence for a model switch. Next: audit the existing loss/ablation evidence under #61/#68 before proposing one missing contrast. For a new minutes hypothesis, predeclare a supported same-model minutes-feature ablation and a new total budget before compute; do not repeatedly optimize this pilot window.

## Reproduce

This evidence PR adds the reproduction script while leaving FPL runtime source at `1302373`. From this pilot checkout, with the recorded library versions and admitted frozen CSV, run from the repository root:

```sh
PYTHONPATH="$PWD" python experiments/fpl_workflow_pilot_2026-10-31/reproduce.py --output /tmp/new-registration
# Explicit execution only, using a different new directory:
PYTHONPATH="$PWD" python experiments/fpl_workflow_pilot_2026-10-31/reproduce.py --output /tmp/new-pilot --dataset /path/to/frozen.csv --execute
```

Source/runtime drift is rejected rather than silently rerun with different settings. [evidence.json](evidence.json) contains compact verified metrics and the byte manifest; [registry.csv](registry.csv) is the existing fpl.experiment output. Raw inputs, predictions, decisions, receipts and logs remain in the primary checkout under the ignored artifact directory named in the evidence file; they are not bundled here. Hash verification establishes captured bytes, not independent scientific validity.
