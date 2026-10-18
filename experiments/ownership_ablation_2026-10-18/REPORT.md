# Ownership-feature ablation: retain the full-feature control

Removing historical ownership features did not produce consistent decision gains.
The standard backtest gained7 net points; the origin backtest lost48. Both paired
intervals include zero. Retain the full-feature control; this does not establish
a general causal benefit from ownership or justify production promotion.

## Registered comparison

[Issue #13 registration](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/13#issuecomment-6246841017)
preceded all fits and new decisions. Plan SHA256:
`0daba4410faaa6d8e8ba761ed432b9f5d3a47a8578d1887977f6113828ba379d`.
Companion registration SHA256:
`e092cdd515d1e9497942f021f7563255201afc017b5292f0dfba71b3d0333d71`.

Control:115 features. Candidate:110, removing only selected_prev, selected_roll3,
selected_roll5, selected_career_avg and selected_season_avg. Both use CatBoost MAE,
depth6/300 iterations/learning rate0.05, two threads and seed0. All32 paired
protocol/position/cutoff cells match requested and effective fitted parameters.

Selection GW137–152, refit every4 rounds, horizon3 truncated at the window end.
Fresh £100m state, initial FT1, no chips, legacy FTcap2/gain1 and target-row price
policy. This research recipe does not reproduce the tuned production baseline.
Ownership is data; the experiment tests its contribution, without classifying it
as pundit opinion.

## Realized decision outcomes

| Protocol / arm | Gross points | Transfer hits | Net points |
| --- | ---: | ---: | ---: |
| Standard / full control | 1034 | 4 | 1030 |
| Standard / no ownership | 1057 | 20 | 1037 |
| Origin / full control | 1003 | 16 | 987 |
| Origin / no ownership | 971 | 32 | 939 |

| Candidate minus control | Net difference | Paired95% interval |
| --- | ---: | ---: |
| Standard | +7 | [-55,+61] |
| Origin | -48 | [-113,+4] |

Each comparison uses16 weekly pairs,10,000 moving-block resamples, block3, seed0.
Standard gross gains were mostly offset by16 extra hit points. Origin lost32 gross
points and incurred16 extra hit points. These are arithmetic descriptions, not a
causal diagnosis of why individual transfers changed. Results are UNVERIFIED and
unranked; correlated rounds and repeated selection-window inspection limit inference.

## Forecast diagnostics are not the verdict

Standard MAE changed from0.78085 to0.77946, while RMSE changed from2.07355 to2.07601.
Origin lead0 MAE changed from0.78323 to0.78188, while RMSE changed from2.07771 to2.08019.
The small typical-error reduction did not translate into better origin net points.
The complete standard/origin summaries retain bias, calibration, ranking, top-player
capture and positional support; origin leads1/2 are separate. Top-player capture
is a forecast diagnostic, not measured captaincy contribution. Training-scaled MASE
and calibrated probabilistic evidence remain unmeasured in this comparison.

## Execution and verification

One attempt completed:64 fits and64 native-optimal decisions in551.2s total,
with194.3s captured fit-call time and295.1s optimizer-stage wall time. The latter
includes the optimizer invocation, not just native solver work. Other phase times
are not inferred. One completed run is not a general reliability estimate.

Recomputed all170 captured output hashes, verified all64 fit-start/completion
pairs,32 matched recipe/training-support cells,64 stable-ID native certificates,
comparison CSV hashes and gross/net/hit arithmetic. Zero native stopping gaps
were configured. A reported relative gap of1.53e-16 required the existing numerical
certificate tolerance rather than an exact floating-zero post-check; largest
primal/dual difference was4.27e-14. No solve was rerun or accepted at a looser gap.
The corrected post-check is recorded in verification.json.

## Limits and reproduction

These are previously inspected selection rounds and one seed. Standard multiweek
planning sees later form; origin form is frozen, but target-row prices, fixture
knowledge and player registration are not fully deadline-qualified. Legacy missing
outcome zero-fill has not passed separate strict selected-player settlement for
this run. Full independent site legality and complete outcome-blind execution remain
NOT ASSESSED. Native optimality applies to the encoded model, not those wider gates.
No later reference/holdout evaluation or production selection was performed.

The complete immutable raw bundle, dataset copy, forecasts, decisions, logs and
automatic scorecards are externally preserved under
`TrainingGroundOwnershipAblation_20261018/first` in task artifacts. The copied
complete.json/worker-complete.json index that full bundle, not just these tracked
summaries. Original registry rows and all results remain unchanged.

```sh
PYTHONPATH=. python -m fpl.research_runner run \
  experiments/ownership_ablation_2026-10-18/plan.json \
  --dataset /path/to/the/preserved/frozen-GW231.csv --output /path/to/a/new-attempt
```

A rerun must respect the registered source/runtime identities and use a new explicit
attempt; it is not needed to reproduce the saved report. Use the runner's report
command on the full bundle to recheck its preserved outputs.
