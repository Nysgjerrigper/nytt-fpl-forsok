# Backtest comparison contracts

Historical callers retain the gross `actual_total_points` default. New comparisons can
explicitly request `--points-col actual_net_points`: net equals gross minus four points
per `alpha` paid transfer. Missing hit counts are not assumed zero. If a saved net column
exists, it must agree. Invalid values, unknown/mixed scoring protocols, duplicate weeks
and mismatched windows are rejected. A comparison with no protocol labels on either side
remains usable as **UNVERIFIED** historical evidence, not a promotion verdict.

```sh
python -m fpl.milp.compare_backtests candidate.csv control.csv \
  --points-col actual_net_points --json-output new-comparison.json
```

The receipt includes exact CSV hashes, metric, weeks, bootstrap configuration and paired
results. Both CSVs are parsed from the same bytes that are hashed. The positive bootstrap
replicate fraction is labelled separately from the sign-test p-value. Serial dependence,
block choice and repeated inspection limit inference; no automatic promotion follows.

## Strict comparison admission

Supply both `--contract-a candidate-contract.json` and `--contract-b control-contract.json`.
Schema version 1 requires:

- `schema_version`, `purpose` (`framework_validation` or `registered_research`), `run_id`,
  `intervention` (`forecast`, `features`, `optimizer`, `reporting`), `backtest_sha256`, `metric`.
- `context`: `season`, `start_season`, exact sorted `gameweeks`, `information_protocol`
  (`standard` or `origin`), positive `horizon`, `scoring_protocol`, `price_policy`,
  `sell_price_policy`, `initial_state`, `rule_set`, `chip_inventory`, `runtime`.
  State, chip inventory and runtime are explicit nonempty objects, including disabled chips.
- `artifacts`: `dataset`, `outcomes`, `player_pool`, `predictions`, `parameters`, `features`,
  `model_source`, `optimizer_source`, `scorer_source`. Each has `path` and `sha256`;
  relative paths resolve from the manifest's directory. Every file is verified.

All context must match. Forecast interventions may change predictions, parameters and
model source. Feature interventions may also change the feature artifact. Optimizer
interventions may change optimizer source only; reporting comparisons change none.
Dataset, outcomes, pool and scorer bytes must always match. Different experiments may
need a separately reviewed contract extension; arbitrary ignored fields are not allowed.

**CONTRACT_VALIDATED means byte-bound declared comparability.** It does not prove causal
validity, pre-registration, full player coverage, legality or native solver optimality.
Promotion remains NOT ASSESSED here. Existing registered MoE gates and gross controls
are unchanged; new net-primary families require their own approved protocol.

## Preserve decisions for later replay

The legacy optimizer supports an optional `--decisions-json new-decisions.json`.
This immutable snapshot is written before realized settlement. It preserves stable player
IDs, original bench priority, captain/vice, transfers and state transitions, independently
of display-name sorting in the human CSV. New HiGHS runs require both PuLP optimal
statuses, finite feasible integer values, native optimal status, valid primal/dual
evidence and zero relative/absolute stopping gaps. The certificate uses a predeclared
1e-7 absolute numerical tolerance and checks the PuLP objective against the native
objective. CBC is labelled PULP_REPORTED_OPTIMAL, without a native certificate.
Older snapshots remain NOT ASSESSED; an objective value or PuLP status alone does not
retroactively certify them. No certificate establishes independent FPL-rule legality.
Failed solves abort the entire export rather than emitting fallback decisions. An existing
snapshot is never overwritten. CSV and JSON destinations must be distinct.

New settled legacy CSVs add `transfer_hit_points` and `actual_net_points`; historical
`actual_total_points` stays gross. Forecasts, optimization objectives, selected squads
and production configuration are unchanged by these reporting additions.
