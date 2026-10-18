# Checkpointed ownership replay: failure isolated to candidate GW139

The candidate did not prove optimality at GW139 within the declared120-second
native solve limit. Two candidate rounds and all16 control rounds were saved.
There is no paired point result, confidence interval or ownership-feature verdict.
Issue13 stays open; production remains single:catboost.

## Registered attempt

[Registration](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/13#issuecomment-6249794554)
plan SHA256 `80abc0bc04662cebacb5e29faf7be6f94afbd208227bf35f18abd20167578a6d`,
source027ec0e/PR49. Same preserved full115-feature/no-ownership110-feature CatBoost
MAE forecasts, selectionGW137–152/seed0/H3, zero fits. Both arms use captured
registered-forecast-row pools, origin-price/half-profit acquisition accounting,
fresh1000/initialFT argument1 (fresh solverFT0), modern FT5 and no chips.
The plan's original_registration_sha256 identifies the original registered plan,
not its separate companion registration document.

One attempt, maximum32 vendor-wrapper calls,120 seconds per solve,4200-second
parent wall limit, no retry or gap relaxation. This is explicitly separate from
the earlier60-second attempt, which remains preserved. It is not a price-only
sensitivity or a comparison with the legacy historical-policy totals.

## Observed operational result

| Arm | Attempted wrapper calls | Normal returns | Saved decisions | Terminal |
| --- | ---: | ---: | ---: | --- |
| Full control | 16 | 16 | 16 | Completed |
| No ownership | 3 | 2 | 2 | Failed at GW139 |

Parent exit1 at691.77s; worker failure at684.67s. The failed call's vendor wall
was122.83s, which includes native execution and other vendor work. Its native
runtime/build split and incumbent/bound/gap were not exported and remain unknown.
Wrapper counts are not a claim about internal native invocations. No partial
candidate score or shortened-window comparison was computed.

Measured successful native time: control395.585s across16 returns; candidate104.745s
across2 returns. Total vendor wall time:436.281s control,233.262s candidate (including
the failed call). Measured preparation:1.279s control/0.212s candidate. These are
observations from this instrumented attempt, not evidence of a speed change or
isolated model-build time. Missing timing totals are null with measured count0.

## Independent checks on saved decisions

Rehashed all181 sealed raw files and checked every event's sequence, start binding,
origin order and exact terminal counters. All18 saved squads belong to their
corresponding capture registers; the16 control snapshot rows equal their saved
checkpoint decisions. Candidate prefix checkpoints remain preserved even though
no complete candidate CSV/snapshot was emitted.

The existing independent financial auditor passed16 control and2 candidate rounds
against capture prices, FTcap5/gain1 and half-profit sales. In a separate derived
copy only, first-round purchases were checked against the complete selected squad
and represented as initial acquisitions rather than a paid-transfer list. Raw
snapshots/checkpoints were unchanged. This verifies no-chip budget/holding/transfer/
hit arithmetic for those saved origins, not full site legality or exact-deadline
price/registration proof. No new realized settlement was performed: both complete
arm snapshots were required before reading totals, and the candidate never completed.

## Preserved evidence and consequence

Full raw bundle: `OwnershipCheckpointedPoolReplay_20261018` in task artifacts.
Post-checks: `OwnershipCheckpointedPoolReplayAudit_20261018`; both the initial
and clarified derivative are retained. The initial derivative represented wholly
unmeasured timing sums as0 with measured count0; the clarified derivative uses null
and excludes configured budgets from elapsed timings. This was a reporting-only
post-check correction, with no refitting, resolving or raw-output edits.
Tracked seals index those external bundles rather than this small published subset.

The failing origin/state and admitted/prepared input identities now make a targeted
native profiling case possible. Next in issue6: preserve the exact model and native
status/bounds/gap on failure through a separately reviewed vendor diagnostic patch,
then profile this same mathematical case. Do not simply increase the timeout again.
Issue22 still needs a verified unranked failed-attempt scorecard/league adapter.

Fixture-publication/blank-forecast planning, full independent site legality,
complete exported native certificates and complete outcome blindness remain
unassessed. The parent/raw files contain outcomes. These are already inspected
selection rounds and one seed, not a fresh holdout or production-promotion study.
