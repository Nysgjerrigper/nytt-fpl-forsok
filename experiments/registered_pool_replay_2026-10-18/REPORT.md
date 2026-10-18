# Captured-register ownership replay: paired attempt failed

The candidate did not prove optimality within the registered 60-second per-solve
limit. The adapter rejected the solve. No candidate score or paired confidence
interval exists. The original legacy outputs remain ineligible for an ownership-
effect conclusion, and issue #13 stays open. Production remains unchanged.

## Protocol and execution

[Preregistered protocol](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/13#issuecomment-6247100879)
SHA256 `b560336e14eaa122d628bccb702bb64f80d45b8720b97962dcdcbe6536ad9400`:
preserved full-feature/no-ownership origin forecasts, selection GW137–152,
zero refits, H3, same Solio backend, captured registered-forecast-row pools,
origin prices with acquisition-cost half-profit sales, fresh1000/FT1, modern FT5,
no chips. This is a separate policy comparison from the legacy ablation.

One attempt, maximum32 solves,60 seconds per solve,2100-second parent wall limit,
zero retries. Parent exit1 at180.97s; worker failure at180.50s. The control exported
16 decisions; no candidate decisions were exported. The exact attempted-solve
count and candidate partial decisions were not recorded and cannot be inferred.
The configured zero gap and native-optimal acceptance were not relaxed.

## Completed control engineering checks

All16 control squads belong to their corresponding supplied capture-time registers.
The original future-registration counterexample is eliminated for this arm.
The solver input omitted actual points and minutes; complete outcome-blind
execution remains unassessed because the parent/raw files contain outcomes.

[Settlement addendum](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/40#issuecomment-6247125272)
pinned the16-event collection before reading totals. The reviewed PR46 code used
unchanged per-event capture/register/complete380-fixture-ledger checks and shared
corrected scoring. Strict control settlement completed: gross983, transfer hits24,
net959, with12 separately audited retrospective blank player-round zeros.
Original forecast/outcome bytes are unchanged. No candidate comparison was computed.
These control-only totals cannot be compared with the old policy to infer ownership
or price effects: registered pool, transfer rules and price assumptions differ.

## Evidence and limits

All214 files in the external `OwnershipRegisteredPoolReplay_20261018` task-artifact
bundle were rehashed against attempt-seal.json. The tracked seal and settlement
receipt index the full external bundle, not just this small published subset.
Plans, source hashes, worker, raw forecasts,16 captures, decision input copies,
control decisions, logs, timeout traceback and strict settlement remain preserved.
The failed attempt is recorded in registry.csv; it is not a completed comparison.

Capture-time registration is not exact-deadline proof. Advance fixture publication,
missing forecast-cell planning, full independent financial/rule legality, exported
native certificate completeness and complete outcome blindness remain unassessed.
Core native-status guards do not establish those broader gates. No model promotion,
backend migration, fresh-holdout result or general runtime reliability estimate.

The next engineering step in issue #6 is opt-in per-origin start/completion/failure
receipts and immutable decision checkpoints. Measure preparation, vendor-call and
successful native runtime separately; failed native/build timing is unavailable
without additional instrumentation. Do not retry or automatically resume this sealed
attempt; any later evaluation needs a separate registered protocol and budget.
