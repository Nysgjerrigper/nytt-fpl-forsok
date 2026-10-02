# Current handoff

Verified-state checkpoint: 2026-10-02. This document describes the uncommitted
`fix/branch-recovery` integration draft. `main` has not been changed or merged.
Canonical operating rules are in [CLAUDE.md](CLAUDE.md); open work is in [TODO.md](TODO.md).
Older handoffs and reports are [historical snapshots](docs/archive/README.md).

## Engineering state

The recovery draft contains packaging/import cleanup, a separate immutable research
input, mutation-checked feature caching, exact native-crash containment, realized-minute
exports, corrected autosub/vice/chip scoring, and the fresh-build bank conservation fix.
No production model strategy or tuned parameter artifact has changed. Rejected player
selection, preseason and conformal candidates were not activated or ported.

A read race and a mutable admission marker in the older provenance/cache patch were
hardened. Scoring now handles an incomplete playing XI without incorrectly rejecting a
legal defender replacement, and Free Hit outputs the temporary squad. Tests cover BB,
TC vice activation, DGW minutes, legacy scoring and invalid-minute rejection.

The previous recovery screen completed with 51,004 member rows and 202 passing tests.
Its source bundle and hashes are retained under `experiments/branch_recovery_2026-10-02/`;
source files have since changed for this integration draft. Do not compare its original
code hashes to the current files and conclude its archived source is missing.

Independent verification: **432 tests passed**, with two non-failing environment/import
warnings. The full frozen dataset cached and uncached feature frames match exactly.
The wheel builds offline, imports outside the checkout, and preserves all 44 tuned
JSON artifacts plus the frozen-data manifest byte for byte. `git diff --check` is clean.

## Scoring lineage and control identity

| Evidence | Standard GW153-183 | Origin GW153-183 | Meaning |
| --- | ---: | ---: | --- |
| July preserved standing decisions | 2057 | 1880 | Legacy scoring without autosubs/vice activation. |
| September replay of unchanged decisions | 2088 | 1900 | Earlier corrected-scoring implementation; historical evidence. |
| Current same-input bank control | 2088 | 1901 | Corrected scoring including incomplete-XI replacements. |
| Current exact-bank candidate | 2088 | 1901 | Same predictions and scoring as the bank control. |

Both bank comparisons have difference **0, block-bootstrap 95% CI [0,0]**, with all
31 round scores tied and all 124 solves optimal. Some degenerate bench/squad choices
changed; lineup, captain and vice choices did not. The origin increase from historical
1900 to 1901 comes solely from a legal GW181 replacement in an incomplete playing XI,
with the historical selected decisions unchanged. It is a scoring correction, not a
forecast gain. These controls are not interchangeable model results. The input SHA hashes, control/candidate source difference, commands,
logs, squad CSVs and confidence intervals are preserved under
`experiments/engineering_recovery_2026-10-02/`. Do not call research `catboost_mae` artifacts
production `catboost` artifacts, or claim default parameters reproduce the tuned control.

GW191-221 confirmation scored 1705 standard/1499 origin under its historical protocol;
it is spent. Those results are not a current live forecast or rerun in this recovery.

## Research dispositions

- Registered neural studies remain 44/52 in preserved evidence. PyTabKit 1.7.3/macOS-arm64
  is blocked after repeated native crashes; one-epoch historical smokes are superseded by
  full-study crash evidence. TabR is incomplete. No neural launch is authorized by readiness tests.
- Stable-subset MoE lost 1806 versus the valid 2050 control, difference -244, CI [-372,-118].
  Against preserved 2057 legacy standing decisions it lost -251, CI [-402,-102]. These are
  distinct controls. No origin/seeds 1-2 rerun can rescue that registered candidate.
- MID per-player pilot failed its predeclared screening gate (MASE 0.61384 -> 0.62055).
  It stopped before MILP. No model promotion followed.
- The recovered four-model screen has better MASE for MAE-loss CatBoost but substantial
  downward level bias. It is exploratory only; no realized MILP/promotion claim exists.
- GW232+ remains prospective and is not admitted into this research recovery.

## Forecast guard recovery

Convex ensemble weights and finite, aligned member predictions are validated; every
strictly positive member is fitted and inactive members are skipped. Ambient tuned
artifacts cannot have a cutoff later than their training rows. Special model overrides
that were silently ignored now fail clearly. Static split partitions are checked before
fitting; baseline row reordering rebuilds blend/evaluation masks in prediction order.
Focused and full regression tests cover these changes; no model or parameter promotion
follows from these guards.

## Tournament and checkpoint recovery

Tournament fitting/prediction remains per fixture; comparison rows sum player-round
points and forecasts, with training-round MASE and observed training maxima. The
shared validator rejects incomplete/misaligned expert panels, mixed seeds and invalid
provenance. Future rows are excluded before preflight; existing output is never overwritten.
Missing minutes history retains the existing MID low-regime policy, including through
round aggregation and pivoting. Frozen preflight passed on 13,652 selection fixture
rows, including 901 extra DGW rows and 48 missing routing values, without model fits.

Opt-in `serial_trial_seed_v1` checkpoints bind data/feature bytes, exact folds, source
and runtime identity, hold exclusive POSIX ownership, and reproduce clean-resume
trial sequences for identical objective outcomes. Orphan trials and incomplete timeouts
fail closed. Exports are isolated and refused by legacy registered-study validators;
existing tuning studies and artifacts remain untouched. Native numerical determinism
is not proven by these engineering tests. Layer-two evidence is preserved locally under
`experiments/engineering_recovery_2026-10-02/`.

## Documentation audit

The repository had conflicting commit instructions, stale August handoff/TODO claims,
and obsolete neural-readiness advice. Current rules now have one canonical source in
CLAUDE.md; HANDOFF records verified status, and TODO contains only remaining work.
July audits/reports and previous instructions were preserved verbatim in the archive,
including paired report PDFs and sources. Mutable live data and frozen research data
are explicitly documented separately. Agent navigation and generated-file ignores
were updated. Historical datasets and research evidence were retained.

## Branches and remaining work

Five obsolete worktrees were fully archived with verified ignored files/checkpoints,
then retired. The merged `codex/position-specialist-moe` and `exp/position-moe` local
branches were deleted. Archive recovery instructions are in the dated recovery plan.

Keep the integration draft, original forecast audit (superseded guards and historical evidence), live API audit
(unique ingestion/audit edits), five-commit validation branch, and c1d1 engineering/evidence
until their remaining differences are reconciled. Historical remote experiments retain
unique code/design; they are preserved research history rather than current run priorities.

The PO has authorized committing, opening a PR and merging after verification. Review the integration draft as coherent
layers: packaging/provenance, scoring/budget, forecast/tournament guards, opt-in checkpoints,
and documentation currency. Live API ingestion and forced-chip recovery are now included;
the validation branch must not be merged wholesale.

## Active-season and forced-chip recovery

Official histories admit exactly checked events, retain fixture-level DGWs and validate
stable player/fixture/opponent identities. Fixture sides determine the club at match time;
post-round `ep_this` never becomes historical xP. API failures abort, archive-only mode
avoids official API calls, canonical season offsets preserve gaps, and output is atomic.

Positive chip targets are forced, invalid targets rejected, and WC/FH exempt transfer
hit bounds. First-period FH respects temporary-squad budget and permanent-state guards.
The legacy FT cap/reset policy remains unchanged. Same-input chip-disabled controls
retain standard 2088 and origin 1901, each difference 0 with 95% block-bootstrap CI [0,0].
All 124 solves were optimal. No forecast fits, live refresh or model promotion followed.
Evidence is under `experiments/engineering_recovery_2026-10-02/live_chip_recovery/`.

The [optimizer review](docs/OPTIMIZER_REVIEW.md) recommends evaluating a pinned Solio
adapter with explicit modern chip inventory checks. No replacement has been activated.
The [stakeholder overview](docs/STAKEHOLDER_OVERVIEW.md) explains purpose, goals and tools.
