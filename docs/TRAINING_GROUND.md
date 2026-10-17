# The Training Ground

The Training Ground is our offline playground for checking ideas before they reach
**Matchday**, the configured weekly system. Its first version compares saved squad
backtests, generates scorecards and maintains separate descriptive league tables.
It does not yet train models, orchestrate ablations, launch the optimizer or promote
anything to production. Those connections belong to the existing research-runner work.

```mermaid
flowchart LR
    A[Saved paired forecasts] --> B[Hash and exact coverage checks]
    B --> C[Player-round diagnostics by protocol and lead]
    D[Saved squad results] --> E[Gross and net points with paired intervals]
    C --> F[Evidence scorecard]
    E --> F
    F --> G[Comparable points league or unranked entry]
    G --> H[Research review before any promotion]
```

## Try a synthetic exhibition

Use the project's Python environment from the repository root:

```sh
python -m fpl.playground init-demo /tmp/fpl-exhibition
python -m fpl.playground validate /tmp/fpl-exhibition/spec.json
python -m fpl.playground run /tmp/fpl-exhibition/spec.json
python -m fpl.playground report experiments/runs/synthetic-exhibition/first
python -m fpl.playground league
python -m fpl.playground league --json
```

The demo directory must be new. Scores are deliberately synthetic and the result is an
**EXHIBITION**, never model-selection evidence. The league leaves its rank blank.
The example requires no API access, frozen CSV, model fitting or solver run.

## Compare saved backtests

A version-1 specification has the following structure. Replace placeholder hashes with
SHA-256 digests of the actual bytes. Paths resolve relative to the specification file.

```json
{
  "schema_version": 1,
  "id": "saved-comparison",
  "purpose": "historical_reanalysis",
  "hypothesis": "Re-examine preserved candidate and control outcomes after transfer costs.",
  "metric": "actual_net_points",
  "candidate": {
    "name": "candidate",
    "input": {"path": "candidate.csv", "sha256": "REPLACE_WITH_ACTUAL_SHA256"}
  },
  "control": {
    "name": "matching-control",
    "input": {"path": "control.csv", "sha256": "REPLACE_WITH_ACTUAL_SHA256"}
  },
  "bootstrap": {"n_boot": 10000, "block_len": 3, "seed": 0}
}
```

Use `framework_validation` for wiring checks and `historical_reanalysis` for saved
historical outcomes. Neither purpose permits model selection or automatic promotion.
Prospective rounds beyond the tracked frozen cutoff are refused. Previously inspected
and spent windows do not become fresh holdouts because we produce a new report.
The frozen cutoff is read from the tracked manifest; real fitting still requires the
separately restored, SHA-admitted dataset.

Quantile and bucket research entrypoints now use that same frozen snapshot rather
than the mutable weekly dataset. They verify the complete source bytes first, then
retain raw history only through the requested evaluation end (or tuning cutoff)
before constructing features. Earlier rows remain available for training and causal
form features. Static training must finish before the evaluation window. Bucket
tuning cannot exceed `config.TUNING_TRAIN_MAX_GW`; evaluation and spent rounds are
not admitted to its search. A missing or changed frozen snapshot stops the run.
This input validation does not fit models, validate a new result or promote a candidate.

## Explicit forecast recipes

The standard and origin forecasting APIs accept optional keyword-only `params_by_pos`
and `seed`. Supply both to bypass ambient tuned files. Each position needs exactly
the positive-weight registered members in its strategy, with a parameter dictionary
for each member. `resolve_fit_recipe` fills registered constructor defaults, preserves
the member's named loss/objective and applies the declared seed. Conflicting loss or
seed overrides fail before any member fits.

```python
from fpl.model.recipes import resolve_fit_recipe

positions = ("GK", "DEF", "MID", "FWD")
weights = {pos: {"catboost_mae": 1.0} for pos in positions}
overrides = {pos: {"catboost_mae": {"depth": 6, "iterations": 300,
                                  "learning_rate": 0.05}} for pos in positions}
resolved = resolve_fit_recipe(weights, overrides, seed=0)
# Pass params_by_pos=resolved, seed=0 and weight_strategy="single:catboost_mae"
# to walk_forward_predictions or origin_based_predictions.
```

Record the returned complete recipe, ordered features and runtime/library/source
identity in the experiment manifest. A recipe alone does not establish provenance.
The explicit forecast path currently supports single-member strategies, without
blend-weight fitting, level calibration or MID gates. Features must be unique,
present and exclude raw outcome labels; the research runner must additionally
admit the causal feature set and data window. Calling without either new option
keeps the existing production/tuned-artifact fitting behavior.

Each backtest needs one row per `gameweek` and finite `actual_total_points` (gross).
Net comparisons also require nonnegative integer paid transfers in `alpha`; a supplied
net column must agree with gross minus hits. Scoring labels must match if declared;
both absent means historical **UNVERIFIED** evidence. Missing hit counts are not zero.

For strict declared comparability, add a `contract` record with `path` and `sha256` to
both candidate and control. The format is documented in
[Backtest comparison contracts](BACKTEST_COMPARISONS.md). Reanalysis with contracts
requires contracts labelled `registered_research`; a framework manifest cannot be
relabeled to enter a ranked historical competition. Contract validation alone does
not prove that a study was registered in advance or that the decisions were valid.

## What is saved?

```text
experiments/runs/<experiment-id>/<attempt>/
  started.json
  spec.json
  candidate.csv
  control.csv
  candidate-contract.json / control-contract.json   # when supplied
  candidate-forecasts.csv / control-forecasts.csv   # when supplied
  comparison.json
  scorecard.json
  REPORT.md
  complete.json OR failed.json
```

Generated runs are ignored by Git. The completion receipt hashes the required results,
input snapshots and specification. Source and reporting-runtime identity are recorded.
Original model-fit time, failure rate, calibrated uncertainty, independent legality
and native solver proof remain explicitly unmeasured/unassessed by the reporting adapter.
Forecast diagnostics are measured only when both pinned panels are supplied.

## Adding forecast diagnostics

Add an optional `forecasts` object to the experiment specification:

```json
"forecasts": {
  "information_protocol": "standard",
  "grain": "player_round",
  "population": "Preserved optimizer common pool, including zero-filled rows",
  "candidate": {"path": "candidate-predictions.csv", "sha256": "<actual SHA-256>"},
  "control": {"path": "control-predictions.csv", "sha256": "<actual SHA-256>"}
}
```

Both panels need stable positive integer `player_id`, global `GW`, `position`,
`predicted_total_points` and `actual_total_points`. Their entire coverage, positions
and outcomes must match; the adapter never silently intersects different populations.
The declared rounds must exactly match the decision comparison. If strict comparison
contracts are present, their protocol and bound prediction hashes must also match.
Otherwise the forecast-to-decision linkage remains **NOT ASSESSED**.

Declare `grain: "fixture"` only for fixture rows, with a positive integer `fixture`
identifier. Predictions and outcomes are summed into player-rounds before evaluation.
Declare `player_round` for already aggregated exports; original fixture counts then
remain unknown. A common optimizer pool can contain zero-filled rows; label this in
`population` and do not present these diagnostics as recovered original model coverage.

The adapter reuses `fpl.model.metrics` for MAE, RMSE, signed bias, aggregate calibration,
within-position/gameweek Spearman and top-ranked player capture. It reports per-position
support, undefined/skipped rank groups and tied top predictions. Ties use the lowest
stable player code. Undefined ratios become JSON `null`; negative actual points can
produce ratios outside [0, 1], which are preserved. Capture is a ranking diagnostic,
not realized captaincy points. MASE stays **NOT MEASURED** without validated training-only
scales and refit-cutoff lineage; evaluation outcomes are never used to estimate a scale.

For `information_protocol: "origin"`, both panels also need `origin_gw`. Results are
separated by lead (`GW - origin_gw`, with zero meaning the origin's current target).
Repeated target outcomes are not pooled into one overall score. Standard forecasts
remain separate from origin forecasts and do not establish deployment-honest evidence.
These diagnostics do not change league ranking or promotion eligibility.

The JSON and report show the same points, difference, interval and evidence status.
The comparator supplies all point-accounting and statistical calculations. The positive
bootstrap fraction is not a probability of future superiority; the sign test has an
independence assumption and interval width depends on block choice.

The first attempt is called `first`. An unchanged completed attempt is reused only after
verification. A failed or interrupted attempt remains preserved. Retry deliberately:

```sh
python -m fpl.playground run experiment.json --attempt retry-1
```

Do not delete failed runs to force a restart. Reporting a saved completed run verifies
its captured outputs and does not require the original external paths to still exist.
Rerunning preflight does require the pinned original inputs. Contract snapshots retain
their original relative references: these are not complete portable research bundles.

## How the league works

- **CONTRACT_VALIDATED:** descriptive point ranks only within a matching historical
  cohort: same metric, control CSV, control input/source bytes, context and intervention.
- **EXHIBITION:** synthetic/wiring checks, visible but unranked.
- **UNVERIFIED:** historical comparisons without sufficient contracts, visible but unranked.
- **FAILED / INCOMPLETE:** preserved attempts, visible without invented scores.

Tied totals share a rank. Identical observations from retries appear once, rather than
counting as extra evidence. Standard and origin contracts, different windows, controls,
price/rule settings and gross/net metrics cannot share a ranked cohort.

A first-place entry is not automatically the champion. Native optimality, legality,
causal validity, seed stability and the registered promotion gate remain separate.
The configured Matchday strategy stays `single:catboost`.

## Run a bounded forecast experiment

`fpl.research_runner` connects the existing forecast, optimizer and scorecard APIs.
Start with a matched CatBoost MAE/RMSE contrast, or keep the same model on both arms
and remove a predeclared feature subset. Changing both factors together is rejected.
Plans contain resolved parameters, ordered features, a seed, source/data/runtime
identities and fit/solve/time budgets. Preparing a plan does not fit models.

```mermaid
flowchart LR
  A[Register matched contrast] --> B[Verify frozen inputs and budget]
  B --> C[Save standard and origin forecasts]
  C --> D[Corrected MILP backtests]
  D --> E[Forecast diagnostics and paired points intervals]
  E --> F[Preserved evidence and league entry]
  F --> G[Separate promotion review]
```

```sh
python -m fpl.research_runner prepare --dataset /path/to/frozen/master_dataset_gw231.csv \
  --start-gw 141 --end-gw 144 --output /path/to/plan.json
# Register the plan and its hash in the research issue before execution.
python -m fpl.research_runner run /path/to/plan.json \
  --dataset /path/to/frozen/master_dataset_gw231.csv --output /path/to/new-attempt
python -m fpl.research_runner report /path/to/new-attempt
```

Selection diagnostics and historical evaluation must each remain wholly within their
registered windows. Earlier history is retained; rows after the target window are
removed before features. Spent and prospective windows are rejected. The same frozen
bytes, code and library versions must still match at execution.

The worker saves all four standard/origin candidate/control forecast panels before
any optimizer run. Each actual fit has start/completion receipts including training
rounds and fitted parameters. Corrected net points and paired intervals accompany
forecast accuracy, calibration and ranking diagnostics. HiGHS must prove optimality
for every encoded MILP. A timeout or error preserves a failed attempt without retries.
The worker binds all non-log outputs before exiting. After it exits successfully,
the parent verifies those bytes and seals the final receipt with the closed log,
including native shutdown output. Completed reports verify every captured file
before displaying results. Older verified receipts remain readable.

These initial runner comparisons remain **UNVERIFIED** and unranked: their manifests
do not yet establish independent FPL legality, an outcome-blind interface or promotion
contracts. Observed historical windows are not fresh holdouts. A four-round pilot
checks the execution path; it cannot establish which model should run in Matchday.

## Opt-in history-only cold starts

`fpl.model.cold_start` supplies a separate path for registered players missing from
a forecast snapshot. It keeps their form and outcomes unknown, then uses a pooled
fixture prior rather than passing invented history to a fitted model:

```python
from fpl.model.cold_start import (
    add_registered_cold_starts, fit_pooled_priors, predict_cold_starts,
)

priors = fit_pooled_priors(history, origin_gw=global_origin)
extended = add_registered_cold_starts(snapshot, register)
new_rows = extended.loc[extended.is_cold_start]
new_fixture_priors = predict_cold_starts(new_rows, priors, origin_gw=global_origin)
```

`history` contains real observations only. Research callers must first admit the
canonical frozen dataset. The helper excludes rounds at or after the global origin,
includes DNPs, and shrinks each team-position distribution toward its pooled position
distribution with weight `n / (n + 20)`. Distinct DGW fixtures remain separate samples;
ambiguous or duplicate observations fail. Predictions are points per fixture.

The optional `build_future_predictions(..., cold_start_priors=priors)` connection
expands the fixture schedule and routes only known players through the normal/MID-gate
models. Its snapshot global origin must match the prior table. Without the option,
default outputs stay unchanged. Marked synthetic rows are rejected by feature building
and shared training entrypoints. This option is currently an API for controlled callers;
the weekly CLI does not enable it. Historical experiments require archived registration
and fixture evidence: today's register cannot establish past eligibility.

`prior_std` is historical population dispersion; `p_played` and `p60` are historical
fixture frequencies. They are **uncalibrated**, not live probabilities or confidence
intervals. `prior_count` is the raw team-position count (or pooled count on fallback),
not an effective posterior sample size. Their presence does not establish better FPL
points. Availability scaling is a separate heuristic and does not calibrate this prior.

Prior DataFrame attributes bind origin, latest admitted round, fixed strength and
history/table fingerprints. Keep the table and metadata together, for example JSON
with `{"metadata": priors.attrs, "rows": priors.to_dict("records")}`. Restore rows
with the exported `PRIOR_COLUMNS`, restore `.attrs`, then call
`validate_pooled_priors(table, global_origin)`. A standalone CSV loses this metadata.
Pandas fingerprints depend on runtime; external dataset/source manifests are still
required. The registered engineering smoke under
`experiments/cold_start_prior_smoke_2026-10-17/` tests cutoff invariance and this
round trip on real frozen history, without assessing forecast or MILP gains.

## Where the next connections belong

### Strict settlement replay

Saved schema-1 decision snapshots can be checked without refitting or solving:

```sh
python -m fpl.milp.replay run decisions.json predictions.csv \
  --reference squads.csv --output /path/to/new-replay
python -m fpl.milp.replay report /path/to/new-replay
```

Replay checks stable squad/XI/bench/captain identities, position counts, club limits,
formation and the original bench order. It requires explicit current-round outcomes
for every squad member, including the bench. DGW outcomes are summed, repeated origin
copies must agree, and the shared corrected scorer handles autosubs, captain/vice,
BB, TC and hits. Both normalized `TC` and preserved `TC_name` labels award the same
triple bonus. An optional reference requires exact per-round corrected gross/net
reconciliation. Reports verify captured bytes; failed attempts stay preserved.

An absent player row is **unknown**, not proof of a blank. Strict replay rejects it.
The shared scorer's compatibility mode retains zero filling but labels such rows
`LEGACY_ZERO_FILL` and reports their missing-player count. This does not retrospectively
qualify old CSVs. Whole-team blank fixtures and player-club identity need explicit
evidence before the corresponding zero outcomes can be admitted.

This is settlement evidence, not full financial/rule legality: bank, transfer prices,
FT/chip inventories, native solver verification and outcome-blind execution remain
separate assessments. A legacy BB snapshot with 15 starters cannot supply a unique
legal XI and bench order; replay rejects it rather than inventing that structure.
The registered four-case pilot in `experiments/decision_replay_pilot_2026-10-17/`
preserves all four NOT QUALIFIED outcomes at GW143. A separate follow-up under
`experiments/blank_outcome_replay_2026-10-17/` verifies settlement with documented
blank evidence while retaining those failed attempts and original forecast bytes.

For an absent selected player, optionally supply `--blank-evidence evidence.json`.
The descriptor's folder must contain `fixtures.csv`, `teams.csv`,
`bootstrap-static.json.gz` and the historical capture `manifest.json`. Schema 1
records `season`, `first_global_gw`, `event`, a 40-character
`fixture_source_commit`, and `input_sha256` for all four files. Supply source URLs
and registration/calendar limitations alongside those required fields. Hashes bind
caller-reviewed source claims; they do not authenticate an archive publisher.

Admission requires a complete, finished 20-club/380-fixture ledger with every
ordered home/away pair once, scores, consistent season/deadline dates and explicit
registered stable player identities. A missing player at a club with a fixture is
still rejected. The capture manifest must bind the decompressed official register
payload and document a matching predeadline capture within the configured 24-hour
limit. Added zeros go to a separate `explicit-blank-outcomes.csv`, without fixture
IDs or forecasts; `predictions.csv` remains byte-identical to its original input.
The scorer validates their zero arithmetic and records `explicit_blank_players`.

This is a **retrospective actual-fixture** check with **capture-time registration**.
It does not prove a deadline-known forecast calendar, exact-deadline identity,
financial/rule legality or outcome-blind execution. The four follow-up cases passed
settlement/reference reconciliation with unchanged totals, but remain unranked
and ineligible for promotion.

### Explicit decision state

New legacy snapshots and Solio rows retain `state_mode` (`fresh_build` or
`continuing`), `permanent_squad_start/end` and `permanent_bank_start/end`.
Banks use tenths of a million pounds; existing `q_start/end` retain the backend's
free-transfer contract. Permanent holdings use original stable IDs, serialized as
JSON lists even in the legacy human CSV.

During a Free Hit, `squad` is the temporary selected squad, while permanent
start/end holdings and bank remain unchanged. `temporary_selection_bank_end`
records available wealth minus that temporary squad's cost, and is null otherwise.
It is not money carried into the next round. Legacy `budget_end` still describes
permanent bank in a continuing Free Hit. Modern and legacy price/FT/chip policies
remain distinct; these extra fields do not harmonize their rules.

A first-round Free Hit requires an existing permanent squad. Historical snapshots
without these fields retain their evidence gaps; do not invent first-row holdings.
Capturing state is distinct from independently validating its legality.

### Independent financial arithmetic

`fpl.milp.financial_audit.audit_legacy_finances` checks fixed legacy **no-chip**
choices without calling the optimizer or scorer. Supply explicit initial state,
FT cap/gain and unique current player-round prices in integer tenths. It checks
15-player holdings, exact transfer deltas, affordability, bank conservation, paid
hits and FT carry. Captured optional permanent-state fields must agree.
Unsupported chips are rejected rather than certified.

The default `current_price` policy checks reported accounting. A separate
`half_profit` policy tracks explicit acquisition prices and resets basis on rebuy;
continuing squads require every original purchase price. With
`compare_reported=False`, it reports a fixed-decision selling-policy sensitivity,
including bank differences, and still rejects unaffordable choices. It does not
optimize a new strategy or change point totals.

The registered pilot audit under `experiments/financial_replay_2026-10-17/`
reconstructs initial state and price assumptions from byte-matched evaluated source.
Those are source-derived facts, not retrospectively explicit plan registration.
The price evidence labels forward/back fill, missing-value defaults and disagreeing
copies. All four recorded arithmetic paths pass; the half-profit sensitivity leaves
all choices affordable, with £0.1m less closing bank for standard control only.

This component does not validate lineups/club limits, chip rules, official price
publication times, full transfer-rule chronology or outcome-blind execution. Its
PASS is **financial arithmetic under the declared evidence**, not full site legality.
Settlement structure checks and price/information provenance remain separate gates.

### Recovered historical scorecards

The preserved October 3 seed-0 study can be read through the current scorecards
without rerunning its models. The recovery script under
`experiments/recovered_ablations_2026-10-17/` checks archived evaluated sources,
registered settings, prediction/result hashes and the earlier selection lock.
It writes separate normalized accounting exports because that historical format
stored net points in `actual_total_points`, with gross points in a separate column.
Original bytes remain intact; this is accounting/report recovery, not a new scoring
or decision replay. Supply the preserved original bundle and a new output directory:

```sh
PYTHONPATH=. python experiments/recovered_ablations_2026-10-17/recover_scorecards.py \
  /path/to/original-bundle /path/to/new-reports
```

The nine recovered entries remain unranked. Their tuned MAE/RMSE contrast changes
capacity as well as loss, and their historical optimizer exports lack native proof
and stable-ID decisions. A prior `Optimal` log label does not fill those gaps.

The shared scorecard is deliberately conservative: saved squad CSVs alone do not
establish original execution telemetry. Optional forecast panels and runner receipts
fill the sections they measure, without interpreting missing evidence as a pass.
Research definitions can live under
`experiments/specs/`; the results registry and research log remain the evidence index.
GitHub issues remain the work tracker, with no separate roadmap/status file here.

The legacy decision snapshot records **settlement_completed=false**, not a guarantee
that outcomes were inaccessible. The historical optimizer loads a CSV containing actuals;
a genuine outcome-blind interface and independent replay audit require further work.
The UI and these reports must not manufacture that stronger claim.
