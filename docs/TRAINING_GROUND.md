# The Training Ground

The Training Ground is our offline playground for checking ideas before they reach
**Matchday**, the configured weekly system. Its first version compares saved squad
backtests, generates scorecards and maintains separate descriptive league tables.
It does not yet train models, orchestrate ablations, launch the optimizer or promote
anything to production. Those connections belong to the existing research-runner work.

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
  comparison.json
  scorecard.json
  REPORT.md
  complete.json OR failed.json
```

Generated runs are ignored by Git. The completion receipt hashes the required results,
input snapshots and specification. Source and reporting-runtime identity are recorded.
Original model-fit time, failure rate, forecast diagnostics, calibrated uncertainty,
independent legality and native solver proof remain explicitly unmeasured/unassessed.

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

## Where the next connections belong

The shared scorecard is deliberately conservative: this adapter has squad CSVs, not
validated fixture-level prediction panels or original execution telemetry. Future
adapters should populate those sections using existing model metrics and OOF validation,
without interpreting missing evidence as a pass. Research definitions can live under
`experiments/specs/`; the results registry and research log remain the evidence index.
GitHub issues remain the work tracker, with no separate roadmap/status file here.

The legacy decision snapshot records **settlement_completed=false**, not a guarantee
that outcomes were inaccessible. The historical optimizer loads a CSV containing actuals;
a genuine outcome-blind interface and independent replay audit require further work.
The UI and these reports must not manufacture that stronger claim.
