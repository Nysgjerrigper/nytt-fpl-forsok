# Live forecast tracking

`python -m fpl.run_week` now preserves each forecast in
`fpl/live_forecasts/<timestamp>-<unique id>/` before updating `predictions_latest.csv`.
Each local capture contains forecast bytes and a JSON receipt with season, capture
time, target deadlines, strategy and SHA-256. These ignored files must be retained
and backed up; Git does not preserve them. Captures are made before optimization,
so a failed optimizer does not erase a valid forecast.

After refreshing the mutable actuals dataset with `python -m fpl.data.fetch`, run:

```sh
python -m fpl.live_evaluation > /path/to/new-report.json
```

The read-only report checks forecast bytes and season, admits only official
`data_checked` rounds, sums double-gameweek fixtures into player rounds, and
reports MAE, RMSE, bias, aggregate calibration, within-position Spearman and top1
capture by round/position. Every capture and lead is separate. Missing outcomes
are reported as unknown with player IDs; they are never silently scored as zero.
A partial report describes only its explicit-outcome subset. Unchecked rounds
remain pending. A changed API season requires a matching archived bootstrap;
the Python `evaluate` interface accepts it explicitly.

A recovered older file must use `capture(..., recovered=True)` and an explicitly
identified strategy (use `unknown` if unverified). Capture time and filesystem
mtime do not prove original pre-deadline creation. Never refit a model now and
label its output an original live forecast. The receipt hash protects forecast
bytes from accidental changes; it is not independent certification of metadata.

This is forecast monitoring, not squad settlement: actual selections, autosubs,
captain/vice and transfer hits need the separate decision replay contract.
These outcomes do not admit GW232+ into frozen research, establish training
cutoffs, or justify promotion. Production remains `single:catboost`.

On 2026-10-31, the primary checkout retained GW8/GW9 squad recommendations but
only GW9 full forecasts; its mutable outcomes ended at GW8. Consequently no full
GW8 forecast-error result can be claimed from those files. Recovery and any later
numeric results belong in issue #152 with artifact hashes, not reconstructed
forecast claims. Evaluation is an explicit command; it is not a scheduled job.
