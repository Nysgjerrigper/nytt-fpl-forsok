# FPL forecasting and squad optimization

A Python pipeline for player-point forecasts and rolling-horizon FPL squad decisions.
It replaces the thesis-era R/LSTM system; the old validation predictions remain a fixed
benchmark. Production uses `single:catboost` for GK, DEF, MID and FWD.

## Start here

- [Setup and operating rules](CLAUDE.md): canonical instructions for every agent.
- [GitHub issues](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues): priorities, progress, blockers and remaining work.
- [Stakeholder overview](docs/STAKEHOLDER_OVERVIEW.md): purpose, goals and tools in plain English.
- [Alternative optimizer review](docs/OPTIMIZER_REVIEW.md): current-season requirements and candidate gaps.
- [Open FPL adapter](docs/OPEN_FPL_ADAPTER.md): optional modern-rule backend, inputs and measured comparison.
- [Research history](RESEARCH_LOG.md) and [run registry](experiments/results.csv): dated evidence,
  including negative results; historical numbers are not automatically current controls.
- [Historical documents](docs/archive/README.md): superseded handoffs, the July audit,
  old reports and the initial recovery run plan.

## Setup and use

Python 3.11+; LightGBM needs OpenMP (`brew install libomp` on macOS).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest tests/ -q
```

`requirements-runtime.txt` pins pipeline dependencies; `requirements.txt` adds development
and notebook tools. `requirements-research.txt` is optional and does not enable a model
that fails its runtime guard.

```bash
python -m fpl.run_week --team-id <team-id> --horizon 3
```

Without a team ID the driver builds a fresh squad. Live recommendations refresh historical
data, use official team/fixture/availability inputs and refit models; they require a deadline-current team/fixture/availability check.
Historical backtests and research use the separate immutable input at
`Datasett/frozen/master_dataset_gw231.csv`, validated against the tracked SHA manifest.
A live refresh must never overwrite this frozen file. It is deliberately ignored and
must be restored from the preserved snapshot, not reconstructed by fetching current data.

```bash
python -m fpl.model.predict --start-gw 153 --end-gw 183 --output <predictions.csv>
python -m fpl.milp.optimize --predictions-csv <predictions.csv> \
  --start-gw 153 --max-gw 183 --horizon 3 --scoring-mode corrected --output <squads.csv>
python -m fpl.milp.compare_backtests <candidate-squads.csv> <matching-control-squads.csv>
```

Those GW numbers apply to the current 2020-21 start season. Derive them again if that
start changes. Tuned production files are named `tuned_params_<POSITION>_catboost.json`;
research `catboost_mae` files are a different namespace. An absent production artifact
means registry defaults, not a reproduced tuned baseline.

An optional direct-HiGHS Open FPL Solver adapter is available as `python -m fpl.milp.solio`
and through `run_week --optimizer solio`. It preserves our forecasts and corrected scorer,
checks current FT/chip rules, and requires proven optimal, independently legal decisions.
The initial matched static-price comparison was about 1.6 times slower, with inconclusive
point differences. Legacy remains the default. See the adapter guide before supplying
owner state or interpreting these results as live advice.

## Layout

- `fpl/`: config, ingestion, causal features, model registry/research tools and MILP.
- `tests/`: correctness, causality, provenance, scoring and runtime guards.
- `Datasett/`: generated live/frozen inputs and explicitly labelled thesis-era raw data.
- `legacy/baseline_outputs/`: fixed LSTM benchmark, still used by training diagnostics.
- `experiments/`: dated evidence and run registry; local recovery payloads are ignored.
- `docs/archive/`: historical documents; their instructions and results are not active policy.

The MoE interface is research-only. Available stable-expert maps can be requested explicitly
with `--expert-map GK=catboost,DEF=lightgbm,MID=xgboost,FWD=catboost`. Registration or an
exploratory screen does not establish promotion. The known PyTabKit 1.7.3/macOS-arm64 crash
combination is blocked before native fitting; no neural rerun is proposed without a proven fix.

## Recovery research interfaces

Tournament OOF fits fixture rows and compares summed player-gameweek predictions;
training-only MASE uses summed training rounds. Expert panels must cover identical
players/weeks with consistent provenance. Existing artifacts are preserved, and new
output records its player-round schema. Missing minutes history retains the established
MID low-regime policy; missing targets or predictions are rejected.

Optuna persistence is opt-in with `--checkpoint <sqlite-path>` and optional
`--study-name`. The separate `serial_trial_seed_v1` protocol binds capped data, ordered
features, exact folds, source hashes and runtime. Clean resumes reproduce the same
trial sequence for identical objective outcomes. Orphan RUNNING/WAITING trials and
identity mismatches are preserved and refused. The total trial target may be extended;
the timeout is per invocation. Incomplete studies cannot emit tuned artifacts.
Checkpoint exports go under `fpl/models/.optuna_exports/` and are rejected by legacy
tournament validation. This interface does not restart or replace registered studies.
