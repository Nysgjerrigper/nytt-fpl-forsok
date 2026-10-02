# Canonical operating instructions

This is the single operating document for all agents in this repository. `AGENTS.md`
points here. The agent owns engineering and research rigor; the user is the Product
Owner and controls direction, landing changes, and irreversible actions.

## Scope and navigation

The active system is `fpl/`: ingestion -> causal features -> per-position forecasts ->
rolling-horizon MILP. Production remains `config.PRODUCTION_WEIGHT_STRATEGY =
"single:catboost"`. Research members and probabilistic forecasts are separate opt-in
interfaces, not evidence of production promotion.

Use `README.md` for setup/navigation and [GitHub issues](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues)
for all open work, blockers, priorities and progress. Keep dated research evidence in
`RESEARCH_LOG.md`/`experiments/results.csv`. Do not create or maintain a parallel progress,
TODO or handoff Markdown tracker. Documents under `docs/archive/` are historical snapshots, not current instructions. Keep historical
results and their provenance; do not erase negative findings as cleanup.

## Communication and autonomy

- English only in responses, code, documents, comments, logs and commit messages.
- Lead with the result/recommendation, explain the evidence and practical limit plainly.
- Own mechanical implementation, traceback fixes and verification. Escalate design
  changes, production promotion, paid work, result-discarding or irreversible actions.
- Do not expand scope silently. Keep the PO informed of significant findings and changes.
- End a turn with what was done, how it compares with prior state, its consequence,
  and a concrete next step. Do not present synthetic tests or partial runs as results.

## Git and worktrees

- Non-trivial features, bugfixes and experiments require an isolated worktree/branch
  off current `main`, using `feature/`, `fix/` or `exp/`. Reuse an unmerged worktree only
  for its continuing task. After merging, retire it and start anew from updated `main`.
- **Commit, merge and push only when the PO authorizes them.** There is no exception for
  frequent local commits. Never force-push `main` without explicit approval.
- Run the appropriate focused checks and full `python -m pytest tests/ -q` before
  proposing a commit. Fix failures first; inspect the actual diff in the primary session.
- Share the primary checkout's `.venv` explicitly. Paths rooted at `config.ROOT` resolve
  per worktree, including ignored datasets and tracked tuned JSONs.
- Before retiring a dirty worktree, preserve and verify unique files and ignored artifacts.
  Inspect runner state; do not clear locks or transfer execution authority when that state
  cannot be classified. Git ancestry alone cannot prove dirty files are obsolete.

## Setup and entrypoints

Python 3.11+. Install pinned dependencies from `requirements.txt`, then install the project
with `python -m pip install --no-deps --no-build-isolation -e .`. LightGBM requires OpenMP
(`brew install libomp` on macOS). CI installs the package and runs the regression suite.

```bash
python -m fpl.data.fetch                              # refresh mutable live input
python -m fpl.run_week --team-id <id> --horizon 3       # live refresh/refit/decisions
python -m fpl.model.train                             # historical model diagnostics
python -m fpl.model.predict --start-gw N --end-gw M    # historical forecasts
python -m fpl.milp.optimize --start-gw N --max-gw M --horizon H
python -m fpl.model.tuning                            # historical capped tuning
python -m pytest tests/ -q
```

Training evaluates/refits; it does not save a fitted ensemble for later reuse. Every live
and historical consumer refits through `train.fit_position_ensembles`. The registry keeps
negative candidates for research history. Tracked tuned parameter JSONs preserve expensive
completed studies; fitted binaries, datasets and local caches remain ignored. Do not regenerate
or activate tuned artifacts merely to tidy the tree.

## Data and gameweek contracts

`GW_global` is season-ordinal, not a calendar-fixed constant. With
`config.DEFAULT_START_SEASON="2020-21"` and 38 rounds per season, 2024-25 is GW153-190,
2025-26 GW191-228 and 2026-27 starts at GW229. Derive windows with `config.season_window`
and tournament runtime cutoff helpers if the start season changes.

Historical ingestion uses Vaastav; active-season histories use official FPL element
summaries and admit exactly `data_checked` events. GW1's API deadline identifies the
active season. API failures abort without stale archive fallback; archive-only builds
use `include_live=False`. Mutable saves are atomic and cannot target the frozen path.
Discovery is dynamic, never a
hardcoded season/GW list. Player identity is the stable FPL code; prices and availability are live
inputs. New players without PL history remain a documented blind spot. Owned-player sell
prices use the accepted buy-price simplification; do not imply exact sell-price accounting.

Separate input paths:

- `MASTER_DATASET_PATH`: mutable live refresh output used by `run_week`.
- `FROZEN_RESEARCH_DATASET_PATH`: preserved GW231 snapshot, admitted by byte SHA, row,
  season and cutoff checks. Historical point forecasting, tuning, train diagnostics,
  MoE selection and screening use this input. No GW232+ research admission.
- A cached feature frame is local/regenerable and bound to dataset bytes, raw-frame
  fingerprint, feature source, cache schema and pandas version. Mutated frames cannot
  reuse an admission marker. Never treat a cache filename as provenance on its own.

Every player-derived rolling/expanding/EWMA feature is strictly shifted; double-gameweek
fixtures share the first fixture's deadline-known form. Opponent form is shifted by round.
Fixture data is known ahead and legitimately per fixture. Entire-season absent statistics
stay NaN; missing collection is not zero. No same-round target enters another fixture's
features. Trees consume NaNs; other estimators use imputation and scaling where appropriate.

## Research and promotion

Never judge a modeling change on MAE/MASE alone. These favor conditional medians; squad
optimization needs calibrated means, rankings and captaincy upside. Report RMSE, bias,
aggregate calibration, Spearman and top1_capture alongside accuracy. The decision metric
is **realized MILP points with a paired `fpl.milp.compare_backtests` confidence interval**.

For a material forecast/feature/optimizer change:

1. Register the hypothesis, exact candidate/control settings, seeds, windows and data/code
   provenance before evaluation. Preserve raw output bytes and hashes.
2. Diagnose forecasting behavior, run standard walk-forward forecasts into horizon-3 MILP
   on 2024-25 GW1-31 (currently GW153-183), and compare identical-input/scoring controls.
3. Use origin-based forecasts for deployment-honest evidence; freeze form at each origin.
   Standard forecasts have future-round lookahead and do not establish live performance.
4. Log every experiment through `fpl.experiment` and `RESEARCH_LOG.md`, negatives included.
   State incomplete/no-result explicitly; registry membership and methodology PASS are not results.

Current registered MoE chronology: discovery <=GW136, selection GW137-152, final evaluation
GW153-183, dynamically derived. GW191-221 is spent and may never be used for selection again.
GW232+ remains prospective. `run_week` refresh/refit is not a prospective-holdout experiment.

A complete MoE position map and optional MID gate are the only registered promotion family.
The MID gate uses only deadline-known `mins60_rate_roll5` with a training-only MASE scale.
Tuning artifacts, OOF rows and finalist/control predictions must be hash-bound and validated.
OOF comparisons sum fixture forecasts/targets into player rounds, with a training-round
MASE denominator and identical expert coverage. The opt-in serial checkpoint sampler is
a distinct protocol; its exports are not admissible legacy registered-study evidence.
Seeds 0/1/2 are required. Every seed must improve standard/origin totals; standard CI lower
bound must be positive, origin CI lower bound no worse than -40. Holm-adjusted one-sided
paired sign tests must pass at alpha 0.05, and the simpler passing architecture wins.
Exact paired sign tests drop ties; block bootstrap supplies CIs, **never null p-values**.
The known native neural crash must fail clearly without impairing production. Re-probe only
in a separately validated runtime; repeated known-failing native fits are not useful evidence.

## Optimizer and comparison contracts

The Kristiansen MILP solves budget, formation, captain/vice, transfers and optional chips
with PuLP/HiGHS; CBC remains a fallback. Preserve `MILP_GAP_REL=0` and check every research
solve is proven optimal. Do not sacrifice points or validity to a faster solver gap.
The free-transfer planning cap is an intentional policy of 2; live initial FT stacks are honored.
The legacy chip interface and FT reset policy do not establish full modern-season rule
support. Current rules allow five banked transfers and two chip sets. Evaluate migration
separately from historical controls; see [optimizer review](docs/OPTIMIZER_REVIEW.md).
The optional `fpl.milp.solio` backend vendors the pinned offline Open FPL solve function
with reviewable compatibility patches. It enforces modern inventories, FT preservation,
FH state/prices, legal BB lineups and exact optimal status; our independent output
validator and corrected scorer remain authoritative. Forecast admission rejects mixed
origins, duplicate/ambiguous fixtures and wholly missing requested rounds. Forecast
exports retain fixture/opponent/home identity without changing fitted models.
Initial static-price standard/origin comparisons did not establish faster execution or
better points. Keep legacy as the default until a further registered comparison supports
switching. Preserve upstream bytes, license wording, patch and hash provenance when
refreshing the vendor; do not relabel changed upstream source as the pinned revision.

Chip CLI `0` means disabled, not unspecified. All standing comparison backtests disable chips.
Positive chip targets force execution and are validated before solving. WC/FH exempt
transfer-hit bounds; FH preserves permanent holdings/bank and uses a temporary squad.
Fresh-build and continuing-squad modes have different budget/state inputs; preserve their contracts.
Corrected realized scoring uses whole-round minutes, legal bench order, vice activation and chip
multipliers; CSVs without minutes are explicitly labelled legacy. `--scoring-mode corrected`
rejects unknown/invalid minutes. Never compare scores across different scoring protocols as a
forecast improvement. Exact matching controls and historical scoring lineage are preserved in
`RESEARCH_LOG.md`, the run registry and the [archived 2026-10-02 handoff](docs/archive/2026-10-02-issue-migration/HANDOFF.md).

## Issue workflow

- Before substantial work, find or create the relevant GitHub issue; avoid duplicates.
- Describe the concrete problem, acceptance criteria, dependencies and evidence links.
- Record meaningful progress, blockers and verification in that issue, not a status file.
- Mark parked/blocked work explicitly; rejected experiments stay historical evidence.
- Reference the issue in the PR. Close it only when its acceptance criteria are met;
  use `Fixes #N` only for fully completed work. Partial PRs use `Refs #N`.
- Keep setup, architecture, user guides and canonical rules in maintained docs. Issues
  track work; research logs and artifact hashes retain reproducible results.

## Engineering quality

Public pipeline functions need type hints and docstrings stating data dependencies/math.
Vectorize ETL/features; no `iterrows`/`itertuples` there unless unavoidable and justified.
Put paths, parameters and thresholds in `fpl/config.py`. Use logging for diagnostics and
human-facing print output for CLI reports. Add meaningful focused regression tests for new
behavior, then run the full suite. Correctness tests do not prove a model or optimizer is better.

## Delegation

The primary session owns design, modeling, MILP formulation and research judgments.
Delegate read-only lookup to `searcher`, and implementation from a precise decided spec to
`implementer`. Subagents do not spawn other subagents. Verify load-bearing audit claims,
inspect implementer diffs, and rerun tests in the primary session.

`.claude/agents/` uses Claude-specific Haiku/Sonnet settings. Other runtimes use their
available role/model settings; those Claude names are not cross-runtime model identifiers.
Codex role definitions live in `.codex/agents/` and point back to this document. Do not
silently invent a model mapping or spend on a higher-tier model contrary to the user's choice.
