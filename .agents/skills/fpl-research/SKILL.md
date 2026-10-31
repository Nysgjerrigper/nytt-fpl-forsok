---
name: fpl-research
description: Design, register, run and skeptically review bounded FPL forecasting experiments through Training Ground. Use for natural-language research questions and pilots, not live team advice or production promotion.
---

# FPL research

Read `CLAUDE.md` first. Use [Training Ground](../../../docs/TRAINING_GROUND.md)
for the workflow CLI, architecture and external-skill comparison.

## Design before results

Inspect the relevant GitHub issue, `RESEARCH_LOG.md`, `experiments/results.csv` and
specifically relevant artifacts. Identify completed/failed contrasts before compute.
Do not reopen gated/parked studies through this skill. Turn the question into a
falsifiable hypothesis with one intervention. Record control/candidate, ordered
features, resolved parameters, seed, source/runtime/data hashes, validation rounds,
compute and success criteria before execution. A fixed research control is not the
tuned production model.

Use `fpl.research_runner prepare` for supported loss contrasts or same-model feature
ablations. Unsupported expected-minutes models need a separate minimal implementation
and tests; never silently substitute a different question. If a compute budget is
unspecified, register a safe dry-run first. No paid API, external data, CUDA assumption
or automatic production changes.

Use only selection diagnostics for iteration. Derive windows with `protocol_cutoffs()`;
never select on historical final evaluation, spent rounds or prospective rounds.
The helper rejects non-selection plans. Frozen admission may read all source bytes,
but features/results must end at the selection cutoff. Do not inspect later outcomes
or use final-evaluation reports to choose variants. A fresh final holdout requires a
separately authorized outcome-blind evaluation boundary; this workflow does not
sandbox arbitrary file reads or make observed historical rounds fresh again.

## Register and execute

Prepare a small ordered list of plans sharing control, window, seed and provenance.
Write a JSON specification with nonempty strings `question`, `hypothesis`,
`success_criteria`, `stopping_rule`, `limitations`; positive `max_worker_seconds`;
and `plans` (plan paths relative to the specification). See the maintained guide.

Register with `python -m fpl.research_workflow register SPEC --output NEW_DIRECTORY`.
Pin the registration hash in the issue before running. Registration captures bytes
without fitting. Plan count limits iterations; summed worker caps must fit the total
budget. These reserve compute conservatively, excluding planning and agent tokens.
Failures/incomplete launches consume their slot and stop the series.

Execute one step with `python -m fpl.research_workflow next DIRECTORY --dataset
FROZEN_CSV --reason "Why this contrast is justified now"` (one shell line).
Use the shared primary `.venv` explicitly with `PYTHONPATH` set to this worktree.
Read `python -m fpl.research_workflow report DIRECTORY` after each step; review
evidence before choosing another. Stop on the registered condition, provenance
failure, budget exhaustion or lack of an evidence-based next contrast.
Preserve negatives. Do not edit registered plans, delete failures/locks or reset
budgets to retry. Adaptive plans require a new registration and an explicit remaining
budget accounting for earlier attempts in the issue; do not endlessly tune selection.

## Engineering and agent costs

The primary session owns design, modeling and final judgment. Delegate precise
implementation to the existing `implementer` only when useful; use `searcher` for
bounded lookup. Specify owned files, prerequisites and checks; subagents must not
spawn agents. Reuse compact receipts; never put datasets or entire logs in context.

When already available, use Matt Pocock's `to-spec` for issues, `tdd` for behavioral
regressions, `diagnosing-bugs` for a failing loop and `code-review` for the diff.
These optional skills are not installed by this workflow. Use repository practices
when absent. Do not install a suite, invent aliases or switch paid model tiers.
Use a supported economical model for mechanical tasks only when configured; role
names alone do not guarantee lower cost.

## Review and report

For material experiments request one separate reviewer with the registered hypothesis,
plan, verified compact metrics and relevant artifact/code paths. Ask independently
about leakage (origin form, prices and fixture admission), reproducibility, coverage,
selection bias, uncertainty, period generalization and forecast versus FPL quality.
Do not supply a preferred conclusion. If unavailable, label primary-session critique
explicitly; never claim independence.

Use `read_result`, scorecards and `compare_backtests` as numerical truth. Report
MAE/RMSE, bias, calibration, Spearman/top1 capture by protocol/lead, corrected MILP
net points and paired CIs. MASE/unmeasured metrics stay unavailable. Standard
lookahead is diagnostic; origin evidence retains documented limitations. Bootstrap
intervals are not null p-values. A short seed-0 pilot cannot prove robust improvement.

Report directly in Codex: hypothesis; configuration; candidate versus control;
uncertainty/limitations; skeptical review; keep/reject/investigate further; next
justified experiment. Append dated evidence to `RESEARCH_LOG.md`; log through
`fpl.experiment` (runner already emits `registry.csv`); link artifacts/hashes in the
issue. Keep pilot evidence in a separate PR. No automatic production promotion.
