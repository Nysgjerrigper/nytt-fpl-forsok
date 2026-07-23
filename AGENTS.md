# AGENTS.md

This file provides guidance to coding agents other than Claude Code when working in this
repository.

**Read `CLAUDE.md` — it is the single canonical operating document for ALL agents.**
Everything in it applies here unchanged: what the repo is, setup and commands, the
gameweek-numbering convention, architecture, the backtesting reference point and research
protocol, git workflow, and code-quality standards. This file used to hold a hand-maintained
copy of that content; the copy drifted (it still quoted retired baseline numbers), so per the
project's own hygiene rules the duplicate was removed (2026-07-23) rather than maintained in
parallel.

Two rules worth restating because agents violate them most often:

1. **English only, in both output directions** (PO decision, 2026-07-11). The user may write
   in Norwegian; you always answer in English, and everything landing in the repo (docs,
   comments, commit messages, log entries) is in English.
2. **Never judge a modeling change on MAE/MASE alone.** The decision metric is realized
   points from the MILP backtest vs the standing baseline in `CLAUDE.md`, with a
   `python -m fpl.milp.compare_backtests` confidence interval — see "Research &
   experimentation protocol" in `CLAUDE.md`.
