# Why this FPL project exists

## The purpose

Fantasy Premier League is a practical setting for studying decisions under uncertainty.
Each week, a manager must choose players, transfers and a captain with a limited budget,
incomplete information and consequences that extend into future weeks. This project
turns that problem into a reproducible workflow: collect data, estimate likely player
points, and find a feasible squad and transfer plan.

The broader goal is to develop and demonstrate analytical skills that connect economics,
forecasting and software engineering. FPL provides observable outcomes against which
decisions can be tested. The project cannot guarantee wins or predict individual matches
with certainty.

## What we want to achieve

- Produce useful weekly recommendations that respect squad, budget and transfer rules.
- Measure whether changes improve the points earned by selected squads, rather than
  assuming that a better prediction statistic produces better decisions.
- Keep experiments reproducible, with clear records of inputs, settings and outcomes.
- Test future performance honestly: results that become available later must not enter
  earlier training or influence which model is selected.
- Make the repository understandable and maintainable, with one current set of
  instructions and clearly labelled historical documents.

## How it works

1. **Collect information.** Historical player and fixture data come from the public
   Vaastav FPL archive. Official FPL API data provide active-season histories, fixtures,
   player registration, prices and availability. Only finalized, checked rounds enter
   the recovered active-season history pipeline.
2. **Estimate player points.** Python prepares recent form and fixture information.
   CatBoost currently supplies separate forecasts for goalkeepers, defenders,
   midfielders and forwards. Alternative models are research candidates.
3. **Choose a plan.** A mathematical optimizer compares legal squads, lineups,
   captains and transfers over several weeks. It weighs expected points against budget
   limits and transfer costs. A forecast is an estimate; an optimal mathematical
   solution is optimal only for the inputs and rules supplied.
4. **Evaluate outcomes.** Historical backtests score the selected squads using realized
   player points. Paired comparisons and confidence intervals show whether an apparent
   gain is convincing or could reflect a particular run of gameweeks.

## Tools and their roles

| Tool | Purpose |
| --- | --- |
| Python, pandas and NumPy | Collect, clean and transform data. |
| CatBoost and other model libraries | Estimate player points and test alternatives. |
| PuLP and HiGHS | Express and solve squad planning; an optional Open FPL integration uses HiGHS directly. |
| Optuna | Explore model settings under a defined historical cutoff. |
| pytest | Check budgets, identities, scoring, data safeguards and other behavior. |
| Git and GitHub | Track changes, review pull requests and preserve project history. |
| Codex coding agents | Assist with implementation and audits; conclusions still require
  checked code, tests and recorded evidence. |

## Why the current cleanup matters

Several branches and documents represented different stages of the project. Some
instructions and reported baselines had become outdated. The recovery consolidates
current guidance, preserves historical evidence, strengthens data and scoring checks,
and makes active work easier to distinguish from completed or rejected experiments.
The same-input engineering comparisons retain the existing corrected controls of
2,088 standard points and 1,901 origin points over the historical 31-round window.
These are historical controls, not promises about future returns.

## Limits and next steps

New players without Premier League history remain difficult to forecast. Availability,
prices and fixtures can change close to a deadline. The existing optimizer also retains
a legacy transfer-banking policy and chip interface; the recovery does not establish
complete support for the modern season rules.

An optional Open FPL Solver integration now uses our forecasts and passes independent
rule checks. Its first historical comparison was slower, and the point differences were
too uncertain to establish an improvement. The existing optimizer remains the default.
The next step is to understand that runtime cost and test realistic team/price/chip state
before switching. [The adapter guide](OPEN_FPL_ADAPTER.md) documents these results.
Model improvements likewise require realized-point evidence; successful software tests
alone do not justify promotion.

Progress and remaining work are tracked in [GitHub issues](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues).
Verified research results remain in the research log and experiment registry.
This document explains the project to stakeholders and does not replace its operating instructions.
