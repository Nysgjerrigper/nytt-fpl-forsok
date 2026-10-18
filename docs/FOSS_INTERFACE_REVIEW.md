# FOSS interface reuse review

Research deliverable for [issue #15](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/15).
Reviewed 2026-10-18. This note compares free and open-source interfaces that could
support inspecting forecasts, squads, transfers and optimizer results. It recommends
reuse patterns for the future dashboard in [issue #20](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/20);
it does not select a product journey, deploy a service or change the Python pipeline.

## Recommendation

Use **[PITCHSIDE](https://github.com/bjarkiSigur7/fpl-ai-assistant/tree/42ef49820f12432ee6d3d43e89ca208dc4fa7e18)**
as the primary presentation reference and candidate for a small MIT-licensed adaptation.
Its static-bundle boundary, modern Next.js/TypeScript UI and pitch/verdict components
fit this project better than the alternatives. Reuse selected presentation components
only after the first journey is agreed in #20; keep all forecasting, legal checks,
pricing and optimizer logic in Python.

Do **not** fork the whole application or import its model, datasets or business logic.
The bounded first slice should be a read-only screen fed by one preserved run, with
explicit readiness and missing-evidence states.

The remaining candidates are useful references, but each has a material licensing,
age or coupling limitation. No existing project should be adopted wholesale.

## Method and evidence boundary

- Retrieved repository metadata through the GitHub API and pinned each reviewed
  default-branch commit on 2026-10-18.
- Made shallow read-only clones under `/tmp` and inspected the README, license files,
  package manifests, UI source trees and tests. No candidate application was executed.
- Checked public demos or repository screenshots where available. Stars and commit
  activity are discovery signals, not evidence that a forecast or optimizer is correct.
- Compared feature presence only. No candidate's predictive accuracy or FPL-point gain
  was evaluated or is claimed here.

## Ranked shortlist

The ranking is for **reuse fit in this repository**, not overall product quality.

| Rank | Project | Pinned revision | License evidence | Maintenance signal | Interface and demo | Reuse assessment |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | [PITCHSIDE](https://github.com/bjarkiSigur7/fpl-ai-assistant/tree/42ef49820f12432ee6d3d43e89ca208dc4fa7e18) | `42ef498` | MIT in [`LICENSE`](https://github.com/bjarkiSigur7/fpl-ai-assistant/blob/42ef49820f12432ee6d3d43e89ca208dc4fa7e18/LICENSE) | Default-branch commit 2026-09-12; repository push 2026-10-17 | [Live site](https://bjarkisigur7.github.io/fpl-ai-assistant), [dashboard screenshot](https://github.com/bjarkiSigur7/fpl-ai-assistant/blob/42ef49820f12432ee6d3d43e89ca208dc4fa7e18/docs/screenshots/dashboard.png) | **Preferred.** Permissive license, static JSON bundle, pitch and decision-card components, multi-gameweek planner, local tests. |
| 2 | [FPL Forecast](https://github.com/daniel-mehta/fpl-forecast/tree/b59c56ef7f0f7ef768d29ea07da972b155100521) | `b59c56e` | AGPL-3.0-only in [`LICENSE`](https://github.com/daniel-mehta/fpl-forecast/blob/b59c56ef7f0f7ef768d29ea07da972b155100521/LICENSE) | Default-branch commit 2026-09-27; repository push 2026-10-17 | [Public dashboard](https://danielmehta.com/fpl-forecast/) | Strong static React architecture and data-contract discipline, but AGPL network/source obligations make direct code reuse a licensing decision, not a routine copy. Use as a reference only. |
| 3 | [Open FPL](https://github.com/bapairaew/open-fpl/tree/f0672eb153efdf3d625d44e9ce7f64bf475584fb) | `f0672eb` | Apache-2.0 declared in all [`package.json`](https://github.com/bapairaew/open-fpl/blob/f0672eb153efdf3d625d44e9ce7f64bf475584fb/package.json) manifests; no top-level `LICENSE` or `NOTICE` in the reviewed tree | Default-branch commit 2022-02-15 | [Live site](https://www.openfpl.com/) | Feature-rich planner and pitch UI, but old Next.js 11/React 17 stack and missing distributed license text. Obtain the notice file and assess upgrade cost before reusing code. |
| 4 | [FPL Helper](https://github.com/ekkyvalent/fpl-helper/tree/ce71079cd3b5ca8174c2b6d5273d7235859e9d55) | `ce71079` | README says MIT, but no `LICENSE` file is present in the reviewed tree | Default-branch commit 2026-08-21 | No public demo identified; README describes a local Next.js app | Simple, modern pitch/transfers/captain/chip UI. Useful interaction reference, but code reuse needs explicit license clarification from the author. |
| 5 | [MattBryantt/fpl](https://github.com/MattBryantt/fpl/tree/95643e93b9d1cb72a6621ed6e74b7b948dabad98) | `95643e9` | No license file or declaration | Default-branch commit 2026-10-08 | Static board and browser worker described in repository setup; no README or demo identified | Interesting static-snapshot plus in-browser solver architecture, but no permission to copy or distribute the code. Reference only. |
| 6 | [FPL Decision Engine](https://github.com/Linus-J/FPL-decision-engine/tree/b77f804db54242892b3663ff2b19301f2b54bdcd) | `b77f804` | No license file or declaration | Default-branch commit 2026-10-16 | Local Streamlit dashboard with squad, fixtures, news, history and chip pages | Comprehensive product flow, but tightly coupled to its SQLite/data engine, dependencies and absent license. Reference for screen inventory only. |
| 7 | [fpl-optimizer](https://github.com/rishabhchitlangia/fpl-optimizer/tree/0467574d9f1859b10b7a6d3aa620e09c199b2cae) | `0467574` | No license file or declaration | Default-branch commit 2026-10-17 | CLI output plus an optional local HTML pitch | Small Python/PuLP implementation with a useful pitch view, but little reusable web UI and no license permission. |

## Feature comparison

Legend: **Yes** means the reviewed README/source exposes the feature; **Partial** means it
is limited or uses a simplified heuristic; **No** means it was not found. This is a feature
inventory, not a quality ranking.

| Project | Forecasts | Squad pitch | Transfers/planning | Captain/chips | Team import | Result/history |
| --- | --- | --- | --- | --- | --- | --- |
| PITCHSIDE | Yes | Yes | Yes, multi-gameweek | Yes / yes | Yes, FPL entry ID | Yes |
| FPL Forecast | Yes, probabilistic | Yes, lineup view | Partial, single-gameweek fixed squad | Yes / No | Manual legal squad | Yes |
| Open FPL | Partial statistics | Yes | Yes, multi-gameweek planner | Yes / yes | Yes, FPL entry ID | No |
| FPL Helper | Partial, heuristic scores | Yes | Yes, heuristic suggestions | Yes / yes | Yes, FPL team ID | Yes, live points |
| MattBryantt/fpl | Yes | Yes | Yes, multi-gameweek | Yes / yes | Yes, team ID | Not found |
| FPL Decision Engine | Yes, distributional | Yes | Yes | Yes / yes | Yes, FPL API state | Yes |
| fpl-optimizer | Yes | Yes, optional local page | Yes, multi-gameweek | Yes / yes | Yes, team ID | Partial, CLI tables |

## License and attribution findings

Licenses are not interchangeable and data rights are separate from code rights. These
notes summarize repository evidence at the pinned revisions; they are not a legal opinion.

| Project | Code reuse position | Required action before copying |
| --- | --- | --- |
| PITCHSIDE | MIT permits reuse, modification and redistribution. | Preserve the copyright and permission notice in copies or substantial portions; add a project third-party notice. Do not copy provider data or implied FPL/Premier League assets. |
| FPL Forecast | AGPL-3.0-only is strong copyleft with network-source obligations. | Treat as reference only unless the project deliberately accepts AGPL for an isolated combined work and completes a compatibility review. Do not copy into the PolyForm Noncommercial codebase by default. |
| Open FPL | Apache-2.0 is declared in manifests, but the reviewed tree has no top-level license/notice file. | Obtain the authoritative license and NOTICE text, preserve notices, and record modifications before reuse. |
| FPL Helper | README claims MIT, but the repository has no license file. | Ask the author to add or confirm the license before adapting code. |
| MattBryantt/fpl | No license found. | Do not copy or distribute code or assets without explicit permission. |
| FPL Decision Engine | No license found. | Do not copy or distribute code or assets without explicit permission. |
| fpl-optimizer | No license found. | Do not copy or distribute code or assets without explicit permission. |

Official FPL content, club marks, player images, fixtures and provider datasets are not
made reusable by a repository's software license. The first slice should use the
project's own identifiers and generated outputs, not copied third-party artwork or data.

## Why PITCHSIDE is the best small adaptation

The reviewed frontend separates a generated static bundle from the UI. Its
[`publish.py`](https://github.com/bjarkiSigur7/fpl-ai-assistant/blob/42ef49820f12432ee6d3d43e89ca208dc4fa7e18/backend/src/fplai/publish.py)
documents the bundle files, and the frontend adapts them through
[`staticBundle.ts`](https://github.com/bjarkiSigur7/fpl-ai-assistant/blob/42ef49820f12432ee6d3d43e89ca208dc4fa7e18/frontend/src/lib/staticBundle.ts).
The reusable presentation points are concentrated in
[`Pitch.tsx`](https://github.com/bjarkiSigur7/fpl-ai-assistant/blob/42ef49820f12432ee6d3d43e89ca208dc4fa7e18/frontend/src/components/Pitch.tsx),
[`VerdictCard.tsx`](https://github.com/bjarkiSigur7/fpl-ai-assistant/blob/42ef49820f12432ee6d3d43e89ca208dc4fa7e18/frontend/src/components/VerdictCard.tsx)
and the shared UI helpers. The frontend stack is Next.js 16, React 19, Tailwind CSS 4
and a small dependency surface.

That boundary matches the desired architecture: Python remains authoritative, a generated
artifact describes one run, and the browser only presents that artifact. It also makes
failure states testable: a missing bundle, stale run or unavailable comparison should be a
visible UI state rather than a fabricated recommendation.

## Proposed bounded first slice

This estimate is provisional and excludes discovery, deployment and live owner-state
validation.

1. **Define one read-only run contract (about 1 day).** Include run identity and hash,
   target gameweek and deadline, data cut-off, readiness state, squad/XI/bench, captain and
   vice, proposed transfers and hits, horizon, comparison availability and explicit caveats.
   Keep forecast and observed outcomes visibly separate.
2. **Adapt only the presentation layer (about 1-2 days).** Start with a single-screen pitch,
   recommendation card and compact evidence panel. Preserve the MIT notice, record the
   pinned upstream revision and identify every copied or adapted file.
3. **Integrate one preserved run and failure states (about 1-2 days).** Feed the screen from
   existing generated artifacts through a small adapter. Add contract tests for missing,
   stale, failed and experimental runs, plus a narrow-screen and keyboard smoke check.

Total rough effort: **3-5 engineer-days**. This is not a delivery commitment; #20 discovery
may reduce or expand the scope before implementation.

## Reuse boundaries

- Reuse presentational code and interaction patterns, not another project's forecasting
  model, optimizer, data or published claims.
- Keep all FPL rules, pricing, transfer hits, legality checks and scoring on the Python side.
- Do not copy club badges, player images, official FPL imagery or provider datasets.
- Preserve upstream copyright, license text, attribution and a record of modifications.
- Do not describe a synthetic or preserved-run screen as live-validated; that remains
  gated by #5 and the product pilot in #20.
- Do not add a dashboard dependency to the production weekly path. The UI consumes a
  preserved artifact and never becomes the authority for a recommendation.

## Sources reviewed

- PITCHSIDE: repository, `README.md`, `LICENSE`, frontend `package.json`, `Pitch.tsx`,
  `VerdictCard.tsx`, `staticBundle.ts`, screenshots and live site.
- FPL Forecast: repository, `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `DATA_NOTICE.md`,
  frontend source and public dashboard.
- Open FPL: repository, `README.md`, package manifests, team-planner source and live site.
- FPL Helper: repository, `README.md`, Next.js source, pitch/transfer/captain/chip components.
- MattBryantt/fpl: repository, setup notes, static build server and browser UI source.
- FPL Decision Engine: repository, `README.md`, Streamlit dashboard source and project manifest.
- fpl-optimizer: repository, `README.md`, Python solver/tests and optional pitch output.
