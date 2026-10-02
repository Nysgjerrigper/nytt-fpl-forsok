# Recovery status: 2026-10-02

The authorized recovery run is complete. The longer integration plan is ready for
review; no commits, merges, source pushes, tuned-parameter activation, or production
strategy changes were made.

## Executed and verified

- Searcher agents audited remaining dirty worktrees and all five unmerged validation
  commits; the primary session spot-checked the load-bearing findings.
- The implementer repaired the forecast screen in an isolated recovery worktree.
  Independent full verification passed **202 tests** in 9.06 seconds. The 214 warnings
  are existing CPU-detection, PuLP deprecation, and DataFrame-fragmentation warnings.
- The original four-model, seed-0 screen now completes. It fits fixture rows and scores
  summed player rounds; OOF covers GW137-152 and diagnostics cover GW141-152 after
  four warmup weeks. There are **51,004 member rows**, including **3,604 DGW rows**.
- Dataset, code and output SHA hashes, unique member rows, four-expert shared coverage,
  member training cutoffs, and combiner training cutoffs were independently checked.
  `promotion_eligible=false`; no realized MILP result or confidence interval is claimed.
- Complete archives of 975b, 15c2, bce4, cacc and position-moe passed per-file checks.
  Source files were rehashed before retirement; post-retirement archive hashes were
  checked again. All ignored datasets, tuned/checkpoint files and runner state are kept.
- Retired those five worktrees and merged local branches
  `codex/position-specialist-moe` and `exp/position-moe`.

All archives and screen evidence live in this worktree under
`experiments/branch_recovery_2026-10-02/`. Archives are local recovery payloads,
not proposed files for Git. Recovery instructions are in `BRANCH_RECOVERY_PLAN.md`.

## Screen interpretation

CatBoost MAE-loss has the lowest MASE at each position in this exploratory comparison,
but total calibration is only 0.464 DEF, 0.536 FWD, 0.676 GK, and 0.613 MID. Its
apparently favorable absolute error accompanies substantial downward bias. Other
members/combiners have better mean calibration or RMSE. These are selection-window
diagnostics, not an assessment against the tuned production control. The sensible next
research action is a registered, mean-aligned candidate comparison after engineering
integration, not selecting a production model from this table.

## Remaining work with concrete recommendations

| Work | Finding | Next action |
| --- | --- | --- |
| c1d1 engineering | Unique provenance/cache, scoring, packaging and crash guards; original recorded 215-test checkpoint is historical. | Review actual diff; prepare separate coherent integration patches and rerun tests. Preserve all research evidence. |
| `150517e` exact bank | Useful unmerged conservation fix. Constraint equality can change decisions; the original commit also weakens two membership assertions. | Extract constraint and exact-bank regression separately, review assertions, run same-input MILP comparison plus CI before adoption. |
| `5422b5f` checkpoints | Storage identity omits dataset/features hashes and fold count; seeded sampler restart is not persisted sampler state. | Implement provenance-bound resume semantics before integration. CPU/buffer changes do not solve neural crashes. |
| `9a2a652` preseason | No deadline cutoff and inadequate duplicate identity checks; rolling reconstruction uses approximate historical means. | Preserve the candidate; do not activate. Extract output-path utility only if independently useful. |
| `3290447`, `c02eea7` docs | Historical August advice and stale research status; product explanation still useful. | Preserve dated advice and selectively update product guide after current integration. |
| Forecast audit guards | Convex weights, positive-member handling, tuned cutoff admission, chronological partitions, comparable OOF panels are useful and unique. | Review and port separately, reconcile with c1d1, add DGW round semantics before tournament use. Keep original branch meanwhile. |
| Live API audit | Unique ingestion and budget edits remain. Parameter recovery exists elsewhere; price robustness/calibration are unresolved. | Compare against c1d1 and exact-bank patch, consolidate only the verified differences. |
| Historical remote experiments | Unique code/design retained; Bayesian experiment underperformed, later MID pilot rejected. | Preserve as research history; no unchanged reruns. |

All five validation commits remain recoverable on `exp/position-moe-validation`.
The original forecast-audit branch retains useful guards beyond the recovered screen.
The live API branch and c1d1 remain intact. No useful unique work was discarded.

## Runtime and holdout gates

The process inventory showed no FPL tuning/backtest runner; the unnamed Python process
was checked separately and belonged to `Developer/UiT Python lab`, outside this project.
No locks were cleared and no old runner authority was transferred.

Neural tuning remains 44/52 from preserved evidence. RealMLP/TabM require a demonstrated
runtime fix before execution. The stable MoE map and MID selector were rejected and
should not be repeated unchanged. Selection remains capped at GW152; GW191-221 is
spent and GW232+ remains prospective. No live refresh occurred in this run.
