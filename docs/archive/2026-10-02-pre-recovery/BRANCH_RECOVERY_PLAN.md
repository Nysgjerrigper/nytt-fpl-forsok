# Branch recovery and research run plan

Started 2026-10-02. Execution is isolated on `fix/branch-recovery`; production remains `single:catboost`.

## Goal and completion conditions

Bring the project to a small set of purposeful branches without losing unique code,
research evidence, ignored datasets, or checkpoints. Recover the useful forecast-screen
work into a reviewable patch, complete a causal exploratory run, and document the
integration order for remaining engineering. No commit, merge, code push, production
parameter activation, or research promotion is authorized by this run.

This run is complete when the repaired screen has meaningful tests and real artifacts,
every existing branch has a disposition, obsolete lanes have verified recovery archives,
safe cleanup has been verified, and the remaining approval-dependent integration work is
specified. A failed screen must remain explicitly failed, with a diagnosed blocker.

## Execution roles

The primary agent owns row semantics, research design, preservation decisions, and
verification. Read-only searcher agents audit commits and worktree differences. An
implementer agent handles the approved screen repair and its regression tests.
The repository's Haiku/Sonnet labels are Claude-specific; this runtime uses its available
searcher and implementer roles rather than claiming to execute unavailable Claude models.

## Ordered work

1. **Inventory and preserve.** Compare branch ancestry and actual dirty files. Classify
   active processes before retirement. Archive complete obsolete worktrees, including
   ignored files and runner reservations; verify each file against archive bytes and
   recheck source hashes before removing a worktree. Preserve recovery instructions.
2. **Repair exploratory screening.** Fit and predict fixture rows as the pipeline already
   does. Sum predictions and outcomes to player-position-gameweek totals for fair
   combiner comparisons. Use training-only player-round totals for MASE. Reject ambiguous
   fixture duplicates. Test causality, double gameweeks, shared rows, and failure status.
3. **Execute the recovered screen.** Use the preserved GW231 dataset, verify its full SHA,
   and physically exclude rows after the registered selection cutoff before features or
   fitting. Reuse the original four-model exploratory specification and seed 0. Preserve
   dataset/code/parameter hashes, member OOF, diagnostics, weights, and correlations.
   This run has no promotion eligibility and does not consume GW153+ outcomes.
4. **Review integration candidates.** Separate exact-budget conservation, checkpoint
   persistence, optional output paths, and historical documentation from unvalidated
   preseason feature changes. Preserve all five original commits on the validation
   branch; do not merge the branch wholesale.
5. **Close with evidence.** Run the recovery worktree's full regression suite, inspect the
   actual diff, verify output hashes and final worktree/branch inventory, and publish a
   disposition and status report. Keep original forecast and live-audit worktrees until
   their remaining unique changes have been reconciled.

## Longer horizon: integration and research gates

After PO authorization for landing, prepare separate coherent changes in this order:

1. Review and revalidate `c1d1` provenance/cache, scoring corrections, runtime crash guards,
   packaging, and tests. Its recorded 215-test result is historical until rerun. Corrected
   scoring anchors are 2088 standard and 1900 origin; do not mix them with legacy scoring.
2. Validate exact fresh-build bank conservation. Its constraint change may alter rolling
   transfer decisions; run the mandatory same-input MILP comparison and confidence
   interval before adopting it. Preserve stronger captaincy assertions where appropriate.
3. Harden Optuna checkpoint identity with dataset/features hashes, folds, cutoff, seed,
   model parameters, and environment provenance. Persist or explicitly constrain sampler
   resume semantics; CPU and writable buffers are not neural crash fixes.
4. Review the recovered exploratory screen results. Register any new candidate before
   further evaluation; selection is discovery <=GW136 / selection GW137-152. Material
   modeling changes require realized MILP points and confidence intervals on GW153-183.
   GW191-221 is spent; GW232+ remains prospective and must not inform selection.
5. Retire reconciled branches only after integration or explicit archival disposition.

Do not rerun the rejected stable MoE map or MID selector unchanged. RealMLP/TabM require
a separately validated non-Darwin or fixed dependency runtime. Optional preseason features
require deadline cutoffs, duplicate checks, faithful feature semantics, and registered
evaluation before any production use.

## Initial branch disposition

| Branch/worktree | Disposition |
| --- | --- |
| `main` | Preserve production. |
| `fix/branch-recovery` | Active isolated screen recovery and plan. |
| `codex/position-specialist-moe` / 975b | Merged; archive complete local runner state, then retire. |
| 15c2 / bce4 / cacc detached lanes | Mostly integrated duplicates; archive all evidence, then retire. |
| `exp/position-moe` | Reconciliation found only superseded behavior; fully archived and retired. |
| `exp/position-moe-validation` | Preserve five unmerged commits and completed selection evidence. |
| `feature/forecast-audit` | Preserve unique guards and tests; recover screen separately. |
| `fix/live-current-season-api` | Preserve unique ingestion/audit edits pending reconciliation. |
| c1d1 detached worktree | Preserve substantive engineering and authoritative research artifacts. |
| `experimental/bayesian-mdp-manager` | Retain unique historical experiment; no further run proposed. |
| `experimental/per-player-model-selection` | Retain historical design; later MID pilot is rejected. |

## Recovery archives

Complete tar archives and per-file hash manifests live under
`experiments/branch_recovery_2026-10-02/archives/` in this worktree. These are local
recovery artifacts, not proposed source additions. Restore by creating a new Git
worktree at the recorded `head`, extracting the archived contents to a temporary
directory, and copying files into the new checkout **excluding the archived `.git`
file**. Its `.git` pointer names the retired worktree and must never be reused.
Verify hashes against the associated JSON manifest. Reservation/lock files are
historical evidence; their existence does not grant authority to restart a runner.

## Execution checkpoint: 2026-10-02

Completed this run: five obsolete worktrees archived and retired; two merged local
branches deleted; a fresh recovery branch created. Six worktrees remain, including
the primary checkout and recovery worktree. Five local branches remain. The original
four-model screen completed after the DGW repair; 202 tests passed independently.
All screen dataset/source/output hashes and causal cutoffs were independently checked.
See `BRANCH_RECOVERY_STATUS.md` for results and the concrete integration queue.
