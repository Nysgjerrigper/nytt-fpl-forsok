# Reproducible forecast research

## Preserved evidence and current interfaces

The historical registrations, source archives, result summaries and verification
reports imported from PR #98 retain their original bytes. They describe their
pinned evaluated code, not the integration changes made on 2026-10-31. No models
were refitted and no historical MILP replay was rerun for that integration.
`experiments/pr98_split/source_results.csv` preserves the original run registry;
the main registry keeps the union of both branches' columns and nonduplicate rows.
The split manifest records source file hashes and their disposition.

Active callers use `params_by_pos` with explicit parameters and a seed. Registered
recipes retain their existing admission rules. Separate exploratory recipes admit
only the opt-in CNN/blend members; they do not extend the registered MoE universe.
Cold priors require an explicit global origin, including the opt-in weekly CLI.
The legacy optimizer retains main's native certificate and FT constraints, with
additional research receipt fields. Labelled historical net exports retain their
gross source and are validated without charging transfer hits twice.

Native CNN tests require the optional `requirements-cnn.txt` runtime. Standard CI
skips only the two native drivers when Torch is absent; validation tests still run.
When installed, native validation stays in fresh subprocesses. A failed native fit
must not be retried without a separately validated runtime.

Historical scripts under `experiments/` are evidence tied to their registered
inputs and source hashes. Do not rewrite those scripts or their hashes to make an
old verification pass against current code. Any new run needs a new registration,
output directory and source binding under the canonical research protocol.


**2026-10-04 audit qualification:** historical solve counts below are counts of
reported PuLP `Optimal` statuses. The old wrapper omitted `gapRel=0`, retaining
HiGHS' `0.0001` default, and did not check solution status. Those saved logs do not
prove exact optimality. The audit also corrected lookahead free-transfer rollover.
Saved squads, realized scores and negative findings remain historical evidence;
comparisons under the corrected optimizer require new, paired prediction replays.
See the dated setup/logic audit in `RESEARCH_LOG.md`.

**Corrected replay completed:** the original-byte MAE/RMSE forecasts were replayed for seeds 0/1/2 with 372 native-certified solves and independent points/budget reconstruction. RMSE minus MAE was −55/−83/−182 standard and +15/−134/−183 origin points. This does not support switching configurations. The six paired confidence intervals, missing-outcome disclosure and 25.34-minute local runtime are recorded in the dated corrected-optimizer entry in `RESEARCH_LOG.md` and `experiments/corrected_optimizer_replay_2026-10-04/replay_v3/verification.json`. Other historical candidate families have not yet been replayed under the corrected optimizer.

Production remains `single:catboost`. The research package compares explicitly pinned
CatBoost MAE/RMSE refits and four bounded feature removals. It neither restores the
missing historical production parameter namespace nor authorizes promotion.

## Run the first package

```bash
python -m fpl.model.research_package \
  --dataset /absolute/path/to/master_dataset_gw231.csv \
  --output-dir experiments/research_priorities_YYYY-MM-DD/selection_seed0 \
  --parameters selection --seed 0
```

Use a new directory for every invocation. Existing outputs are refused, including
incomplete runs. `--parameters defaults` pins registry defaults instead of tracked
selection studies. `selection` validates all eight seed-0 MAE/RMSE artifacts against
their position, model, stage, discovery cutoff and hashes. Missing or inadmissible
artifacts fail closed; there is no fallback. Seeds 1/2 change refit randomness only;
they do not become newly tuned studies. Parameters, five threads and disabled model
file writing are recorded in registration, with input bytes, library versions,
ordered feature columns and every pipeline source hash.
The tuned MAE/RMSE comparison evaluates complete configurations: their tree capacity
also differs, so it does not isolate the causal effect of the loss function.

The current chronology is derived by `protocol_cutoffs()`: discovery through GW136,
selection GW137–152, historical reference GW153–183. The verified frozen input is
trimmed before each feature construction. Selection is built separately through
GW152; GW232+ cannot enter this loader. GW191–221 remains spent. The full window
GW153–183 has been examined repeatedly and is not an untouched holdout.

Before any fitting, `registration.json` and the experiment registry record:

- Exact MAE/RMSE settings and identical-input control definition.
- Full features and four separate MAE removals: fixture, minutes, opponent and xP.
- Origin horizon 3, retraining every 4 rounds and seed.
- Chip-disabled legacy HiGHS MILP, zero relative gap and corrected `autosubs_v2` scoring.
- Net points after four points per transfer hit, preserving gross points separately.

All six variants run on selection origins. Highest earlier origin MILP points chooses
at most one ablation; ties retain full features. RMSE is a predeclared challenger.
`selection_lock.json` is written before reference evaluation; losing ablations are
not evaluated there. Standard and origin forecasts then feed identical optimizers.
Standard horizon forecasts have future form information; origin forecasts freeze
form at each deadline. Neither archived prices nor fixture ratings currently prove
complete historical publication-time availability.

Each run preserves prediction bytes before optimization, validates finite corrected
scoring inputs and fixture/DGW identity, and requires every solve to be optimal.
Admission/result JSONs retain settings, diagnostics, hashes, timings, gross/net points
and hits. CSV/log payloads remain local and ignored; compact JSON evidence is tracked.
The recorded `fit_seconds` measures forecast generation, including refits, prediction
and origin snapshot rebuilding; it is not a pure single-fit CPU benchmark.
`summary.json` checkpoints completed variants and records incomplete failures.
A changed pipeline source after registration stops the package. Paired comparisons
use `fpl.milp.compare_backtests`, 10,000 block-bootstrap draws, block length 3.
Forecast diagnostics include player-round MAE, MASE using earlier training rounds,
RMSE, bias, total calibration, Spearman and top-one capture. They do not select a
production winner. One seed and this exploratory family do not satisfy the registered
MoE promotion protocol.

## Frozen probabilistic admission

`fpl.model.probabilistic` and `fpl.model.probabilistic_buckets` now use the same verified
frozen input, accept `--dataset`, and trim raw rows before constructing features.
Invalid/inverted windows, static fitting/tuning beyond the selection ceiling, and
cutoffs beyond the frozen snapshot are refused. Static and causal walk-forward modes
retain their existing different fitting semantics. Bucket exports preserve minutes
and fixture/opponent/home identity for corrected scoring. This hardens admission;
it does not establish new probabilistic calibration or decision gains.

## Opt-in cold-start coverage

```bash
python -m fpl.run_week --team-id <team-id> --horizon 3 --cold-start-priors
```

This live command refreshes/refits; it is not a frozen experiment. The opt-in flag
adds API-registered stable player codes missing from the historical snapshot, using
current team, position and price. Unknown form/outcomes remain missing. Synthetic
rows never enter training. Registered cold rows bypass the trees and use a pooled
position distribution mixed with a team-position distribution, weighted by
`n/(n+20)`. An unseen team falls back to position history. DNP fixture outcomes are
included. Existing historical rows retain their model forecasts.

Owned players must all occur in the exported forecast pool before either live
optimizer runs. A registered player with no fixtures across the requested horizon
can still lack a forecast row; the driver now reports that limitation explicitly
instead of optimizing incomplete holdings. The legacy optimizer applies the same
guard to direct continuing-squad calls. No synthetic fixtures are inserted.

Exports add `is_cold_start`, `forecast_source`, `prior_std`, `p_played`, `p60` and
`prior_count`. Probabilities are empirical fixture frequencies; `prior_std` is an
outcome spread, not a calibrated confidence/prediction interval. The current mean
MILP does not consume this uncertainty. Availability heuristically gates the prior
with an independent Bernoulli event: mean `q*m`, variance
`q*v + q*(1-q)*m*m`, participation probabilities `q*p`.

`retrospective_cold_start_predictions(raw, start_gw, end_gw)` supplies a separate
first-recorded-player cohort diagnostic using strictly earlier priors, retaining
DNPs and every first-round DGW fixture. It does not reconstruct historical API
registration or establish a cold-start MILP benefit. Missing registered historical
pools remain a limitation on that evaluation; no such benefit is claimed.

## Subsequent work

The first registered seed-0 package completed with 282 optimal solves and 52.6
minutes of summed forecast-generation/MILP runtime. New MAE controls scored
2082 standard / 1905 origin net points; tuned RMSE scored 1939 / 1852; the locked
opponent-feature removal scored 2040 / 1932. Paired standard/origin differences
and 95% CIs were RMSE -143 [-255,-17] / -53 [-227.025,149], and opponent removal
-42 [-141,74] / +27 [-87,136]. Retain production and full features: no challenger
established an improvement. See `RESEARCH_LOG.md` and
`experiments/research_priorities_2026-10-03/selection_seed0/summary.json`.

That experiment's original evaluated sources are preserved in `evaluated_sources/`.
Its selection-summary bytes were saved before reference results and verified
against the original finalist lock. The original completion registry summary hash
describes canonical JSON content; `verification.json` and the later VERIFIED
registry entry bind the actual final file bytes. The current runner automatically
preserves immutable selection bytes and records final file hashes. Subsequent
admission/hash hardening did not change the preserved forecasts or scores.

New runs also archive registered sources before evaluation and recheck the full
frozen input, tuned artifacts, registration, source archive, selection locks and
every prior scored output around fitting, ranking and paired comparisons. A changed
file leaves an incomplete checkpoint and stops the run without an automatic retry.
Seeds 1/2 reuse the preserved seed-zero tuned studies with explicit refit seed
overrides; they are not new tuning studies or independent outcome samples.

The completed serial seeds 1/2 took 125.05 minutes together. Each had 282
optimal solves. All sources, inputs, selection locks and outputs were independently
verified, including recomputed paired confidence intervals. Reference net points:

| Refit seed | MAE standard/origin | RMSE standard/origin | Earlier-selected ablation standard/origin |
|---:|---:|---:|---|
| 0 | 2082 / 1905 | 1939 / 1852 | Without opponent: 2040 / 1932 |
| 1 | 1937 / 1965 | 1979 / 1913 | Without fixture: 1958 / 1917 |
| 2 | 1979 / 1987 | 1882 / 1881 | Without minutes: 2101 / 1893 |

RMSE origin deltas were -53, -52 and -106. Each seed selected a different
ablation, and none of seeds 1/2's challenger intervals excludes zero. These
results support retaining the full-feature MAE research control; they do not
establish a production promotion or a pooled ablation effect. Seed 0's source
archive differs from later engineering-hardened sources, as disclosed before
the repetitions; numerical features, tuning studies and settings match.

External identity, historical coverage and publication timestamps must be audited
before integrating Understat, cup workloads/rest/Elo, or richer availability.
Blends, uncertainty calibration, CNNs and scenario optimization require their own
registered hypotheses and admissible data. No broad model-adapter rewrite is needed
for this package. Current defensive-contribution scoring cannot be assessed from
this 2024/25 reference alone.

## Earlier-only interval diagnostics

```bash
python -m fpl.model.interval_research \
  --dataset /absolute/path/to/master_dataset_gw231.csv \
  --output-dir experiments/interval_calibration_YYYY-MM-DD/seed0
```

The separate runner registers and archives exact settings and sources before fitting.
It fits existing LightGBM p10/p50/p90 fixture models, sums distinct fixtures into
player rounds, and computes signed interval corrections from eight strictly earlier
rounds. Selection fits through GW128 and calibrates GW129–136 before evaluating
GW137–152; reference fits through GW144 and calibrates GW145–152 before evaluating
GW153–183. Boundaries derive from the current protocol. There is no tuning or
post-reference recalibration. Summed fixture quantiles approximate a round interval;
they are not exact quantiles of a sum. Dependence across players and time prevents
an established exchangeability coverage guarantee.

The correction uses the finite order statistic from
[Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222), with a
conservative modification retaining the unchanged median inside each interval.
Coverage, width, interval score and tail misses are preserved alongside calibration
and evaluation CSV hashes. These are distribution diagnostics; neither point-model
forecasts nor MILP inputs change.

New interval runs seal registration, archived sources and each phase's CSV bytes
immediately after writing them. They recheck earlier artifacts before and after
later phases and completion; the final summary carries an immutable manifest.
This subsequent engineering hardening does not rewrite the completed seed-zero
study or claim that its original runner had these stronger checks.

The completed seed-0 run took 44.7 seconds locally. All eight position/phase
corrections were zero, so no interval improvement was observed. Reference coverage
was 90.0% overall, with 98.9% for DNP rounds and 78.3% among played rounds. Actual
participation is known after the match; these descriptive subgroups cannot become
deadline-known calibration inputs. Large zero-score atoms explain the unchanged
finite order statistics. Preserve the null result without retuning on the reference.

## Bounded sequence challenger

```bash
python -m fpl.model.cnn_research \
  --dataset /absolute/path/to/master_dataset_gw231.csv \
  --output-dir experiments/cnn_research_YYYY-MM-DD/seed0
```

Eight prior recorded player rounds provide eight statistic channels and explicit
presence masks. Recorded rounds can span calendar gaps and seasons; they are not
eight calendar weeks. DGW fixtures are summed before constructing history, missing
collection stays missing, and current-round outcomes are excluded. All 72 sequence
columns are rebuilt from history strictly before each forecast origin, then frozen
over its horizon. Trees use the same added columns for a controlled comparison.
The CNN uses fold-local preprocessing and fixed architecture/epochs without tuning.
Install its separately pinned runtime with
`python -m pip install -r requirements-cnn.txt` after the ordinary project setup.
The preserved CPU study used PyTorch 2.13.0. PyTabKit is not needed for this lane.

The runtime keeps Torch in a fresh persistent CPU child process. A mixed GBM/Torch
in-process validation previously segfaulted; that path is guarded and is not retried.
Protocol errors, child crashes and timeouts fail closed. Passing isolated synthetic
tests establishes runtime behavior, not forecasting quality.

The completed seed-0 study fit once through GW136 and compared origin decisions
on GW137–152. Net points were 926 for base MAE, 893 for sequence MAE, 1,024 for
sequence RMSE and 982 for the CNN. All 64 solves were optimal. The CNN failed its
strict advancement gate against the strongest tree, so reference data were not
built or evaluated. The run took 8.2 minutes; its forecast timers include training,
worker startup, predictions and snapshots rather than pure training time.
This bounds one candidate and does not reject CNNs generally. No production or
registered MoE family was changed.

## Earlier-only temporal blend

```bash
python -m fpl.model.blend_research \
  --dataset /absolute/path/to/master_dataset_gw231.csv \
  --source-dir experiments/cnn_research_YYYY-MM-DD/seed0 \
  --output-dir experiments/temporal_blend_YYYY-MM-DD/seed0
```

The runner verifies the completed source study against its original source archive
and hashed cached predictions. It fits one convex CNN weight per position from
current-origin player rounds in GW137–144. The no-intercept least-squares weight
is clipped to [0, 1], with identical predictions retaining the tree. Distinct DGW
fixtures are summed before fitting; each fitted weight applies across the horizon.

The later GW145–152 pilot replays saved forecasts with a fresh, identical optimizer
state for blend and RMSE control. It locks weights and settings before scoring.
Only a strict net-points gain admits reference fitting through GW152, comparing
blend, sequence RMSE and full MAE on GW153–183 in standard and origin modes.
These windows derive from the current protocol. Paired block-bootstrap intervals
accompany realized points. All sources, cached inputs, locks and scored outputs
are repeatedly hashed; mutation stops the study without a retry.

The full selection window was already examined by the CNN study, including choosing
the tree member. This later pilot is exploratory continuation, not independent
unseen validation. Earlier-only weight fitting does not undo that member-selection
reuse. Reference results cannot update the weights. This family remains separate
from production, tuning and the registered MoE promotion protocol.

The completed seed-zero run took 24.3 minutes and proved all 202 solves optimal.
Its late pilot scored 579 versus 504 net points (delta +75, CI +9 to +141).
The reference outcomes were:

| Fixed-fit variant | Standard net points | Origin net points |
|---|---:|---:|
| Full MAE | 2,032 | 1,906 |
| Sequence RMSE | 2,056 | 1,944 |
| Temporal blend | 2,112 | 1,956 |

Every reference paired interval includes zero. Blend versus RMSE: standard +56
[-22, 161], origin +12 [-148, 147.025]. Blend versus MAE: standard +80 [-48, 226],
origin +50 [-172, 251]. These are exploratory findings with fixed-fit controls,
not the earlier rolling-refit package or historical production replay. The pilot
gain does not establish a repeatable reference gain or support promotion.

## Scenario research admission

Joint scenario optimization remains conditional on an explicit risk objective,
validated origin-horizon marginal distributions and dependence between players
and fixtures. The interval study above validates static marginal intervals only;
it supplies neither joint scenarios nor a dependence model. A successful mean
blend would not establish those prerequisites. Expected captain points depend
on the mean: doubling equal means leaves them equal regardless of variance.
Risk-sensitive selection therefore needs a stated objective, such as downside
CVaR or upside/rank utility, before changing the optimizer. Production remains
the existing expected-points formulation.

## Historical price-path limitation

Origin freezing covers player form, including the added sequence channels. It does
not freeze the historical price path: `predict.origin_based_predictions` preserves
each target row's `value`, and `milp.optimize` builds a player/GW value matrix used
for future-horizon purchase and sale constraints. Future realized prices are not
deadline-known inputs at the origin. The comparisons above share this registered
replay protocol but do not establish fully point-in-time deployment performance.
A separate preregistered origin-known price experiment is needed; changing prices
inside a running comparison would invalidate its control. Historical fixture
ratings and availability snapshots also need publication-time evidence.

The opt-in price proxy study reuses the completed seed-zero MAE/RMSE origin
forecasts without fitting models. It compares target-price replay with the latest
price recorded strictly before each origin, held constant throughout its horizon:

```bash
python -m fpl.model.price_research \
  --dataset /path/to/verified/master_dataset_gw231.csv \
  --source-dir experiments/research_priorities_2026-10-03/selection_seed0 \
  --output-dir experiments/price_provenance_YYYY-MM-DD/seed0
```

Both policies share the same eligible players: the phase's forecast-player union
with a prior observed price at that origin. Unknown prices exclude players; an
owned player lacking a price stops the run. Catalogs cover blank-round holdings
as well as players with horizon fixtures. Each origin uses its own catalog, with
no global deduplication of origin prices. Four arms run in each window, with
unchanged forecast bytes and fresh identical initial states. All catalogs,
cached inputs, archived code and outputs are sealed before and around scoring.

These restricted controls differ from the original unrestricted replay. A lagged
observation is not a verified deadline capture, and this experiment retains the
legacy purchase/sale formulation and free-transfer rules. Archived roster,
team/position, fixture ratings, availability and true selling-price provenance
remain separate limitations. Paired historical points measure sensitivity to
this price proxy, not full deployment validity or a production promotion.
The proxy also replaces the current origin's target-row price, so differences
combine a changed starting price with a frozen horizon. They do not isolate
future-price lookahead from current-price proxy error or preseason repricing.

The completed seed-zero proxy study took 13.6 minutes, with 188 optimal solves
and independently reconstructed budgets. Reference net points were:

| Cached model | Common-eligibility target replay | Lagged-price as-of |
|---|---:|---:|
| MAE | 1905 | 1889 |
| RMSE | 1678 | 2110 |

RMSE minus MAE flips from -227, CI [-427, -18], under replay to +221,
CI [36, 483], under the lagged proxy. Within RMSE, the price-policy difference
is +432, CI [326, 569.025]; within MAE, -16, CI [-174, 90]. This single-seed,
repeatedly examined result exposes sensitivity to price assumptions, not a
production winner. In particular, current-price proxy error and horizon effects
are combined. The original unrestricted RMSE scored 1852; that control allowed
a player without prior observed prices during GW153–176. Missing prior price
does not establish missing historical registration. The new common-eligibility
replay is a distinct control and cannot replace the original result silently.

Although catalog ages exceed 100 rounds, selected players under the as-of policies
had observations at most two rounds old in this run. One selected RMSE replay
player-round had a 39-round-old observation and no horizon fixture; this remains
an availability/roster provenance limitation, not evidence of a valid cheap bench.

To separate horizon effects from the changed starting price, the next runner
holds the reconstructed replay price at the current origin constant across the
horizon. It reuses the verified replay/as-of results and scores only MAE/RMSE
under this third policy, with the same cached forecasts and eligibility catalogs:

```bash
python -m fpl.model.horizon_price_research \
  --dataset /path/to/verified/master_dataset_gw231.csv \
  --source-dir experiments/price_provenance_2026-10-03/seed0 \
  --output-dir experiments/price_horizon_YYYY-MM-DD/seed0
```

This isolates horizon price changes conditional on the same reconstructed current
price vector. That vector can retain legacy forward/back filling and target-row
observations; it is not proof of the prices available at historical deadlines.
The four new evaluations require 94 optimal solves and no model fits. Both old
controls and new outputs retain separate archived code and immutable manifests.

The completed horizon-only study took 6.8 minutes, with 94 optimal solves and
independently verified budgets. Earlier net points were MAE 930/RMSE 993; reference
net points MAE 1938/RMSE 1856. RMSE minus MAE was -82, CI [-259, 98]. Relative to
their common-eligibility replay, MAE changed +33, CI [-24, 91], and RMSE +178,
CI [42, 312]. The model ranking remains MAE-first in this seed, with uncertainty.

The lagged-price proxy has a concrete season-boundary limitation: its initial
reference MAE squad costs 1035 and RMSE squad 1080 under reconstructed starting-round
prices, against a 1000-unit budget. Those prices are still not verified deadline
prices, but this counterfactual shows that carrying old-season values into a new
season can change budget feasibility substantially. The proxy's 2110 score cannot
establish a practical model improvement. Subsequent price robustness excludes
this proxy for that reason and fixes both replay/current-horizon policies before
scoring seeds 1/2; it cannot manufacture a fresh untouched validation window.

The preregistered reference-only repetitions for seeds 1/2 completed in 22.4
minutes, using cached forecasts and no new fitting or tuning. Independent checks
verified 248 optimal solves, 248 reconstructed budgets, all paired reports and
432 sealed input/output files. Together with the separately verified seed-zero
study, the reference net points are:

| Refit seed | Replay MAE | Replay RMSE | Fixed-current MAE | Fixed-current RMSE | Fixed-current RMSE minus MAE, 95% CI |
|---:|---:|---:|---:|---:|---|
| 0 | 1905 | 1678 | 1938 | 1856 | -82 [-259, 98] |
| 1 | 1965 | 1891 | 1958 | 1872 | -86 [-209, 27.025] |
| 2 | 1995 | 1881 | 1999 | 2016 | +17 [-116.025, 181.025] |

All arms use a common known-price eligibility rule within each seed. They are
distinct from the unrestricted source controls. Fixing the current price vector
changes RMSE's points by +178, -19 and +135 across seeds, compared with +33, -7
and +4 for MAE. The seed-one RMSE policy CI is [-213.025, 144], and seed-two
[-21, 337]; neither establishes a repeated policy gain. Every fixed-current model
comparison includes zero. The seeds share historical outcomes and reuse the
seed-zero tuning studies; they are not independent samples and should not be
pooled as though they were.

Keep production `single:catboost` and retain the full-feature MAE research control.
The next prerequisite is deadline-bound historical prices and registered rosters,
including season repricing and selling-price rules. Additional model complexity
cannot resolve those data gaps. The current policies remain conditional research
probes: reconstructed current prices still lack historical publication proof.

The cached replication entrypoint is:

```bash
python -m fpl.model.price_research \
  --dataset /path/to/verified/master_dataset_gw231.csv \
  --source-dir experiments/research_priorities_2026-10-03/selection_seed1 \
  --output-dir experiments/price_provenance_YYYY-MM-DD/reference_seed1 \
  --seed 1 --reference-only --include-current-proxy --skip-lagged-proxy
```

Register any new seeds or policies before scoring. Preserve completed directories;
the runner rejects overwrites. The parent protocol and independent verification
are in `experiments/price_provenance_2026-10-03/seed_robustness_*.json`.

## Evidence verification

`experiments/research_priorities_2026-10-03/package_verification.json` binds the
ten completed study summaries and their independent verification reports. It
records 1,642 optimal MILP solves across the package and rechecks 487 unique
immutable output files; reused controls are counted only in their original study.
It is a scientific evidence snapshot, not a promotion assessment. The completed
source archives remain distinct from the later hardened implementation.

```bash
python experiments/research_priorities_2026-10-03/verify_package.py
```

The latest full suite passed 1,260 tests, with four native-parent skips covered by
separate fresh-process tests. Production remains `single:catboost`.

## Selected historical availability captures

The offline adapter in `fpl.data.deadline_snapshots` admits independently dated
Internet Archive captures. See [external data admission](EXTERNAL_DATA.md) for
the CDX byte proof, freshness checks and incomplete coverage. The current corpus
admits 78/136 discovery, 16/16 selection and 28/31 historical reference origins.
Missing capture/player signals stay NaN and outcome fixture rows remain present.

The research challenger adds six status indicators, published next-round chance
and capture age to the same 115-feature MAE control, with identical tracked
parameters. Its exact-origin snapshot callback freezes all eight additions across
the horizon. A strict earlier selection gain is required before reference scoring;
all inputs, coverage masks, catalogs and settings are sealed before fitting.

```bash
python -m fpl.model.availability_research \
  --dataset /path/to/verified/master_dataset_gw231.csv \
  --audit-dir experiments/deadline_snapshot_audit_2026-10-03 \
  --output-dir experiments/availability_research_YYYY-MM-DD/seed0
```

Origin arms share strictly prior-price eligibility and freeze reconstructed
current prices across the horizon. These are conditional historical comparisons:
crawl-time availability may change before the deadline, current prices remain
reconstructed, and the forecast pool still comes from historical rows. Standard
forecasts retain future-round lookahead. No complete point-in-time, cold-start
decision-gain or production-promotion claim follows from this interface.

Verify completed results without repeating fitting or scoring:

```bash
python experiments/availability_research_2026-10-03/verify_study.py \
  experiments/availability_research_2026-10-03/seed0
```

The separate registered-prior coverage audit uses the admitted crawl-time rosters
and strictly earlier outcomes. Existing pooled priors cover all 191 missing codes
across the 44 admitted selection/reference origins; cumulative prior fitting took
10.26 seconds. This is coverage verification, with no target scoring or MILP result.
Its registration metadata correction clarifies that parsing the frozen CSV reads
outcome columns, while the calculation uses only strictly earlier outcomes.


The MAE availability family was replicated with fixed seed-zero tuned capacities.
Selection candidate minus control was +22 (seed0), -40 (seed1) and -44 (seed2).
Only seed0 passed the gate; its standard reference difference was -165, CI
[-357, 0.025], and origin difference -16, CI [-122.025, 51.025]. Seeds1/2
correctly have no reference evaluations. The three-seed family did not establish
a stable gain and remains research-only. Its parent/individual verification binds
220 optimal solves; the two additional seeds completed in 8.44 minutes.

An independently registered RMSE seed-zero pilot tests the same information with
its own fixed tracked parameter capacity. Use `--model-name catboost_rmse` with
the same entrypoint; the default remains MAE. The pilot's separate control and
candidate share inputs, pricing and the strict earlier selection gate. This is
not a pure loss comparison or a promotion protocol. The pilot registration is
under `experiments/availability_research_2026-10-03/rmse_pilot_registration.json`;
its results must be verified before interpretation.

The RMSE seed-zero pilot completed in 91.2 minutes with 156 optimal solves.
Reference standard control/candidate scored 1939/2055, difference +116,
paired CI [-16, 237]. Origin control/candidate scored 1856/1943, difference
+87, CI [-65, 207]. Both intervals include zero. This promising single-seed
result relative to the RMSE control is inconclusive; production remains unchanged.
The verified RMSE-inclusive package binds 2018 optimal solves and 811 unique
immutable outputs, with separate aggregate and timing reports preserving earlier
versions. The historical reference is repeatedly examined, not a fresh holdout.

RMSE refit seeds 1 and 2 are separately registered before fitting, using the
same frozen input, eight additions, archive coverage, prices and tracked
seed-zero capacity. They run serially with the same strict selection-origin
gate; losing or tied candidates have no reference evaluation. There is no new
tuning or feature selection. The seed-zero results were already examined, and
all seeds share historical outcomes. Registration and execution evidence are in
`experiments/availability_research_2026-10-03/rmse_robustness_*`; this is research
replication, not a production promotion protocol.

Both replications completed and rejected the candidate at selection: seed 1
scored 1062 versus control 1081 (-19, CI [-122, 57]); seed 2 scored 1050 versus
1116 (-66, CI [-181, 23]). Neither has a reference evaluation. The serial parent
took 23.85 minutes. The three-seed selection differences +3/-19/-66 do not
establish a stable gain. The separate completed replication package binds
16 studies, 2082 optimal solves and 953 unique immutable outputs. Earlier
reports remain preserved. See `package_with_availability_replications_verification.json`
and `compute_timings_with_replications_verified.json` in the research-priorities
experiment directory.

Grouped uncertainty diagnostics reuse the verified quantile exports without
refitting. `python -m fpl.model.grouped_interval_research --base-dir BASE
--output-dir NEW --dataset FROZEN` preregisters fixed shifted-minutes groups,
fits adjustments on earlier calibration rows and reconstructs the stored
position-only control. Groups are low/mid/high at 0.2/0.8, with explicit missing
values and position fallback below 100 calibration rows. Reports include proper
interval score and a player-round-weighted moving-block confidence interval;
observed played status is diagnostic only. These outputs do not establish a
MILP benefit, coverage guarantee or production-promotion case.

## Measured local compute

The verified timing ledger is
`experiments/research_priorities_2026-10-03/compute_timings_verified.json`.
It binds the summaries and independent verification reports for 13 completed
studies. Forecast-generation timing includes repeated fits, inference and origin
snapshot construction. Cached price replays exclude training. Whole-study wall
time remains absent where the original runner did not record it; arm subtotals
are not substitutes for elapsed time or active implementation time.

| Completed experiment | Measured wall time | Scope |
|---|---:|---|
| Availability MAE seed0 | 25.1 min | Selection and both reference modes, control/candidate |
| Availability MAE seed1 | 4.2 min | Selection only; candidate rejected |
| Availability MAE seed2 | 4.1 min | Selection only; candidate rejected |
| CNN seed0 | 8.2 min | Selection only; candidate rejected |
| Temporal blend seed0 | 24.3 min | Selection and both reference modes |
| Interval calibration seed0 | 44.7 sec | Two diagnostic phases; no MILP |
| Availability RMSE seed0 | 91.2 min | Selection and both reference modes, control/candidate |

Generate a new ledger from a verified package without fitting or solving:

```bash
python experiments/research_priorities_2026-10-03/summarize_compute.py \
  --package experiments/research_priorities_2026-10-03/package_with_availability_verification.json \
  --output /path/to/new_compute_timings.json
```

The output must be new. After the RMSE pilot's parent verification succeeds,
`experiments/availability_research_2026-10-03/finalize_rmse_evidence.py` rechecks
its input/source hashes and writes a separate package report, compute ledger and
finalization proof. Existing reports are preserved; it never repeats model
fitting or MILP. The finalizer rejects optimized Python and rechecks input/source
bytes after both child steps. Tests cover mutations during either step and
preservation of existing outputs. These measurements support compute planning,
not promotion.

## Registered missing-pool outcome diagnostic

The subsequent fixed coverage-only audit reuses 191 registered player-origin
prior rows and joins same-round frozen fixture labels by stable code. All 191
have observed labels (192 fixtures), with 93 positive-minute rows, 88 positive-
point rows and 180 summed observed points. The original prior fields remain
unchanged; registered team and position match all observed labels. These
repeated observations establish an actual coverage gap, not an achievable MILP
gain. Retrospective fixture counts never create forecasts or modify prior means.
The independent primary reconstruction passed all byte locks and grouped counts.
See `experiments/registered_cold_outcomes_2026-10-04/seed0/verification.json`.

## Bounded league-calendar workload challenger

The fixed research driver `experiments/league_workload_2026-10-04/run.py` tests
three unused calendar inputs with the same tracked MAE parameters and refit seed
as its fresh control. It registers before preprocessing, seals the ordered schema
and context before fitting, freezes current-origin context over the horizon and
admits reference features only after a strict selection-origin net-points gain.
Both origin arms share reconstructed current-price proxies. Missing league history
or origin context stays NaN; current DGWs never contribute to prior workload.
This is EPL-only recorded-calendar information, with incomplete proof of original
schedule publication, and supplies no cup/European/Elo inputs or promotion claim.
Use the verified frozen dataset explicitly and choose a fresh output directory:

```bash
PYTHONPATH="$PWD" python experiments/league_workload_2026-10-04/run.py \
  --dataset /path/to/master_dataset_gw231.csv --output-dir /path/to/new_study
```
