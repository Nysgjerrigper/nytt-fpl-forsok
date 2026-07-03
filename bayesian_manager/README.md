# bayesian_manager

An experimental, self-contained FPL manager implementing the belief-state MDP
approach from Matthews, Ramchurn & Chalkiadakis (2012), *"Competing with
Humans at Fantasy Football: Team Formation in Large Partially-Observable
Domains"* (AAAI 2012).

**This module does not touch, import from, or get imported by `fpl/`.** The
repo's production pipeline (`fpl/model/` + `fpl/milp/`) is a point-forecast
gradient-boosted-tree ensemble feeding a MILP squad optimizer. This module is
a completely different design: instead of predicting a single expected-points
number per player from historical stats, it maintains an explicit probability
distribution over each player's *role* (starts/subbed/unused) and *scoring
behaviour* (scores/assists given they play), updated via closed-form
Bayesian conjugate-prior rules as gameweeks are observed, then uses Monte
Carlo simulation of those beliefs to estimate expected points before handing
them to a knapsack solver. It exists to let the two approaches be compared
on the same historical data, not to replace the production pipeline.

## Why this design (recap of the paper, Sec 3)

FPL is framed as a belief-state MDP: each gameweek, the manager only
observes match outcomes, not player "ability" directly, so it maintains a
posterior belief over ability and re-plans as evidence arrives.

1. **Belief model** (`beliefs.py`): per player, a Dirichlet distribution
   `rho_p` over {start, sub, unused}, and Beta distributions `omega_p`
   (scores | plays) and `psi_p` (assists | plays). All three are conjugate
   priors, so updating on a gameweek's actual result is a closed-form
   pseudo-count increment - no numerical inference needed, which is what
   makes it cheap enough to update ~700 players every gameweek and re-sample
   them thousands of times per decision (see `simulate.py`).
2. **Club-level scoreline model** (`club_model.py`): player-level scoring
   only makes sense on top of a model of how many goals each club scores in
   a given match. See "Simplifications" below for how this deviates from
   the paper's actual Dixon & Robinson (1998) birth process.
3. **Outcome sampling** (`simulate.py`): draws one full plausible gameweek
   (who started, who got subbed on/off and when, who scored/assisted each
   goal) from the current beliefs, for a given squad. Averaging many draws
   gives each player's *expected* points this gameweek - the number the
   knapsack solver actually optimizes.
4. **Action generation via MKP** (`team_select.py`): squad/lineup/captain
   selection is a multi-dimensional knapsack problem (budget, formation,
   club limits, transfer-penalty dummy items) - a linear IP, solved with
   PuLP+CBC (already a project dependency) instead of the paper's CPLEX.
5. **Manager variants** (`manager.py`):
   - `myopic_decision`: the paper's M1/M2 baseline - one MKP solve on
     current beliefs, no lookahead.
   - `q_learning_decision`: the paper's cheaper alternative to full
     depth-limited DFS over the Bellman equation (which the paper reports
     takes ~40 minutes/decision at depth 3). Maintains a small pool of
     candidate teams, estimates a Q-value for each from a one-gameweek
     rollout, and updates the pool via exponential smoothing each round,
     keeping only the strongest few candidates. This is a bounded, 1-step
     lookahead approximation of the paper's idea, not the full multi-step
     Bellman backup - see "Simplifications" below.

## Running a backtest

```bash
source .venv/bin/activate  # repo root venv - pandas/numpy/scipy/pulp/scikit-learn already there
python -m bayesian_manager.backtest --start-gw 77 --end-gw 107 --manager myopic
python -m bayesian_manager.backtest --start-gw 77 --end-gw 107 --manager qlearning
```

Same walk-forward discipline as `fpl/model/predict.py`: at gameweek `gw`,
beliefs are fit only on `GW_global < gw`, a squad/lineup/captain is chosen,
and it's scored against `gw`'s actual recorded `total_points` - no leakage.
`GW77-107` is the 2024-25 season's GW1-31, the same window
`CLAUDE.md`/`fpl/model/train.py` use to compare the production pipeline
against the old LSTM system, chosen here so the two approaches' backtest
scores are at least comparable in principle (see "Comparing to the
production pipeline" below for the caveats on how comparable).

Results are written to `bayesian_manager/backtest_outputs/backtest_<manager>_GW<start>-<end>.csv`
(one row per gameweek: actual points scored, transfers made, solve time)
and a total/average summary is printed, analogous to `fpl/milp/optimize.py`'s
`actual_total_points` sum.

Useful flags:
- `--n-samples`: Monte Carlo samples per player for the myopic manager
  (higher = less noisy expected-points estimate, slower).
- `--refit-every`: how often (in gameweeks) to refit the club-strength model
  and exit-minute distributions - these change slowly, so refitting every
  gameweek is wasted work; refitting every 4-5 GWs is a good speed/accuracy
  tradeoff.
- `--seed`: RNG seed for reproducibility.

## Backtest result

Ran `python -m bayesian_manager.backtest --start-gw 77 --end-gw 107 --manager myopic --n-samples 30 --refit-every 4`
(2024-25 season, GW1-31, same window as the production pipeline's reference
backtest documented in `CLAUDE.md`):

**Myopic manager: total actual points 1083 over 31 gameweeks (average 34.9/GW).**
(`bayesian_manager/backtest_outputs/backtest_myopic_GW77-107.csv`.)

Also ran the Q-learning/candidate-pool manager over the same window
(`--manager qlearning`, default pool size 3):

**Q-learning manager: total actual points 847 over 31 gameweeks (average 27.3/GW).**
(`bayesian_manager/backtest_outputs/backtest_qlearning_GW77-107.csv`.)

Both CSVs are gitignored as generated output - rerun the commands above to
reproduce; results vary slightly run to run since team selection is
Monte-Carlo-sampled (use `--seed` for reproducibility).

Notably the Q-learning manager made **zero transfers across all 31
gameweeks** in this run - its candidate pool kept re-selecting the same
initial squad because a brand-new candidate team's rollout Q-value rarely
beat an already-pooled team's smoothed Q-value by enough to displace it
(see `manager.py::q_learning_decision`'s exponential-smoothing update). This
is a real, if unflattering, finding rather than a bug: the smoothing rate
(`config.Q_LEARNING_SMOOTHING`) and pool size (`config.Q_LEARNING_POOL_SIZE`)
are exactly the kind of hyperparameters the paper would tune per-league; the
defaults used here favour stability over responsiveness. A useful follow-up
would be sweeping `Q_LEARNING_SMOOTHING` upward (faster to react to new
information) or seeding the pool with the myopic manager's fresh solve every
few gameweeks to force some transfer activity.

For reference, the production pipeline (LightGBM/ensemble + MILP) scores
1526 (old LSTM), 1811 (new single-model), 1900 (new 6-model ensemble) actual
points over the identical GW77-107 window. Both manager variants here score
meaningfully lower - expected given the simplifications below (no bonus
points modelled at all, which are a substantial chunk of `total_points` in
practice; single-gameweek horizon with no chip logic vs. the production
MILP's multi-week rolling horizon with wildcards/chips). See "Comparing to
the production pipeline" for why this comparison, while possible, isn't
perfectly apples-to-apples.

## Comparing to the production pipeline

Both approaches can be pointed at the same `GW_global` window and both
report an `actual_total_points`-style sum, so a comparison is *possible*.
It is not perfectly fair, for a few structural reasons worth keeping in
mind:

- The production MILP is a **multi-week rolling horizon** optimizer
  (`--horizon H`, default 3) that can plan transfers ahead of a good
  fixture run and use wildcards/chips. This module's MKP (`team_select.py`)
  is a **single-gameweek** knapsack, matching the paper's per-decision-point
  formulation - it has no chip logic (wildcard/free hit/bench boost/triple
  captain) and does not look more than one transfer-decision ahead except
  via the (still 1-step) Q-learning manager.
- The production pipeline's point predictions come from models trained to
  minimize MAE against `total_points` directly (including bonus points,
  cards, saves, etc. - anything in the training data). This module's
  simulated points (`simulate.py`) only model appearance points, goals,
  assists, and clean sheets - see below.
- The production pipeline's backtest and this one both avoid leakage
  (walk-forward, decide only from strictly-prior data), so at least that
  part of the comparison is apples-to-apples.

## Simplifications versus the paper (and why)

Being upfront about these rather than silently deviating:

1. **Absence handling** (`beliefs.py::update_beliefs_from_gameweek`,
   `config.ABSENCE_PROXY_MAX_MINUTES`). The paper hand-encoded ~1000
   player absences (injury/suspension/loan/etc.) from media reports for
   their 2009-11 seasons and suppressed belief updates on those gameweeks.
   No equivalent labelled absence data exists for this dataset, so as
   instructed, `minutes == 0` is used as a proxy for "this player didn't
   really have a chance to play" and the row is skipped entirely (no
   `rho`/`omega`/`psi` update either way). This is an approximation in both
   directions: an unused-but-fully-fit bench player (who the paper *would*
   count as evidence of "low Pr(start)") gets skipped here, and conversely a
   truly injured player who briefly comes off the bench for a token
   appearance wouldn't be caught by this proxy. In practice this mostly
   affects fringe squad players rather than the core selectable pool.
2. **Club-level scoreline model** (`club_model.py`). The paper uses Dixon &
   Robinson's (1998) continuous-time birth process (goal-scoring modeled as
   a Poisson process whose intensity is a time-varying function of
   attack/defence strength, estimated via partial-likelihood/Cox-style
   methods). This module instead fits the discrete-time, static
   attack/defence/home-advantage Poisson model usually called the
   "Dixon-Coles" formulation, via iterative proportional fitting. It keeps
   the same structural idea (goals arise from an attack-strength x
   defence-strength product, home advantage factored out separately)
   without the continuous-time birth-process machinery, which would need
   match-minute-level event data and a much more involved partial-likelihood
   estimator than is practical here.
3. **Goal-time / on-pitch-at-the-time attribution** (`simulate.py::simulate_gameweek`).
   The paper samples each starter's exit minute from `S_pos` and simulates
   goals arising at specific minutes within the match, so a goal can only be
   attributed to whoever was actually on the pitch *at that instant*. This
   implementation samples exit minutes and substitution timing (so total
   minutes played per player are simulated realistically), but for
   goal/assist attribution it takes a simpler shortcut: any squad player who
   featured for any positive number of minutes in the match is treated as
   equally eligible (weighted by `omega_p`/`psi_p`) for *any* of the match's
   goals, rather than reconstructing which specific players were on the
   pitch at each goal's (also simulated) minute. This slightly overstates a
   late substitute's chance of getting credit for an early goal and vice
   versa, but avoids a full minute-by-minute event timeline.
4. **`phi` (goal/assist split)**: recomputed empirically from
   `Datasett/master_dataset.csv` (`beliefs.py::compute_phi`) rather than
   hardcoding the paper's reported 0.866 (different dataset/era, as
   instructed). Comes out to roughly 0.90 on this data - see the printed
   value in a backtest run.
5. **Scoring rules simulated**: appearance points (1 for <60 mins, 2 for
   60+), goals (position-dependent: GK/DEF 6, MID 5, FWD 4 in this dataset's
   era... see `simulate.py::GOAL_POINTS` for the exact table used), assists
   (+3 flat), and clean sheets (GK/DEF +4, MID +1, FWD 0, awarded to anyone
   who played 60+ minutes in a match their club kept a clean sheet). *Not*
   modelled: bonus points (BPS), yellow/red cards, own goals, penalty
   misses/saves, saves (for goalkeepers). The paper doesn't give a
   generative sub-model for bonus points either (BPS is itself a somewhat
   opaque, multi-factor in-house Opta/FPL formula) - modelling it well would
   need its own belief variable and is out of scope here.
6. **Lookahead depth**: `q_learning_decision` is a 1-step-lookahead
   approximation of the paper's Q-learning idea (candidate pool +
   exponential smoothing), not a full multi-gameweek recursive Bellman
   backup. The paper itself found the recursive DFS backup (their other
   proposed method) far more expensive (~40 min/decision at depth 3) for
   only a modest reported gain over the candidate-pool approach, which is
   why this implementation prioritizes the candidate-pool version and does
   not also implement full DFS.
7. **No chip logic** (wildcard, free hit, bench boost, triple captain) and
   **no multi-week rolling horizon** - see "Comparing to the production
   pipeline" above. The paper's own formulation is also fundamentally a
   single-gameweek-at-a-time decision (with the lookahead applying to the
   *value estimate*, not to solving multiple future gameweeks' knapsacks
   jointly), so this isn't a shortfall versus the paper, just versus the
   production pipeline's more elaborate rolling-horizon MILP.
8. **Rebuilding beliefs from scratch each gameweek** (`backtest.py`): for
   simplicity, the walk-forward backtest replays every prior gameweek's
   belief update at every step rather than carrying incremental state
   forward. This is O(gw) per step rather than O(1), but is fast enough for
   a single-season backtest (a few seconds per gameweek) - see the module
   docstring in `backtest.py` for the drop-in incremental optimisation if
   this is ever run over multiple seasons at once.

## Files

- `config.py` - paths, FPL squad-constraint constants, belief-prior
  hyperparameters.
- `club_model.py` - simplified Dixon-Robinson/Dixon-Coles club
  attack/defence/home-advantage Poisson model.
- `beliefs.py` - `PlayerBelief` dataclass (rho/omega/psi), Bayesian update
  from a gameweek's results, empirical `phi` and exit-minute-distribution
  fitting.
- `simulate.py` - Monte Carlo outcome sampling (one simulated gameweek given
  beliefs) and expected-points estimation (averaging many samples).
- `team_select.py` - the MKP squad/lineup/captain solver (PuLP+CBC) and
  multi-`n_s` candidate-team generation.
- `manager.py` - `myopic_decision` (M1/M2 baseline) and
  `q_learning_decision` (candidate-pool lookahead) manager variants.
- `backtest.py` - walk-forward backtest CLI, scoring decisions against
  actual historical results.
