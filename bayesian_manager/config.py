"""Central configuration for the Bayesian/MKP manager: paths and prior parameters.

Mirrors `fpl/config.py` in spirit (paths derived from ROOT, no hardcoded season
lists) but stays self-contained under `bayesian_manager/` so this experiment
never has to import from, or be imported by, the production `fpl/` pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MASTER_DATASET_PATH = ROOT / "Datasett" / "master_dataset.csv"

OUTPUT_DIR = ROOT / "bayesian_manager" / "backtest_outputs"

GWS_PER_SEASON = 38  # same global GW counter as fpl/config.py::GWS_PER_SEASON

ONFIELD_POSITIONS = ["GK", "DEF", "MID", "FWD"]

# --- FPL squad-composition constants (shared with fpl/milp/optimize.py's
# Kristiansen et al. formulation, duplicated here rather than imported so this
# module has zero coupling to fpl/) ---
SQUAD_GK, SQUAD_DEF, SQUAD_MID, SQUAD_FWD = 2, 5, 5, 3
SQUAD_SIZE = SQUAD_GK + SQUAD_DEF + SQUAD_MID + SQUAD_FWD  # 15
LINEUP_SIZE = 11
LINEUP_GK = 1
MIN_LINEUP_DEF, MIN_LINEUP_MID, MIN_LINEUP_FWD = 3, 2, 1
MAX_PER_CLUB = 3
SQUAD_BUDGET = 1000.0  # FPL's 100.0m budget, in the dataset's 0.1m "value" units
FREE_TRANSFERS_DEFAULT = 1
MAX_FREE_TRANSFERS = 2  # FPL allows banking up to 2 free transfers
TRANSFER_PENALTY = 4  # points deducted per transfer beyond the free allowance

# --- Belief-model priors (Matthews, Ramchurn & Chalkiadakis 2012, Sec. 3.2) ---
# rho_p: Dirichlet prior over {start, sub, unused}. The paper's prior encodes a
# weak belief that a random player is somewhat more likely to be unused than to
# start or be subbed on (most of the ~600+ players in a season barely feature).
RHO_PRIOR = (0.25, 0.25, 0.5)  # (start, sub, unused) pseudo-counts

# omega_p (scores | plays) and psi_p (assists | plays): Beta(a, b) priors.
# Beta(0, 5) is an improper prior (a=0) that the paper uses to mean "starts
# essentially at 0 probability, weak enough that a handful of observed
# goals/assists pulls it up quickly" - we nudge a to a tiny epsilon since a
# literal a=0 makes the Beta density degenerate at 0 before any data arrives,
# which breaks sampling (see beliefs.py::sample_bernoulli).
OMEGA_PRIOR = (1e-3, 5.0)
PSI_PRIOR = (1e-3, 5.0)

# Minute-a-starter-leaves-the-match distribution S_pos: one multinomial per
# position over minute buckets 1..90 (90 = played the full match). Modelled
# empirically per position from historical data (see beliefs.py::fit_exit_minute_distributions);
# this constant only sets the Dirichlet smoothing pseudo-count added to each
# bucket so positions/minutes with few historical observations aren't zeroed out.
EXIT_MINUTE_SMOOTHING = 1.0

# Minutes below this in a historical row are treated as a proxy for "known
# absence" (injury/suspension/loan/etc.) and the row is excluded from belief
# updates, since the paper's hand-curated absence list (from media reports on
# their 2009-11 seasons) isn't available for this dataset. This is a strict
# simplification: minutes==0 also captures unused subs on an available bench,
# who the paper would NOT treat as "absent" - see README.md for discussion.
ABSENCE_PROXY_MAX_MINUTES = 0

# Number of gameweek-outcome samples per candidate action set (n_s in the
# paper). The paper found *lower* n_s with more candidate actions gave better
# action-space coverage than one very-high-n_s team - see team_select.py.
DEFAULT_N_SAMPLES_GRID = [10, 30, 100]

# Q-learning lookahead manager: size of the maintained candidate action pool.
Q_LEARNING_POOL_SIZE = 3
Q_LEARNING_DISCOUNT = 0.9  # gamma
Q_LEARNING_SMOOTHING = 0.3  # exponential-smoothing rate for Q-value updates
