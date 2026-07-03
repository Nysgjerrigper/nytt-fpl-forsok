"""Per-player belief state (rho_p, omega_p, psi_p) plus the shared minute-exit
distributions S_pos and the goal/assist split phi - all as closed-form
conjugate-prior updates, per Matthews, Ramchurn & Chalkiadakis (2012) Sec 3.2.

Why conjugate priors at all: the paper needs to update ~600 players' beliefs
every gameweek and then draw thousands of Monte Carlo samples per candidate
team (see simulate.py). Closed-form Beta/Dirichlet posteriors make both of
those O(1) per player per update/sample - anything needing numerical
posterior inference (e.g. a hierarchical model with shrinkage across
similar players) would be far more faithful to "true" player skill but is
explicitly out of scope for what's a fast, tabular-belief RL loop.
"""
from dataclasses import dataclass, field

import numpy as np

from bayesian_manager import config


@dataclass
class PlayerBelief:
    player_id: int
    name: str = ""
    position: str = ""
    team: str = ""
    value: float = 50.0
    # rho: Dirichlet pseudo-counts over (start, sub, unused)
    rho: np.ndarray = field(default_factory=lambda: np.array(config.RHO_PRIOR, dtype=float))
    # omega/psi: Beta(a, b) pseudo-counts, (successes, failures), conditional on having played
    omega: np.ndarray = field(default_factory=lambda: np.array(config.OMEGA_PRIOR, dtype=float))
    psi: np.ndarray = field(default_factory=lambda: np.array(config.PSI_PRIOR, dtype=float))
    games_observed: int = 0

    def pr_start(self):
        return self.rho[0] / self.rho.sum()

    def pr_sub(self):
        return self.rho[1] / self.rho.sum()

    def pr_unused(self):
        return self.rho[2] / self.rho.sum()

    def pr_plays(self):
        return 1.0 - self.pr_unused()

    def pr_goal(self):
        return self.omega[0] / self.omega.sum()

    def pr_assist(self):
        return self.psi[0] / self.psi.sum()

    def sample_rho(self, rng):
        return rng.dirichlet(self.rho)

    def sample_omega(self, rng):
        return rng.beta(self.omega[0], self.omega[1])

    def sample_psi(self, rng):
        return rng.beta(self.psi[0], self.psi[1])


def _role_for_row(row):
    """Classify a historical row as start / sub / unused. `starts` is a direct
    vaastav/FPL column (1 if in the starting XI); minutes>0 with starts==0
    means used as a substitute; minutes==0 means unused (bench or absent -
    beliefs.py callers are responsible for filtering out the absence proxy
    before calling this, see update_beliefs_from_gameweek)."""
    if row["starts"] >= 1:
        return "start"
    if row["minutes"] > 0:
        return "sub"
    return "unused"


def compute_phi(df):
    """Empirical proportion of goals that come with a recorded assist
    (Sec 3.2's phi, used to weight psi_p's contribution). The paper reports
    0.866 for their 2009-11 dataset; recomputed here since this is a
    different dataset/era, per the task's instruction not to hardcode it."""
    total_goals = df["goals_scored"].sum()
    total_assists = df["assists"].sum()
    if total_goals <= 0:
        return 0.866  # fall back to the paper's figure only if data is degenerate
    return float(min(total_assists / total_goals, 1.0))


def fit_exit_minute_distributions(df):
    """Empirical S_pos: for each position, a distribution over the minute a
    *starting* player was substituted off (or 90 if they played the full
    match). Built from `minutes` among starts==1 rows, bucketed to whole
    minutes and Dirichlet-smoothed so rarely-observed minutes aren't zeroed.

    Returns {position: probs} where probs is a length-90 array (index 0 =
    minute 1, index 89 = minute 90/"played full match").
    """
    dists = {}
    starters = df[df["starts"] >= 1]
    for pos in config.ONFIELD_POSITIONS:
        pos_minutes = starters.loc[starters["position"] == pos, "minutes"].clip(lower=1, upper=90)
        counts = np.full(90, config.EXIT_MINUTE_SMOOTHING)
        if len(pos_minutes):
            idx, c = np.unique(pos_minutes.astype(int).to_numpy(), return_counts=True)
            counts[idx - 1] += c
        dists[pos] = counts / counts.sum()
    return dists


def initial_beliefs(player_info):
    """Build one PlayerBelief per row of `player_info` (player_id, name,
    position, team, value), all starting from the prior - used at the very
    first gameweek of a backtest before any history exists."""
    beliefs = {}
    for _, row in player_info.iterrows():
        beliefs[row["player_id"]] = PlayerBelief(
            player_id=row["player_id"], name=row["name"], position=row["position"],
            team=row["team"], value=row["value"],
        )
    return beliefs


def update_beliefs_from_gameweek(beliefs, gw_df):
    """Sequential Bayesian update: fold one gameweek's actual results into
    each observed player's rho/omega/psi. Rows with minutes <=
    config.ABSENCE_PROXY_MAX_MINUTES are treated as a known-absence proxy and
    skipped entirely (no belief update either way) - this is the
    documented simplification for the paper's hand-curated absence list; see
    README.md. New players not yet tracked get inserted with prior beliefs
    first so every seen player_id ends up with a PlayerBelief.
    """
    for _, row in gw_df.iterrows():
        pid = row["player_id"]
        if pid not in beliefs:
            beliefs[pid] = PlayerBelief(
                player_id=pid, name=row.get("name", ""), position=row.get("position", ""),
                team=row.get("team", ""), value=row.get("value", 50.0),
            )
        belief = beliefs[pid]
        # Keep team/value current even on rows we otherwise skip (transfers, price changes).
        belief.team = row.get("team", belief.team)
        belief.value = row.get("value", belief.value)

        if row["minutes"] <= config.ABSENCE_PROXY_MAX_MINUTES and row["starts"] < 1:
            continue  # absence proxy: suppress rho/omega/psi update this GW

        role = _role_for_row(row)
        rho_update = np.array([role == "start", role == "sub", role == "unused"], dtype=float)
        belief.rho = belief.rho + rho_update
        belief.games_observed += 1

        if role in ("start", "sub"):
            scored = row["goals_scored"] > 0
            assisted = row["assists"] > 0
            belief.omega = belief.omega + np.array([scored, not scored], dtype=float)
            belief.psi = belief.psi + np.array([assisted, not assisted], dtype=float)


def build_player_info(df, as_of_gw):
    """Most-recent known name/position/team/value per player as of (strictly
    before) `as_of_gw`, used to seed/refresh belief entries and to know the
    selectable player pool for a given decision point."""
    hist = df[df["GW_global"] < as_of_gw]
    if hist.empty:
        return df[df["GW_global"] == as_of_gw].drop_duplicates("player_id", keep="first")
    latest = hist.sort_values("GW_global").drop_duplicates("player_id", keep="last")
    return latest[["player_id", "name", "position", "team", "value"]]
