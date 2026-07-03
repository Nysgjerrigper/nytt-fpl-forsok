"""Club-level scoreline model (simplified Dixon & Robinson 1998).

The paper layers player-level scoring/assisting probabilities (omega_p,
psi_p) on top of a club-level model of *how many goals each side scores in a
match* - without that, a "clinical striker" belief has nothing to attach to:
you need Pr(a goal happens for this club in this match) before you can ask
"and is it this player's goal".

Dixon & Robinson's actual model is a continuous-time birth process: each
team's goal-scoring is a Poisson process whose intensity depends on a
time-varying attack strength for the team and defence strength of the
opponent, estimated via partial likelihood (similar machinery to a Cox
proportional-hazards model). Reimplementing the full continuous-time
birth-process/partial-likelihood estimator is a research project on its own,
so this module instead fits the discrete-time equivalent that the football-
analytics literature usually calls the "Dixon-Coles" static version of the
same idea: independent Poisson goal counts per team per match with
multiplicative attack/defence/home-advantage strengths,

    goals_home ~ Poisson(attack_home * defence_away * home_advantage)
    goals_away ~ Poisson(attack_away * defence_home)

fit by (quasi-)Poisson MLE via iterative proportional fitting over all
matches in the training window. This keeps the same *structural* idea (goals
arise from an attack/defence strength product, home advantage separated out)
without the continuous birth-process machinery, and is explicitly flagged as
a simplification in the README rather than silently swapped in.
"""
import numpy as np


def _match_level_goals(df):
    """Collapse player rows into one row per (fixture, team): goals scored by
    that team in that match, plus whether they were home. `df` must already be
    restricted to gameweeks strictly before the one being predicted (no
    leakage) - callers (beliefs.py, backtest.py) are responsible for that
    filtering; this function just aggregates whatever it's given."""
    g = (
        df.groupby(["fixture", "team"], as_index=False)
        .agg(goals=("goals_scored", "sum"), was_home=("was_home", "max"), opponent_team=("opponent_team", "first"))
    )
    return g


def fit_club_strengths(df, n_iter=200, tol=1e-6):
    """Fit attack/defence strength per club + one global home-advantage
    multiplier by iterative proportional fitting (alternating closed-form
    updates), the standard fast way to fit a Dixon-Coles-style bivariate
    Poisson model without a general nonlinear optimizer.

    Returns a dict: {team: {"attack": a, "defence": d}}, plus "_home_advantage".
    Average attack strength is normalised to 1.0 so strengths are relative,
    interpretable multipliers (attack=1.3 -> scores 30% more than an average
    attack against an average defence).
    """
    match_goals = _match_level_goals(df)
    teams = sorted(match_goals["team"].unique())
    if len(teams) < 2 or match_goals.empty:
        # Degenerate (too little history, e.g. very first gameweeks of a
        # season) - fall back to league-average strengths for everyone so
        # downstream code still gets sane, if uninformative, probabilities.
        return {t: {"attack": 1.0, "defence": 1.0} for t in teams} | {"_home_advantage": 1.0}

    attack = {t: 1.0 for t in teams}
    defence = {t: 1.0 for t in teams}
    home_advantage = 1.3  # rough league-wide prior; refined below

    home_rows = match_goals[match_goals["was_home"] == 1]
    away_rows = match_goals[match_goals["was_home"] == 0]
    league_avg_goals = match_goals["goals"].mean() if len(match_goals) else 1.3

    for _ in range(n_iter):
        prev_attack = dict(attack)

        # Attack update: team's attack strength = its average scored goals /
        # (average defence-and-home-adjusted expectation of an average team).
        for t in teams:
            t_home = home_rows[home_rows["team"] == t]
            t_away = away_rows[away_rows["team"] == t]
            numer = t_home["goals"].sum() + t_away["goals"].sum()
            denom = 0.0
            for _, row in t_home.iterrows():
                denom += defence.get(row["opponent_team"], 1.0) * home_advantage
            for _, row in t_away.iterrows():
                denom += defence.get(row["opponent_team"], 1.0)
            attack[t] = numer / denom if denom > 0 else attack[t]

        # Defence update: team's defence "leakiness" (higher = leakier) given
        # opponents' attack strengths and whether the opponent had home advantage.
        for t in teams:
            t_as_away_opponent = home_rows[home_rows["opponent_team"] == t]  # matches where t played away against a home side
            t_as_home_opponent = away_rows[away_rows["opponent_team"] == t]  # matches where t played home against an away side
            numer = t_as_away_opponent["goals"].sum() + t_as_home_opponent["goals"].sum()
            denom = 0.0
            for _, row in t_as_away_opponent.iterrows():
                denom += attack.get(row["team"], 1.0) * home_advantage
            for _, row in t_as_home_opponent.iterrows():
                denom += attack.get(row["team"], 1.0)
            defence[t] = numer / denom if denom > 0 else defence[t]

        # Home-advantage update given current attack/defence estimates.
        expected_home_no_adv = sum(
            attack.get(row["team"], 1.0) * defence.get(row["opponent_team"], 1.0) for _, row in home_rows.iterrows()
        )
        actual_home_goals = home_rows["goals"].sum()
        if expected_home_no_adv > 0:
            home_advantage = actual_home_goals / expected_home_no_adv

        # Renormalise so mean attack strength = 1 (defence/home-advantage
        # absorb the rest) - keeps strengths comparable across refits as the
        # training window grows week over week.
        mean_attack = np.mean(list(attack.values()))
        if mean_attack > 0:
            for t in teams:
                attack[t] /= mean_attack
                defence[t] *= mean_attack

        shift = sum(abs(attack[t] - prev_attack[t]) for t in teams)
        if shift < tol:
            break

    strengths = {t: {"attack": attack[t], "defence": defence[t]} for t in teams}
    strengths["_home_advantage"] = home_advantage
    strengths["_league_avg_goals"] = float(league_avg_goals)
    return strengths


def expected_goals(strengths, team, opponent, was_home):
    """Expected goals for `team` against `opponent` in one match."""
    a = strengths.get(team, {}).get("attack", 1.0)
    d = strengths.get(opponent, {}).get("defence", 1.0)
    lam = a * d
    if was_home:
        lam *= strengths.get("_home_advantage", 1.3)
    return max(lam, 0.05)  # floor to avoid degenerate zero-goal expectations


def sample_team_goals(strengths, team, opponent, was_home, rng):
    """Poisson-sample a goal count for one team in one match."""
    lam = expected_goals(strengths, team, opponent, was_home)
    return int(rng.poisson(lam))
