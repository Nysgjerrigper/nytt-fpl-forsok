"""Outcome sampling: draw one plausible realisation of a gameweek's results
from current beliefs, and estimate each player's expected FPL points by
averaging many such samples (Sec 3.3-3.4 of the paper).

This is the piece that turns "a player has some probability of starting and
some probability of scoring" into "a concrete final score for a squad",
which is what the MKP in team_select.py needs as its objective coefficients.

FPL scoring approximated here (points-per-position rules current in the
dataset's era): appearance points, goals (differ by position), assists (+3
all positions), clean sheets (GK/DEF only, approximated below), and a
minutes-based involvement point. Bonus points (BPS-derived) and cards/saves
are NOT modelled - see README.md for why (BPS in particular has no clean
generative model in the paper and would need its own sub-model).
"""
import numpy as np

from bayesian_manager import club_model


GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
APPEARANCE_POINTS_SHORT = 1  # 1-59 minutes
APPEARANCE_POINTS_LONG = 2  # 60+ minutes


def _appearance_points(minutes_played):
    if minutes_played <= 0:
        return 0
    if minutes_played < 60:
        return APPEARANCE_POINTS_SHORT
    return APPEARANCE_POINTS_LONG


def _select_starters(squad_ids, beliefs, n_needed, rng):
    """Pick n_needed starters from squad_ids, sampled without replacement
    proportional to each player's Pr(start) - the paper's "select 11 starters
    per side proportional to Pr(start)" rule. Falls back to uniform weights
    if all probabilities are ~0 (e.g. a brand new squad with no data yet)."""
    ids = list(squad_ids)
    if len(ids) <= n_needed:
        return ids
    weights = np.array([max(beliefs[i].pr_start(), 1e-6) for i in ids])
    weights = weights / weights.sum()
    chosen = rng.choice(ids, size=n_needed, replace=False, p=weights)
    return list(chosen)


def _select_bench(squad_ids, starters, beliefs):
    return [i for i in squad_ids if i not in starters]


def simulate_gameweek(
    squad_ids, beliefs, fixtures, strengths, rng, phi,
):
    """Draw one sampled outcome for `squad_ids` (an iterable of player_ids)
    given the current beliefs and club strengths, for one gameweek's fixtures.

    `fixtures` is a DataFrame with columns [team, opponent_team, was_home]
    (one row per club playing that gameweek - clubs on a bye/blank week are
    simply absent). Returns a dict {player_id: points_this_sample}.

    Per-match simulation loop (paper Sec 3.3):
      1. Determine each side's starting XI for the match from squad_ids that
         belong to that club (this function is called once per manager
         squad, so only players in `squad_ids` are tracked - opponents'
         players are not part of the manager's squad and don't need scoring).
      2. Sample each starter's exit minute from S_pos; sample replacements
         from the bench proportional to Pr(sub) when a starter's minute comes up.
      3. Sample the match's goal count for the player's club from the
         Dixon-Robinson-style club model; assign each goal to whichever
         player was on the pitch at a uniformly random minute, proportional
         to Pr(omega=1); assign an assist (with probability phi) to another
         on-pitch player proportional to Pr(psi=1).
    """
    points = {pid: 0 for pid in squad_ids}
    minutes_played = {pid: 0 for pid in squad_ids}

    squad_by_team = {}
    for pid in squad_ids:
        squad_by_team.setdefault(beliefs[pid].team, []).append(pid)

    for _, fx in fixtures.iterrows():
        team = fx["team"]
        team_players = squad_by_team.get(team)
        if not team_players:
            continue  # no squad players at this club - nothing to simulate

        gk_ids = [i for i in team_players if beliefs[i].position == "GK"]
        outfield_ids = [i for i in team_players if beliefs[i].position != "GK"]

        starters = []
        starters += _select_starters(gk_ids, beliefs, 1, rng) if gk_ids else []
        n_outfield_starters = min(len(outfield_ids), 10)
        starters += _select_starters(outfield_ids, beliefs, n_outfield_starters, rng)
        bench = _select_bench(team_players, starters, beliefs)

        exit_minute = {}
        for pid in starters:
            pos = beliefs[pid].position
            dist = fixtures.attrs.get("exit_minute_dists", {}).get(pos)
            if dist is None:
                exit_minute[pid] = 90
            else:
                exit_minute[pid] = int(rng.choice(np.arange(1, 91), p=dist))

        on_pitch = set(starters)
        for pid in starters:
            minutes_played[pid] = exit_minute[pid]
        available_bench = list(bench)
        # Handle in-match substitutions in order of who leaves earliest.
        for pid in sorted(starters, key=lambda p: exit_minute[p]):
            if exit_minute[pid] >= 90 or not available_bench:
                continue
            weights = np.array([max(beliefs[b].pr_sub(), 1e-6) for b in available_bench])
            weights = weights / weights.sum()
            sub_in = rng.choice(available_bench, p=weights)
            available_bench.remove(sub_in)
            on_pitch.discard(pid)
            on_pitch.add(sub_in)
            minutes_played[sub_in] = max(90 - exit_minute[pid], 0)

        n_goals = club_model.sample_team_goals(strengths, team, fx["opponent_team"], fx["was_home"], rng)
        # A crude but serviceable proxy for "who was on the pitch when each
        # goal happened": since we don't simulate goal timing, treat every
        # tracked squad player who featured at all in the match (starters +
        # anyone subbed on) as eligible for any of the match's goals. This
        # slightly overstates late-sub involvement in early goals and vice
        # versa, but avoids needing a full timeline reconstruction - flagged
        # in README.md as a simplification of the paper's minute-exact model.
        featured = [pid for pid in team_players if minutes_played.get(pid, 0) > 0]
        for _ in range(n_goals):
            if not featured:
                break
            goal_weights = np.array([max(beliefs[p].pr_goal(), 1e-6) for p in featured])
            goal_weights = goal_weights / goal_weights.sum()
            scorer = rng.choice(featured, p=goal_weights)
            points[scorer] = points.get(scorer, 0) + GOAL_POINTS.get(beliefs[scorer].position, 4)

            if rng.random() < phi and len(featured) > 1:
                assist_candidates = [p for p in featured if p != scorer]
                assist_weights = np.array([max(beliefs[p].pr_assist(), 1e-6) for p in assist_candidates])
                assist_weights = assist_weights / assist_weights.sum()
                assister = rng.choice(assist_candidates, p=assist_weights)
                points[assister] = points.get(assister, 0) + ASSIST_POINTS

        opp_goals = club_model.sample_team_goals(strengths, fx["opponent_team"], team, not fx["was_home"], rng)
        clean_sheet = opp_goals == 0
        for pid in team_players:
            mins = minutes_played.get(pid, 0)
            points[pid] = points.get(pid, 0) + _appearance_points(mins)
            if clean_sheet and mins >= 60:
                points[pid] += CLEAN_SHEET_POINTS.get(beliefs[pid].position, 0)

    return points


def expected_points(
    player_ids, beliefs, fixtures, strengths, exit_minute_dists, phi, n_samples, rng,
):
    """Monte-Carlo estimate of each player's expected gameweek points: the
    mean over `n_samples` calls to simulate_gameweek. Treats the whole
    candidate player pool as one "squad" for sampling purposes (starters are
    chosen per club, so this correctly handles multiple candidate players at
    the same club) - team_select.py then decides which subset of
    `player_ids` to actually keep under the MKP constraints.
    """
    fixtures = fixtures.copy()
    fixtures.attrs["exit_minute_dists"] = exit_minute_dists
    totals = {pid: 0.0 for pid in player_ids}
    for _ in range(n_samples):
        sample_points = simulate_gameweek(player_ids, beliefs, fixtures, strengths, rng, phi)
        for pid, pts in sample_points.items():
            totals[pid] += pts
    return {pid: totals[pid] / n_samples for pid in player_ids}
