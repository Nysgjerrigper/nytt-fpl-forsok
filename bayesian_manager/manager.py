"""The two manager variants described in the paper's evaluation (Sec 4):

- `myopic_decision`: the paper's M1/M2 baseline - solve the MKP once using
  current beliefs (a single "best" candidate team per call, no lookahead
  beyond the current gameweek). This is the solid baseline the task asks to
  get right first.

- `q_learning_decision`: the paper's linear-complexity lookahead alternative
  to full depth-limited DFS over the Bellman equation. Full DFS backup
  (V(b) = max_a Q(b,a), Q(b,a) = E_o,tau[r + gamma*V(b')]) is exponential in
  depth and the paper reports ~40 minutes per decision at depth 3 - far too
  slow to backtest over 100+ gameweeks here. Instead we maintain a small
  pool of candidate actions (teams), estimate each one's Q-value from
  Monte-Carlo rollouts of *next* gameweek's simulated points (a 1-step
  lookahead, not full recursion), and update the pool's Q-estimates via
  exponential smoothing as belief-updates arrive - replacing weak pool
  members with freshly generated candidates each round. This is a
  bounded-depth-1 approximation of the paper's "Q-learning-style" manager,
  not the full multi-step Bellman backup; see README.md for the tradeoff.
"""
from bayesian_manager import config, simulate, team_select


def myopic_decision(
    player_info, beliefs, fixtures, strengths, exit_minute_dists, phi, rng,
    previous_squad=None, previous_budget=None, free_transfers=1, n_samples=30,
):
    """Single MKP solve using current beliefs - the M1/M2 baseline manager."""
    player_ids = list(player_info["player_id"])
    exp_pts = simulate.expected_points(
        player_ids, beliefs, fixtures, strengths, exit_minute_dists, phi, n_samples, rng,
    )
    team = team_select.select_team(exp_pts, player_info, previous_squad, previous_budget, free_transfers)
    return team, exp_pts


def _rollout_value(team, expected_pts):
    """Q-value proxy for one candidate team at the current decision point:
    expected lineup + captain points minus the transfer penalty already
    baked into team_select.select_team's paid_transfers count. This *is*
    next-gameweek's expected reward r(o, a_prev, a) from the Bellman
    equation - the "lookahead" comes from discounting it into a running
    Q-estimate across gameweeks (see q_learning_decision), not from
    simulating multiple future gameweeks per candidate (that would need the
    full DFS backup the paper found too slow to run at scale)."""
    lineup_pts = sum(expected_pts.get(i, 0.0) for i in team["lineup"])
    captain_bonus = expected_pts.get(team["captain"], 0.0) if team["captain"] is not None else 0.0
    penalty = config.TRANSFER_PENALTY * team.get("paid_transfers", 0)
    return lineup_pts + captain_bonus - penalty


def q_learning_decision(
    player_info, beliefs, fixtures, strengths, exit_minute_dists, phi, rng,
    q_pool, previous_squad=None, previous_budget=None, free_transfers=1,
    n_samples_grid=None, pool_size=None, discount=None, smoothing=None,
):
    """One gameweek's decision under the candidate-pool Q-learning manager.

    `q_pool` is mutable state carried across gameweeks by the caller (a list
    of dicts: {"team": ..., "q_value": ...}) - pass an empty list on the
    first call. Each call:
      1. generates fresh candidate teams (team_select.generate_candidate_teams,
         varying n_s per the paper's action-diversity finding),
      2. merges them into the existing pool,
      3. updates every pool member's Q-value estimate via exponential
         smoothing toward its rollout value this gameweek (discounted by
         `discount` to weight recent evidence, matching the paper's use of a
         decaying update rather than a plain running mean),
      4. keeps only the top `pool_size` by Q-value,
      5. returns the best-Q team as this gameweek's action.

    Returns (chosen_team, expected_pts_of_chosen, updated q_pool).
    """
    pool_size = pool_size or config.Q_LEARNING_POOL_SIZE
    discount = discount if discount is not None else config.Q_LEARNING_DISCOUNT
    smoothing = smoothing if smoothing is not None else config.Q_LEARNING_SMOOTHING

    candidates = team_select.generate_candidate_teams(
        player_info, beliefs, fixtures, strengths, exit_minute_dists, phi, rng,
        n_samples_grid=n_samples_grid, previous_squad=previous_squad,
        previous_budget=previous_budget, free_transfers=free_transfers,
    )

    pool = list(q_pool)
    for n_samples, exp_pts, team in candidates:
        rollout = _rollout_value(team, exp_pts)
        pool.append({"team": team, "expected_pts": exp_pts, "q_value": rollout})

    # Exponential smoothing: re-evaluate every existing pool member's rollout
    # value against *this* gameweek's fresh simulation before rescoring, so
    # stale teams from several gameweeks ago don't keep artificially high
    # Q-values from beliefs that have since moved on.
    for entry in pool:
        team = entry["team"]
        team_player_ids = [i for i in team["squad"]]
        fresh_pts = simulate.expected_points(
            team_player_ids, beliefs, fixtures, strengths, exit_minute_dists, phi,
            n_samples=10, rng=rng,
        )
        fresh_rollout = _rollout_value(team, fresh_pts)
        entry["q_value"] = (1 - smoothing) * entry["q_value"] + smoothing * discount * fresh_rollout
        entry["expected_pts"] = fresh_pts

    pool.sort(key=lambda e: e["q_value"], reverse=True)
    pool = pool[:pool_size]

    best = pool[0]
    return best["team"], best["expected_pts"], pool
