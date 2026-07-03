"""Action generation via multi-dimensional knapsack (MKP), Sec 3.4 of the
paper: given each player's expected points (from simulate.expected_points),
pick the 15-player squad + 11-player lineup + captain that maximises total
expected points subject to FPL's constraints, optionally continuing from an
existing squad with a transfer-penalty term.

The paper solves this with CPLEX; we use PuLP+CBC (already a project
dependency, and what fpl/milp/optimize.py uses) since this is a linear
integer program either way - the solver choice doesn't change the model.
This is a *single-gameweek* knapsack (no rolling horizon), matching the
paper's per-decision-point MKP; fpl/milp/optimize.py's multi-week rolling
horizon is a separate, more elaborate design specific to the production
pipeline and is not replicated here.
"""
import pulp

from bayesian_manager import config


def select_team(
    expected_pts, player_info, previous_squad=None, previous_budget=None, free_transfers=1,
):
    """Solve one gameweek's MKP.

    `expected_pts`: {player_id: expected points this GW}.
    `player_info`: DataFrame with player_id, position, team, value - the
      selectable pool P_i for this gameweek.
    `previous_squad`: set of player_ids currently held, or None to build a
      fresh squad from scratch with the full budget (as in fpl/milp/optimize.py's
      "fresh-build" mode, used for the very first gameweek of a backtest).
    `previous_budget`: bank balance in 0.1m units when continuing a squad.
    `free_transfers`: number of transfers allowed this GW before the -4/transfer
      penalty kicks in (paper's R(o, a_prev, a) transfer-cost term).

    Returns dict: squad (list of ids), lineup (list of ids), captain (id),
    transfers_in, transfers_out, budget_remaining, penalised_transfers.
    """
    ids = list(player_info["player_id"])
    pos = dict(zip(player_info["player_id"], player_info["position"]))
    team = dict(zip(player_info["player_id"], player_info["team"]))
    value = dict(zip(player_info["player_id"], player_info["value"]))
    pts = {i: expected_pts.get(i, 0.0) for i in ids}

    prob = pulp.LpProblem("MKP_TeamSelection", pulp.LpMaximize)

    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    lineup = pulp.LpVariable.dicts("lineup", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")

    fresh_start = previous_squad is None
    # Players held going into this GW who have no row at all this GW (e.g. a
    # blank gameweek, or they left the selectable pool entirely) can't be
    # tracked by a transfer_out[i] variable since there's no i in `ids` for
    # them - the manager didn't choose to sell them, the pool forced it. Treat
    # these as free, unavoidable sales that don't count against the paid-
    # transfer budget, rather than silently dropping their sale value.
    dropped_from_pool = set()
    if not fresh_start:
        dropped_from_pool = {i for i in previous_squad if i not in ids}

    if not fresh_start:
        transfer_in = pulp.LpVariable.dicts("transfer_in", ids, cat="Binary")
        transfer_out = pulp.LpVariable.dicts("transfer_out", ids, cat="Binary")
        # Dummy negative-weight "extra transfer" items (Sec 3.4): each unit
        # bought beyond free_transfers costs config.TRANSFER_PENALTY points,
        # modelled as a linear penalty on the count of transfers in. Forced
        # sales of dropped_from_pool players are excluded from this count.
        n_transfers = pulp.lpSum(transfer_in[i] for i in ids) - len(dropped_from_pool)
        paid_transfers = pulp.LpVariable("paid_transfers", lowBound=0, cat="Integer")
        prob += paid_transfers >= n_transfers - free_transfers
    else:
        n_transfers = 0
        paid_transfers = 0

    # Objective: lineup points + captain's points doubled (captain must be in lineup)
    # minus transfer penalty.
    objective = pulp.lpSum(pts[i] * lineup[i] for i in ids) + pulp.lpSum(pts[i] * captain[i] for i in ids)
    if not fresh_start:
        objective = objective - config.TRANSFER_PENALTY * paid_transfers
    prob += objective

    # --- Squad composition ---
    by_pos = {p: [i for i in ids if pos[i] == p] for p in config.ONFIELD_POSITIONS}
    prob += pulp.lpSum(squad[i] for i in by_pos["GK"]) == config.SQUAD_GK
    prob += pulp.lpSum(squad[i] for i in by_pos["DEF"]) == config.SQUAD_DEF
    prob += pulp.lpSum(squad[i] for i in by_pos["MID"]) == config.SQUAD_MID
    prob += pulp.lpSum(squad[i] for i in by_pos["FWD"]) == config.SQUAD_FWD

    by_team = {}
    for i in ids:
        by_team.setdefault(team[i], []).append(i)
    for t, members in by_team.items():
        prob += pulp.lpSum(squad[i] for i in members) <= config.MAX_PER_CLUB

    if fresh_start:
        prob += pulp.lpSum(value[i] * squad[i] for i in ids) <= config.SQUAD_BUDGET
    else:
        for i in ids:
            was_held = 1 if i in previous_squad else 0
            prob += squad[i] == was_held - transfer_out[i] + transfer_in[i]
            prob += transfer_in[i] + transfer_out[i] <= 1
            if was_held == 0:
                prob += transfer_out[i] == 0
        # Every squad slot vacated by a chosen transfer_out or a forced
        # drop-from-pool must be filled by exactly one transfer_in, so squad
        # size stays constant at config.SQUAD_SIZE.
        prob += pulp.lpSum(transfer_out[i] for i in ids) + len(dropped_from_pool) == pulp.lpSum(transfer_in[i] for i in ids)
        budget = previous_budget if previous_budget is not None else config.SQUAD_BUDGET
        # Dropped players' sale value can't be recovered (we have no current
        # price for a player outside the pool) - conservatively assume 0
        # resale value for them rather than overstating available budget.
        sales = pulp.lpSum(value[i] * transfer_out[i] for i in ids)
        purchases = pulp.lpSum(value[i] * transfer_in[i] for i in ids)
        prob += purchases <= budget + sales

    # --- Lineup composition (subset of squad) ---
    for i in ids:
        prob += lineup[i] <= squad[i]
    prob += pulp.lpSum(lineup[i] for i in ids) == config.LINEUP_SIZE
    prob += pulp.lpSum(lineup[i] for i in by_pos["GK"]) == config.LINEUP_GK
    prob += pulp.lpSum(lineup[i] for i in by_pos["DEF"]) >= config.MIN_LINEUP_DEF
    prob += pulp.lpSum(lineup[i] for i in by_pos["MID"]) >= config.MIN_LINEUP_MID
    prob += pulp.lpSum(lineup[i] for i in by_pos["FWD"]) >= config.MIN_LINEUP_FWD

    # --- Captaincy: exactly one captain, must be in lineup ---
    prob += pulp.lpSum(captain[i] for i in ids) == 1
    for i in ids:
        prob += captain[i] <= lineup[i]

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    def is_on(var):
        v = var.varValue
        return v is not None and v > 0.9

    squad_ids = [i for i in ids if is_on(squad[i])]
    lineup_ids = [i for i in ids if is_on(lineup[i])]
    captain_ids = [i for i in ids if is_on(captain[i])]
    captain_id = captain_ids[0] if captain_ids else None

    if fresh_start:
        transfers_in_ids, transfers_out_ids = [], []
        spent = sum(value[i] for i in squad_ids)
        budget_remaining = config.SQUAD_BUDGET - spent
        paid = 0
    else:
        transfers_in_ids = [i for i in ids if is_on(transfer_in[i])]
        transfers_out_ids = [i for i in ids if is_on(transfer_out[i])]
        sales_val = sum(value[i] for i in transfers_out_ids)
        purchases_val = sum(value[i] for i in transfers_in_ids)
        budget_remaining = (previous_budget or config.SQUAD_BUDGET) + sales_val - purchases_val
        paid = max(0, len(transfers_in_ids) - free_transfers)

    return {
        "status": pulp.LpStatus.get(status, "Unknown"),
        "squad": squad_ids,
        "lineup": lineup_ids,
        "captain": captain_id,
        "transfers_in": transfers_in_ids,
        "transfers_out": transfers_out_ids,
        "budget_remaining": budget_remaining,
        "paid_transfers": paid,
    }


def generate_candidate_teams(
    player_info, beliefs, fixtures, strengths, exit_minute_dists, phi, rng,
    n_samples_grid=None, previous_squad=None, previous_budget=None, free_transfers=1,
):
    """Generate several candidate teams by resolving the MKP at a few
    different n_s (samples-per-player) values, per the paper's finding
    (Sec 4.2) that a handful of *lower* n_s solves gives more useful
    action-space variability than one very-high-n_s solve - the sampling
    noise at low n_s effectively perturbs which "good enough" players get
    picked, which is exactly the diversity a downstream lookahead/Q-learning
    manager (manager.py) needs in its candidate pool.

    Returns a list of (n_samples, expected_pts, team_dict) tuples.
    """
    from bayesian_manager import simulate  # local import: simulate imports club_model, avoid cycle risk

    n_samples_grid = n_samples_grid or config.DEFAULT_N_SAMPLES_GRID
    player_ids = list(player_info["player_id"])
    candidates = []
    for n_samples in n_samples_grid:
        exp_pts = simulate.expected_points(
            player_ids, beliefs, fixtures, strengths, exit_minute_dists, phi, n_samples, rng,
        )
        team = select_team(exp_pts, player_info, previous_squad, previous_budget, free_transfers)
        candidates.append((n_samples, exp_pts, team))
    return candidates
