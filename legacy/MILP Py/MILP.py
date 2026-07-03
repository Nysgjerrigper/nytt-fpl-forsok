import pandas as pd
import pulp
import numpy as np
import time
import sys
import os
import math # Used for rounding FT

# --- Input Parameters ---
START_GAMEWEEK = 80 # Example: Start of the 2nd season in a 3-season file
MAX_GAMEWEEK = 85 # Run for a few GWs
SUB_HORIZON_LENGTH = 2 # Keep it simple for debugging
# *** IMPORTANT: Update this path to your actual file location ***
CSV_FILE_PATH = "C:\\Users\\peram\\Documents\\test\\Datasett\\Validation_Predictions_Clean (2).csv"
# --- SET TIMELIMIT HERE ---
SOLVER_TIME_LIMIT = None  # Seconds (e.g., 15 seconds for testing)
# --- Define solver WITH timeLimit ---
solver_to_use = pulp.PULP_CBC_CMD(
    msg=True,
    timeLimit=SOLVER_TIME_LIMIT, # Pass timeLimit here
)

print(f"--- Running on Python {sys.version} ---")
print(f"--- PuLP Version: {pulp.__version__} ---")
print(f"--- Solver: {solver_to_use.name} ---")
# Note: Time limit is now set *inside* the solver command
print(f"--- Solver Internal Time Limit per Subproblem: {SOLVER_TIME_LIMIT}s ---")

# --- 0. Setup & Data Loading ---
# ... (rest of your setup and data loading code is correct) ...
print("\n--- 0. Setup & Data Loading ---")
try:
    allesesonger = pd.read_csv(CSV_FILE_PATH)
    print(f"CSV '{CSV_FILE_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: CSV file '{CSV_FILE_PATH}' not found.")
    sys.exit() # Use sys.exit() for cleaner exit
except Exception as e:
    print(f"ERROR: Failed to load CSV file. Reason: {e}")
    sys.exit()

print("Initial data shape:", allesesonger.shape)
# print("Initial columns:", allesesonger.columns.tolist()) # Optional

# --- Data Pre-processing and Filtering ---
print("\n--- Pre-processing Data ---")
essential_input_cols = ['player_id', 'GW', 'name', 'position', 'team', 'predicted_total_points', 'value']
# ... (rest of pre-processing is correct, including scaling) ...
# Ensure GW and value are numeric
try:
    allesesonger['GW'] = pd.to_numeric(allesesonger['GW'])
    allesesonger['value'] = pd.to_numeric(allesesonger['value'])
    # Handle potential initial NaNs before scaling
    allesesonger['value'] = allesesonger['value'].fillna(5.0 * 10) # Assume a default value *before* scaling
    # --- SCALING --- Ensure this is intended
    # allesesonger['value'] = allesesonger['value'] * 10 # Scale value (e.g., by 10 if original is 4.5, 10.1 etc.)
    # print("Value scaled by 10") # Comment out if not scaling
except ValueError as e:
    print(f"ERROR: Could not convert 'GW' or 'value' column to numeric: {e}")
    sys.exit()

# Filter Gameweeks
data_load_start_gw = START_GAMEWEEK - 1 if START_GAMEWEEK > 1 else START_GAMEWEEK
data_full_range = allesesonger[(allesesonger['GW'] >= data_load_start_gw) & (allesesonger['GW'] <= MAX_GAMEWEEK)].copy()

if data_full_range.empty:
    print(f"ERROR: No data found for GW range {data_load_start_gw}-{MAX_GAMEWEEK}.")
    sys.exit()
print(f"Filtered data for GW {data_load_start_gw}-{MAX_GAMEWEEK}. Shape: {data_full_range.shape}")

# Clean categorical columns
for col in ['team', 'position', 'name']:
    if col in data_full_range.columns:
        data_full_range[col] = data_full_range[col].astype(str).str.strip()


# --- 1. Data Cleaning & FULL Set Definition ---
print("\n--- 1. Data Cleaning & FULL Set Definition ---")
# Define initial FULL sets
T_setofgameweeks_full = sorted(data_full_range['GW'].unique())
P_setofplayers_initial = sorted(data_full_range['player_id'].unique())
C_setofteams_initial = sorted(data_full_range['team'].dropna().unique())
L_substitution = list(range(1, 4)) # Priorities 1, 2, 3

n_T_full = len(T_setofgameweeks_full)
n_P_initial = len(P_setofplayers_initial)

if n_T_full == 0 or n_P_initial == 0:
    print("ERROR: Initial Gameweek or Player set is empty after filtering.")
    sys.exit()
print(f"Initial FULL sets: {n_T_full} Gameweeks, {n_P_initial} Players, {len(C_setofteams_initial)} Teams.")

# Create all player-GW combinations
player_gw_combos_full = pd.MultiIndex.from_product([P_setofplayers_initial, T_setofgameweeks_full],
                                                     names=['player_id', 'GW'])
data_complete_full = pd.DataFrame(index=player_gw_combos_full).reset_index()

# Merge and fill missing data
print("Merging and filling missing data for full range...")
data_merged_full = pd.merge(data_complete_full, data_full_range, on=['player_id', 'GW'], how='left', suffixes=('', '_discard'))
data_merged_full = data_merged_full[[col for col in data_merged_full.columns if not col.endswith('_discard')]]

data_merged_full.sort_values(by=['player_id', 'GW'], inplace=True)
essential_info_cols = ['name', 'position', 'team']
# Forward fill THEN backfill essential info
for col in essential_info_cols:
     data_merged_full[col] = data_merged_full.groupby('player_id')[col].transform(lambda x: x.ffill().bfill())

data_merged_full['predicted_total_points'] = data_merged_full['predicted_total_points'].fillna(0)
# Fill value based on ffill/bfill - get last known value for a player
data_merged_full['value'] = data_merged_full.groupby('player_id')['value'].transform(lambda x: x.ffill().bfill())
data_merged_full['value'] = data_merged_full['value'].fillna(50.0) # Final fallback

# Filter invalid rows
print("Filtering invalid rows for full range...")
initial_rows_full = data_merged_full.shape[0]
data_filtered_full = data_merged_full.dropna(subset=essential_info_cols).copy()
data_filtered_full['team'] = data_filtered_full['team'].astype(str)
C_setofteams_initial_str = [str(team) for team in C_setofteams_initial]
data_filtered_full = data_filtered_full[data_filtered_full['team'].isin(C_setofteams_initial_str)]

# Remove duplicates
rows_before_dedup_full = data_filtered_full.shape[0]
data_cleaned_full = data_filtered_full.drop_duplicates(subset=['player_id', 'GW'], keep='first').copy()
rows_after_dedup_full = data_cleaned_full.shape[0]
dropped_rows_count = initial_rows_full - rows_after_dedup_full
if dropped_rows_count > 0:
    print(f"Removed {dropped_rows_count} rows during cleaning/filtering (NaNs, duplicates, etc.).")

if data_cleaned_full.empty:
    print("ERROR: No data remaining after cleaning.")
    sys.exit()
print(f"Cleaned full data shape: {data_cleaned_full.shape}")

# --- Define FINAL FULL Sets and Parameters ---
print("Defining final sets and parameters...")
p = sorted(data_cleaned_full['player_id'].unique())
l = L_substitution
# Ensure final team set only includes teams with players in the final player list 'p'
final_player_info = data_cleaned_full[data_cleaned_full['player_id'].isin(p)].drop_duplicates(subset=['player_id'], keep='first')
player_name_map = final_player_info.set_index('player_id')['name'].to_dict() # Player name lookup
pos_map = pd.Series(final_player_info['position'].values, index=final_player_info['player_id'])
C_setofteams_final = sorted(final_player_info['team'].unique()) # Should be strings now

n_P = len(p)
n_C_final = len(C_setofteams_final)
n_L = len(l)
print(f"FINAL FULL sets: {n_P} Players, {n_C_final} Teams.")

# Define Subsets
Pgk = sorted([p_ for p_ in p if pos_map.get(p_) == "GK"])
Pdef = sorted([p_ for p_ in p if pos_map.get(p_) == "DEF"])
Pmid = sorted([p_ for p_ in p if pos_map.get(p_) == "MID"])
Pfwd = sorted([p_ for p_ in p if pos_map.get(p_) == "FWD"])
P_not_gk = sorted([p_ for p_ in p if p_ not in Pgk])
print(f"  Positions: {len(Pgk)} GK, {len(Pdef)} DEF, {len(Pmid)} MID, {len(Pfwd)} FWD")

# Ensure P_c uses the final string team names and filter empty teams
P_c_all = data_cleaned_full[data_cleaned_full['player_id'].isin(p)].groupby('team')['player_id'].unique().apply(list).to_dict()
P_c = {team: sorted(players) for team, players in P_c_all.items() if team in C_setofteams_final and players}
C_setofteams = sorted(list(P_c.keys())) # This is the final list of teams *with players*
n_C = len(C_setofteams)
print(f"  Defined player lists for {n_C} teams actually present in the data.")

# Define GW subsets
if n_T_full > 0:
    medianavgameweeks = np.median(T_setofgameweeks_full)
    median_gw_split_point = math.floor(medianavgameweeks)
    T_FH_overall = [t_ for t_ in T_setofgameweeks_full if t_ <= median_gw_split_point]
    T_SH_overall = [t_ for t_ in T_setofgameweeks_full if t_ > median_gw_split_point]
    print(f"  Overall Gameweek halves: FH <= {median_gw_split_point}, SH > {median_gw_split_point}")
else:
    T_FH_overall, T_SH_overall = [], []
    print("  Warning: Cannot define gameweek halves, T_setofgameweeks_full is empty.")

# Define Parameters
R = 4; MK = 2; MD = 5; MM = 5; MF = 3; MC = 3; E = 11; EK = 1
ED = 3; EM = 2; EF = 1; BS = 1000.0 # Scaled budget
phi = (MK + MD + MM + MF) - E; phi_K = MK - EK
Q_bar = 2; Q_under_bar = 1
epsilon = 0.1 # Vice captain factor
kappa = {1: 0.01, 2: 0.005, 3: 0.001} # Substitution priority weights
M_transfer = MK + MD + MM + MF
M_budget = BS + M_transfer * 200
M_alpha = M_transfer + Q_bar
M_q = Q_bar + 1
epsilon_q = 0.1

print("Parameters defined.")

# Prepare Coefficient Data Structures
print("Preparing full coefficient matrices...")
try:
    points_matrix_df = data_cleaned_full.pivot(index='player_id', columns='GW', values='predicted_total_points')
    value_matrix_df = data_cleaned_full.pivot(index='player_id', columns='GW', values='value')
    points_matrix_df = points_matrix_df.reindex(index=p, columns=T_setofgameweeks_full, fill_value=0.0)
    value_matrix_df = value_matrix_df.reindex(index=p, columns=T_setofgameweeks_full)
    for player_id in p:
        value_matrix_df.loc[player_id] = value_matrix_df.loc[player_id].ffill().bfill()
    value_matrix_df.fillna(50.0, inplace=True)

    print("Full coefficient matrices ready.")
except Exception as e:
    print(f"ERROR creating full pivot tables: {e}")
    sys.exit()

# --- Initialize State Variables ---
master_results = []
previous_squad_dict = {}
previous_budget = BS
previous_ft = 1
used_chips_tracker = {'wc1': False, 'wc2': False, 'bb': False, 'tc': False, 'fh': False}

# --- Rolling Horizon Loop ---
for current_gw in range(START_GAMEWEEK, MAX_GAMEWEEK + 1):
    print(f"\n{'='*15} Solving for Gameweek {current_gw} {'='*15}")
    loop_start_time = time.time()

    # 1. Define Sub-Horizon Gameweeks
    t_sub = sorted([gw for gw in T_setofgameweeks_full if gw >= current_gw and gw < current_gw + SUB_HORIZON_LENGTH])
    if not t_sub:
        print(f"Reached end of defined gameweeks or sub-horizon is empty for GW {current_gw}.")
        break
    n_T_sub = len(t_sub)
    print(f"Sub-horizon GWs: {t_sub}")

    t1_sub = t_sub[0]
    if t1_sub not in points_matrix_df.columns or t1_sub not in value_matrix_df.columns:
        print(f"ERROR: Missing essential data for the starting gameweek {t1_sub}. Stopping.")
        break

    # 2. Create New Model Instance
    model = pulp.LpProblem(f"FPL_Opt_GW{current_gw}_Sub{SUB_HORIZON_LENGTH}", pulp.LpMaximize)

    # 3. Define Variables
    print("Defining variables for sub-horizon...")
    var_start = time.time()
    # ... (Variable definitions x, x_freehit, y, f, h, is_tc, u, e, lambda_var, g, w, b, r, v, q, alpha, ft_carried_over_nonneg are correct) ...
    x = pulp.LpVariable.dicts("Squad", (p, t_sub), cat='Binary')
    x_freehit = pulp.LpVariable.dicts("Squad_FH", (p, t_sub), cat='Binary')
    y = pulp.LpVariable.dicts("Lineup", (p, t_sub), cat='Binary')
    f = pulp.LpVariable.dicts("Captain", (p, t_sub), cat='Binary')
    h = pulp.LpVariable.dicts("ViceCaptain", (p, t_sub), cat='Binary')
    is_tc = pulp.LpVariable.dicts("TripleCaptainChipActive", (p, t_sub), cat='Binary')
    u = pulp.LpVariable.dicts("TransferOut", (p, t_sub), cat='Binary')
    e = pulp.LpVariable.dicts("TransferIn", (p, t_sub), cat='Binary')
    lambda_var = pulp.LpVariable.dicts("Aux_LineupInSquad", (p, t_sub), cat='Binary') # For y linearization
    g = {}
    if P_not_gk and l:
        g = pulp.LpVariable.dicts("Substitution", (P_not_gk, t_sub, l), cat='Binary')
    w = pulp.LpVariable.dicts("WildcardChipActive", t_sub, cat='Binary')
    b = pulp.LpVariable.dicts("BenchBoostChipActive", t_sub, cat='Binary')
    r = pulp.LpVariable.dicts("FreeHitChipActive", t_sub, cat='Binary')
    v = pulp.LpVariable.dicts("RemainingBudget", t_sub, lowBound=0, cat='Continuous')
    q = pulp.LpVariable.dicts("FreeTransfersAvailable", t_sub, lowBound=0, upBound=Q_bar, cat='Integer')
    alpha = pulp.LpVariable.dicts("PenalizedTransfers", t_sub, lowBound=0, upBound=M_alpha, cat='Integer')
    ft_carried_over_nonneg = pulp.LpVariable.dicts("FT_Carry", t_sub, lowBound=0)
    print(f"Variables defined in {time.time() - var_start:.2f}s")

    # 4. Filter Parameters
    points_sub = points_matrix_df.loc[p, t_sub]
    value_sub = value_matrix_df.loc[p, t_sub]

    # 5. Define Objective
    print("Defining objective function...")
    obj_start = time.time()
    # ... (Objective function definition is correct) ...
    points_from_lineup = pulp.lpSum(points_sub.loc[p_, t_] * y[p_][t_] for p_ in p for t_ in t_sub)
    points_from_captain = pulp.lpSum(points_sub.loc[p_, t_] * f[p_][t_] for p_ in p for t_ in t_sub)
    points_from_vice = pulp.lpSum(epsilon * points_sub.loc[p_, t_] * h[p_][t_] for p_ in p for t_ in t_sub)
    points_from_tc = pulp.lpSum(2 * points_sub.loc[p_, t_] * is_tc[p_][t_] for p_ in p for t_ in t_sub)
    points_from_subs = 0
    if g:
        points_from_subs = pulp.lpSum(kappa[l_] * points_sub.loc[p_ngk][t_] * g[p_ngk][t_][l_]
                                      for p_ngk in P_not_gk for t_ in t_sub for l_ in l)
    transfer_penalty = pulp.lpSum(R * alpha[t_] for t_ in t_sub)
    objective = (points_from_lineup + points_from_captain + points_from_vice +
                 points_from_tc + points_from_subs - transfer_penalty)
    model += objective, "Total_Expected_Points_Sub"
    print(f"Objective defined in {time.time() - obj_start:.2f}s")

    # 6. Define Constraints
    print("Adding constraints...")
    cons_start = time.time()
    # ... (All constraint definitions are correct based on previous versions) ...
    # --- Gamechips (Link to overall tracker) ---
    t_sub_fh_pool = [t for t in t_sub if t in T_FH_overall]
    t_sub_sh_pool = [t for t in t_sub if t in T_SH_overall]

    if not used_chips_tracker['wc1'] and t_sub_fh_pool:
        model += pulp.lpSum(w[t_] for t_ in t_sub_fh_pool) <= 1, "WC_Limit_FH_Sub"
    elif t_sub_fh_pool: # If WC1 already used, force w=0 in FH pool
        model += pulp.lpSum(w[t_] for t_ in t_sub_fh_pool) == 0, "WC1_Already_Used"

    if not used_chips_tracker['wc2'] and t_sub_sh_pool:
        model += pulp.lpSum(w[t_] for t_ in t_sub_sh_pool) <= 1, "WC_Limit_SH_Sub"
    elif t_sub_sh_pool: # If WC2 already used, force w=0 in SH pool
        model += pulp.lpSum(w[t_] for t_ in t_sub_sh_pool) == 0, "WC2_Already_Used"

    # TC, BB, FH limits
    if not used_chips_tracker['tc']:
        model += pulp.lpSum(is_tc[p_][t_] for p_ in p for t_ in t_sub) <= 1, "TC_Limit_Sub"
    else:
        model += pulp.lpSum(is_tc[p_][t_] for p_ in p for t_ in t_sub) == 0, "TC_Already_Used"

    if not used_chips_tracker['bb']:
        model += pulp.lpSum(b[t_] for t_ in t_sub) <= 1, "BB_Limit_Sub"
    else:
        model += pulp.lpSum(b[t_] for t_ in t_sub) == 0, "BB_Already_Used"

    if not used_chips_tracker['fh']:
        model += pulp.lpSum(r[t_] for t_ in t_sub) <= 1, "FH_Limit_Sub"
    else:
        model += pulp.lpSum(r[t_] for t_ in t_sub) == 0, "FH_Already_Used"

    # One chip per week
    for t_ in t_sub:
        model += w[t_] + pulp.lpSum(is_tc[p_][t_] for p_ in p) + b[t_] + r[t_] <= 1, f"GC_OneChipPerWeek_{t_}"

    # --- Squad, FH Squad, Lineup, Captain, Subs (Iterate over t_sub) ---
    squad_size_total = MK + MD + MM + MF
    for t_ in t_sub:
        # Regular Squad (4.8 - 4.12)
        if Pgk: model += pulp.lpSum(x[p_gk][t_] for p_gk in Pgk) == MK, f"Squad_GK_{t_}"
        if Pdef: model += pulp.lpSum(x[p_def][t_] for p_def in Pdef) == MD, f"Squad_DEF_{t_}"
        if Pmid: model += pulp.lpSum(x[p_mid][t_] for p_mid in Pmid) == MM, f"Squad_MID_{t_}"
        if Pfwd: model += pulp.lpSum(x[p_fwd][t_] for p_fwd in Pfwd) == MF, f"Squad_FWD_{t_}"
        for c_team in C_setofteams:
            players_in_team = P_c.get(c_team, [])
            if players_in_team:
                 model += pulp.lpSum(x[p_tm][t_] for p_tm in players_in_team) <= MC, f"Squad_TeamLimit_{c_team}_{t_}"

        # Free Hit Squad (4.13 - 4.17, using Big-M based on r[t_])
        if Pgk: model += pulp.lpSum(x_freehit[p_gk][t_] for p_gk in Pgk) <= MK * r[t_], f"FH_Squad_GK_Upper_{t_}"
        if Pgk: model += pulp.lpSum(x_freehit[p_gk][t_] for p_gk in Pgk) >= MK * r[t_] - MK * (1 - r[t_]), f"FH_Squad_GK_Lower_{t_}"
        if Pdef: model += pulp.lpSum(x_freehit[p_def][t_] for p_def in Pdef) <= MD * r[t_], f"FH_Squad_DEF_Upper_{t_}"
        if Pdef: model += pulp.lpSum(x_freehit[p_def][t_] for p_def in Pdef) >= MD * r[t_] - MD * (1 - r[t_]), f"FH_Squad_DEF_Lower_{t_}"
        if Pmid: model += pulp.lpSum(x_freehit[p_mid][t_] for p_mid in Pmid) <= MM * r[t_], f"FH_Squad_MID_Upper_{t_}"
        if Pmid: model += pulp.lpSum(x_freehit[p_mid][t_] for p_mid in Pmid) >= MM * r[t_] - MM * (1 - r[t_]), f"FH_Squad_MID_Lower_{t_}"
        if Pfwd: model += pulp.lpSum(x_freehit[p_fwd][t_] for p_fwd in Pfwd) <= MF * r[t_], f"FH_Squad_FWD_Upper_{t_}"
        if Pfwd: model += pulp.lpSum(x_freehit[p_fwd][t_] for p_fwd in Pfwd) >= MF * r[t_] - MF * (1 - r[t_]), f"FH_Squad_FWD_Lower_{t_}"
        model += pulp.lpSum(x_freehit[p_][t_] for p_ in p) == squad_size_total * r[t_], f"FH_Squad_TotalSize_{t_}"
        for c_team in C_setofteams:
            players_in_team = P_c.get(c_team, [])
            if players_in_team:
                 model += pulp.lpSum(x_freehit[p_tm][t_] for p_tm in players_in_team) <= MC, f"FH_Squad_TeamLimit_{c_team}_{t_}"

        # Starting Line-up (4.18 - 4.26)
        model += pulp.lpSum(y[p_][t_] for p_ in p) == E + phi * b[t_], f"Start_Size_{t_}"
        if Pgk: model += pulp.lpSum(y[p_gk][t_] for p_gk in Pgk) == EK + phi_K * b[t_], f"Start_GK_{t_}"
        if Pdef: model += pulp.lpSum(y[p_def][t_] for p_def in Pdef) >= ED, f"Start_MinDEF_{t_}"
        if Pmid: model += pulp.lpSum(y[p_mid][t_] for p_mid in Pmid) >= EM, f"Start_MinMID_{t_}"
        if Pfwd: model += pulp.lpSum(y[p_fwd][t_] for p_fwd in Pfwd) >= EF, f"Start_MinFWD_{t_}"
        for p_ in p:
            model += y[p_][t_] <= x_freehit[p_][t_] + lambda_var[p_][t_], f"Start_InSquad_LinkA_{p_}_{t_}"
            model += lambda_var[p_][t_] <= x[p_][t_], f"Start_InSquad_LinkB_{p_}_{t_}"
            model += lambda_var[p_][t_] <= 1 - r[t_], f"Start_InSquad_LinkC_{p_}_{t_}"

        # Captain/Vice (4.27 - 4.29)
        model += pulp.lpSum(f[p_][t_] for p_ in p) + pulp.lpSum(is_tc[p_][t_] for p_ in p) == 1, f"Captain_Or_TC_Unique_{t_}"
        model += pulp.lpSum(h[p_][t_] for p_ in p) == 1, f"ViceCaptain_Unique_{t_}"
        for p_ in p:
            model += f[p_][t_] + is_tc[p_][t_] + h[p_][t_] <= y[p_][t_], f"Captaincy_In_Lineup_{p_}_{t_}"
            model += f[p_][t_] + h[p_][t_] <= 1, f"Cap_Not_Vice_{p_}_{t_}"

        # Substitution (4.30 - 4.32)
        if g:
            for p_ngk in P_not_gk:
                is_sub = pulp.lpSum(g[p_ngk][t_][l_] for l_ in l)
                model += y[p_ngk][t_] + is_sub <= x_freehit[p_ngk][t_] + lambda_var[p_ngk][t_], f"Sub_If_Benched_A_{p_ngk}_{t_}"
                model += is_sub <= 1 - y[p_ngk][t_], f"Sub_Only_If_Benched_B_{p_ngk}_{t_}"
            for l_ in l:
                model += pulp.lpSum(g[p_ngk][t_][l_] for p_ngk in P_not_gk) <= 1, f"Sub_Priority_Unique_{t_}_{l_}"

        # Transfer In/Out Limit (4.36)
        for p_ in p:
            model += e[p_][t_] + u[p_][t_] <= 1, f"Transfer_In_Out_Limit_{p_}_{t_}"

    # --- Linking Constraints & Evolution ---
    t1_sub = t_sub[0]
    model += q[t1_sub] == previous_ft, f"FT_Link_{t1_sub}"

    if current_gw == START_GAMEWEEK:
        model += v[t1_sub] + pulp.lpSum(value_sub.loc[p_, t1_sub] * x[p_][t1_sub] for p_ in p) <= BS, f"Budget_Initial_{t1_sub}"
    else:
        sales_value_t1 = pulp.lpSum(value_sub.loc[p_, t1_sub] * u[p_][t1_sub] for p_ in p)
        purchase_cost_t1 = pulp.lpSum(value_sub.loc[p_, t1_sub] * e[p_][t1_sub] for p_ in p)
        model += v[t1_sub] == previous_budget + sales_value_t1 - purchase_cost_t1, f"Budget_Link_{t1_sub}"
        for p_ in p:
            model += x[p_][t1_sub] == previous_squad_dict.get(p_, 0) - u[p_][t1_sub] + e[p_][t1_sub], f"Squad_Link_{p_}_{t1_sub}"

    # Penalized Transfers for t1_sub
    model += alpha[t1_sub] >= pulp.lpSum(e[p_][t1_sub] for p_ in p) - q[t1_sub], f"PenalizedTransfers_Calc_{t1_sub}"
    model += alpha[t1_sub] <= M_alpha * (1 - w[t1_sub]), f"PenalizedTransfers_WC_Override_{t1_sub}"
    model += alpha[t1_sub] <= M_alpha * (1 - r[t1_sub]), f"PenalizedTransfers_FH_Override_{t1_sub}"

    # Evolution for t_ > t1_sub
    for gw_idx in range(n_T_sub - 1):
        t_curr_sub = t_sub[gw_idx+1]
        t_prev_sub = t_sub[gw_idx]
        # ... (Budget, Squad, FH Budget, FH Transfer limits for t_curr_sub are correct) ...
        # Budget Evolution (4.34)
        sales_value = pulp.lpSum(value_sub.loc[p_, t_curr_sub] * u[p_][t_curr_sub] for p_ in p)
        purchase_cost = pulp.lpSum(value_sub.loc[p_, t_curr_sub] * e[p_][t_curr_sub] for p_ in p)
        model += v[t_curr_sub] == v[t_prev_sub] + sales_value - purchase_cost, f"Budget_Evolution_{t_curr_sub}"

        # Squad Continuity (4.35)
        for p_ in p:
            model += x[p_][t_curr_sub] == x[p_][t_prev_sub] - u[p_][t_curr_sub] + e[p_][t_curr_sub], f"Squad_Continuity_{p_}_{t_curr_sub}"

        # Free Hit Budget Limit (4.37)
        cost_fh_squad_t = pulp.lpSum(value_sub.loc[p_, t_curr_sub] * x_freehit[p_][t_curr_sub] for p_ in p)
        value_nonfh_squad_prev = pulp.lpSum(value_sub.loc[p_, t_prev_sub] * x[p_][t_prev_sub] for p_ in p)
        model += cost_fh_squad_t <= v[t_prev_sub] + value_nonfh_squad_prev + M_budget * (1 - r[t_curr_sub]), f"FH_Budget_Limit_Upper_{t_curr_sub}"

        # Free Hit Transfer Restriction (4.38 / 4.39)
        model += pulp.lpSum(u[p_][t_curr_sub] for p_ in p) <= M_transfer * (1 - r[t_curr_sub]), f"FH_NoTransfersOut_{t_curr_sub}"
        model += pulp.lpSum(e[p_][t_curr_sub] for p_ in p) <= M_transfer * (1 - r[t_curr_sub]), f"FH_NoTransfersIn_{t_curr_sub}"

        # Penalized Transfers Calculation (for t_curr_sub)
        model += alpha[t_curr_sub] >= pulp.lpSum(e[p_][t_curr_sub] for p_ in p) - q[t_curr_sub], f"PenalizedTransfers_Calc_{t_curr_sub}"
        model += alpha[t_curr_sub] <= M_alpha * (1 - w[t_curr_sub]), f"PenalizedTransfers_WC_Override_{t_curr_sub}"
        model += alpha[t_curr_sub] <= M_alpha * (1 - r[t_curr_sub]), f"PenalizedTransfers_FH_Override_{t_curr_sub}"

        # Free Transfer Evolution (q[t_curr_sub] based on q[t_prev_sub])
        ft_used_effectively_prev = pulp.lpSum(e[p_][t_prev_sub] for p_ in p) - alpha[t_prev_sub]
        model += ft_carried_over_nonneg[t_prev_sub] >= q[t_prev_sub] - ft_used_effectively_prev, f"FT_Carry_Calc_{t_prev_sub}"
        model += ft_carried_over_nonneg[t_prev_sub] <= Q_bar - Q_under_bar, f"FT_Carry_Cap_{t_prev_sub}" # Added cap based on logic
        chip_active_prev = w[t_prev_sub] + r[t_prev_sub]
        q_normal_calc = ft_carried_over_nonneg[t_prev_sub] + Q_under_bar
        model += q[t_curr_sub] <= Q_under_bar + M_q * (1 - chip_active_prev), f"FT_Evo_Chip_Reset_Upper_{t_curr_sub}"
        model += q[t_curr_sub] >= Q_under_bar - M_q * (1 - chip_active_prev), f"FT_Evo_Chip_Reset_Lower_{t_curr_sub}"
        model += q[t_curr_sub] <= q_normal_calc + M_q * chip_active_prev, f"FT_Evo_Normal_Upper1_{t_curr_sub}"
        model += q[t_curr_sub] <= Q_bar + M_q * chip_active_prev, f"FT_Evo_Normal_Upper2_{t_curr_sub}"
        model += alpha[t_prev_sub] <= M_alpha * (Q_bar - q[t_curr_sub] + epsilon_q), f"Transfer_NoPaidIfMaxFT_{t_prev_sub}"
# Eksplisitt alltid like mange overganger ut og inn
    print("  Adding Explicit Transfer Balance constraint...")
    for t_ in t_sub:
        # Ensure total transfers in equals total transfers out for each week
        model += pulp.lpSum(e[p_][t_] for p_ in p) == pulp.lpSum(u[p_][t_] for p_ in p), f"Transfer_Balance_{t_}"
    print(f"Constraints added in {time.time() - cons_start:.2f}s")

    # 7. Solve Sub-Problem
    print(f"[{time.strftime('%H:%M:%S')}] Starting solve for GW {current_gw} (Solver Internal Limit: {SOLVER_TIME_LIMIT}s)...")
    print(f"[{time.strftime('%H:%M:%S')}] Solver output (msg=True) should appear below:")
    solve_start_time = time.time()

    # --- Direct solve call using the solver defined with timeLimit ---
    try:
        status = model.solve(solver_to_use) # Use the solver defined earlier
    except Exception as solve_exception:
        print(f"\n!!! EXCEPTION during model.solve() for GW {current_gw}: {solve_exception} !!!")
        status = pulp.LpStatusUndefined # Indicate failure

    solve_time = time.time() - solve_start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Solve Complete for GW {current_gw} ---")

    # --- Status checking and result processing ---
    print(f"Solver Status Code: {status}")
    status_str = pulp.LpStatus.get(status, 'Unknown Status')
    print(f"Solver Status: {status_str}")
    print(f"Sub-problem Solve Time: {solve_time:.2f} seconds")

    # --- Check if Time Limit was Actually Hit (by comparing solve_time) ---
    time_limit_hit = SOLVER_TIME_LIMIT is not None and solve_time >= SOLVER_TIME_LIMIT * 0.98 # Use a tolerance

    # --- IMPORTANT: Update solution_acceptable logic ---
    objective_value_extracted = None
    try:
        if model.objective is not None:
             objective_value_extracted = model.objective.value()
    except AttributeError:
        pass

    solution_truly_optimal = (status == pulp.LpStatusOptimal)
    # Accept if Optimal OR if status is NotSolved AND time limit was hit AND an objective value exists
    solution_acceptable_timeout = (
        status == pulp.LpStatusNotSolved and
        time_limit_hit and
        objective_value_extracted is not None
    )
    solution_acceptable = solution_truly_optimal or solution_acceptable_timeout

    use_fallback = False
    if not solution_acceptable:
        print(f"!!! Solver failed to find an acceptable solution OR timed out without solution for GW {current_gw} (Status: {status_str}). Implementing FALLBACK. !!!")
        use_fallback = True
        # Try saving the LP model file
        try:
             lp_filename = f"failed_model_gw{current_gw}.lp"
             model.writeLP(lp_filename)
             print(f"Model written to {lp_filename} for debugging.")
        except Exception as write_err:
             print(f"Could not write LP file: {write_err}")

    # --- Implement decision or fallback ---
    if not use_fallback:
        # --- Extraction Logic (same as before) ---
        t1_sub = t_sub[0]
        try:
            print(f"Extracting results for GW {current_gw} (t={t1_sub})...")
            # ... (rest of extraction code) ...
            gw_results = {'gameweek': current_gw}
            def get_var_val(var, default=0.0):
                try:
                    val = var.varValue
                    return val if val is not None else default
                except AttributeError: return default
            gw_results['squad'] = sorted([p_ for p_ in p if get_var_val(x[p_][t1_sub]) > 0.9])
            gw_results['lineup'] = sorted([p_ for p_ in p if get_var_val(y[p_][t1_sub]) > 0.9])
            captain_id = None; tc_active_gw = False; tc_player_id = None
            potential_captains = [p_ for p_ in p if get_var_val(f[p_][t1_sub]) > 0.9]
            potential_tc = [p_ for p_ in p if get_var_val(is_tc[p_][t1_sub]) > 0.9]
            if potential_tc: captain_id = potential_tc[0]; tc_active_gw = True; tc_player_id = captain_id
            elif potential_captains: captain_id = potential_captains[0]
            gw_results['captain'] = [captain_id] if captain_id is not None else []
            gw_results['vice_captain'] = sorted([p_ for p_ in p if get_var_val(h[p_][t1_sub]) > 0.9])
            gw_results['transfers_in'] = sorted([p_ for p_ in p if get_var_val(e[p_][t1_sub]) > 0.9])
            gw_results['transfers_out'] = sorted([p_ for p_ in p if get_var_val(u[p_][t1_sub]) > 0.9])
            gw_results['budget_end'] = get_var_val(v[t1_sub], default=previous_budget)
            alpha_val = get_var_val(alpha[t1_sub]); gw_results['alpha'] = round(alpha_val)
            gw_results['q_start'] = previous_ft
            gw_results['objective_value'] = objective_value_extracted
            wc_active_gw = get_var_val(w[t1_sub]) > 0.9; bb_active_gw = get_var_val(b[t1_sub]) > 0.9; fh_active_gw = get_var_val(r[t1_sub]) > 0.9
            chip_name_display = None
            if wc_active_gw: chip_name_display = 'WC'
            if bb_active_gw: chip_name_display = 'BB'
            if fh_active_gw: chip_name_display = 'FH'
            if tc_active_gw and tc_player_id is not None: tc_player_name = player_name_map.get(tc_player_id, f"ID:{tc_player_id}"); chip_name_display = f'TC_{tc_player_name}'
            gw_results['chip_played'] = chip_name_display
            master_results.append(gw_results)

            # --- State Update Logic (after successful solve) ---
            # ... (State update logic is correct) ...
            if fh_active_gw:
                print(f"FH played in GW {current_gw}. State for next GW reverts.")
                previous_ft = Q_under_bar
            else:
                 previous_squad_dict = {p_: 1 for p_ in gw_results['squad']}
                 previous_budget = gw_results['budget_end']
                 q_current_val = get_var_val(q[t1_sub], default=previous_ft)
                 alpha_current_val = gw_results['alpha']
                 e_current_sum = len(gw_results['transfers_in'])
                 if wc_active_gw:
                     previous_ft = Q_under_bar
                 else:
                     ft_used_current_eff = max(0, e_current_sum - alpha_current_val)
                     ft_carry_current = max(0, q_current_val - ft_used_current_eff)
                     previous_ft = min(Q_bar, math.floor(ft_carry_current + Q_under_bar))
            print(f"End of GW {current_gw}: Budget={previous_budget:.1f}, Next FT={previous_ft}")
            # ... (chip tracker update logic) ...
            if chip_name_display:
                 if chip_name_display == 'WC':
                     if current_gw in T_FH_overall and not used_chips_tracker['wc1']: used_chips_tracker['wc1'] = True; print("--- WC1 activated ---")
                     elif current_gw in T_SH_overall and not used_chips_tracker['wc2']: used_chips_tracker['wc2'] = True; print("--- WC2 activated ---")
                 elif chip_name_display == 'BB' and not used_chips_tracker['bb']: used_chips_tracker['bb'] = True; print("--- BB activated ---")
                 elif chip_name_display == 'FH' and not used_chips_tracker['fh']: used_chips_tracker['fh'] = True; print("--- FH activated ---")
                 elif chip_name_display.startswith('TC_') and not used_chips_tracker['tc']: used_chips_tracker['tc'] = True; print(f"--- {chip_name_display} activated ---")

        except Exception as extract_err:
            print(f"!!! Error extracting results for GW {current_gw}: {extract_err} !!!")
            import traceback; traceback.print_exc()
            print("Stopping rolling horizon due to extraction error.")
            break

    else: # --- Handle Fallback Case ---
        # ... (Fallback logic is correct) ...
        print(f"Applying Fallback for GW {current_gw}: No transfers made.")
        gw_results = {
            'gameweek': current_gw, 'squad': sorted(list(previous_squad_dict.keys())),
            'lineup': [], 'captain': [], 'vice_captain': [],
            'transfers_in': [], 'transfers_out': [], 'budget_end': previous_budget,
            'alpha': 0, 'q_start': previous_ft, 'objective_value': None,
            'chip_played': 'FALLBACK_NO_TRANSFERS'
        }
        master_results.append(gw_results)
        q_current_val = previous_ft; alpha_current_val = 0; e_current_sum = 0
        ft_used_current_eff = max(0, e_current_sum - alpha_current_val)
        ft_carry_current = max(0, q_current_val - ft_used_current_eff)
        previous_ft = min(Q_bar, math.floor(ft_carry_current + Q_under_bar))
        print(f"End of GW {current_gw} (Fallback): Budget={previous_budget:.1f}, Next FT={previous_ft}")

    # --- Loop continuation logic ---
    print(f"Total time for GW {current_gw} loop: {time.time() - loop_start_time:.2f}s")
    print("-" * 50)

# --- Post-Loop: Process and Display Results ---

print("\n" + "="*20 + " Final Rolling Horizon Results " + "="*20)
if not master_results:
    print("No results were generated.")
else:
    results_df = pd.DataFrame(master_results)
    def map_ids_to_names(id_list):
        if not isinstance(id_list, (list, tuple, set)): return id_list
        if not id_list: return []
        return sorted([player_name_map.get(p_id, f"ID:{p_id}") for p_id in id_list])
    id_list_columns = ['squad', 'lineup', 'captain', 'vice_captain', 'transfers_in', 'transfers_out']
    print("\nConverting player IDs to names in results DataFrame...")
    for col in id_list_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(map_ids_to_names)
    print("ID to Name conversion complete.")
    for index, row in results_df.iterrows():
        gw = row['gameweek']; print(f"\n--- GW {gw} Summary ---")
        print(f"  Chip Played: {row['chip_played'] if row['chip_played'] else 'None'}")
        print(f"  Transfers In ({len(row['transfers_in'])}): {', '.join(row['transfers_in'])}")
        print(f"  Transfers Out ({len(row['transfers_out'])}): {', '.join(row['transfers_out'])}")
        print(f"  FT Available (Start): {row['q_start']}")
        print(f"  Penalized Transfers (Hits): {row['alpha']}")
        cap_name = row['captain'][0] if row['captain'] else 'None'; vice_name = row['vice_captain'][0] if row['vice_captain'] else 'None'
        print(f"  Captain: {cap_name}"); print(f"  Vice-Captain: {vice_name}")
        print(f"  Budget End: {row['budget_end']:.1f}")
        obj_val_str = f"{row['objective_value']:.2f}" if row['objective_value'] is not None else "N/A"; print(f"  Objective Value (Sub-problem): {obj_val_str}")
    try:
        output_csv_name = "rolling_horizon_results_with_names.csv"
        for col in id_list_columns:
             if col in results_df.columns:
                 results_df[col] = results_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
        results_df.to_csv(output_csv_name, index=False); print(f"\nResults with names saved to {output_csv_name}")
    except Exception as save_e: print(f"\nERROR: Could not save results to CSV. Reason: {save_e}")

print("\n--- Script Finished ---")