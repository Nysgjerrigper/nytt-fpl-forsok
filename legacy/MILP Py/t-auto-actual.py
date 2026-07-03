import pandas as pd
import pulp
import numpy as np
import time
import sys
import os
import math # Used for rounding FT
import argparse # Add this import

# Dette skriptet er forskjellig fra t ved at den bruker .bat filen
# Denne filen har ikke valgfritt starter lag
# DVS. den kjøres i en løkke uten å måtte åpne vscode
# Går sekvensielt
# Forskjellen er at denne gir faktisk poeng til laget også.
# Denne bruker actual total points

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description='FPL Optimization with MILP')
parser.add_argument('--horizon', type=int, default=3, help='Sub-horizon length (weeks to look ahead)')
parser.add_argument('--start_gw', type=int, default=78, help='Starting gameweek')
parser.add_argument('--max_gw', type=int, default=78+29, help='Maximum gameweek')
args = parser.parse_args()

# --- Input Parameters ---
CSV_FILE_PATH = "C:/Users/peram/Documents/test/Validation_Predictions_Clean_v2.csv"

START_GAMEWEEK = args.start_gw # Now uses command line argument
MAX_GAMEWEEK = args.max_gw # Now uses command line argument
SUB_HORIZON_LENGTH = args.horizon # Now uses command line argument
# --- SET TIMELIMIT HERE ---
SOLVER_TIME_LIMIT = None  # Seconds (e.g., 600 for 10 mins) or None for no limit


# ---  Gamechip manual usage or None ---- 
SEASON_START_GW = START_GAMEWEEK
TARGET_GW_WC1_OPPORTUNITY = 0 #SEASON_START_GW + 1
TARGET_GW_TC = 0 #None#SEASON_START_GW + 2 
TARGET_GW_FH = 0 #None#SEASON_START_GW + 3 
TARGET_GW_WC2_OPPORTUNITY = 0 #None#SEASON_START_GW + 4 
TARGET_GW_BB = 0 #None#SEASON_START_GW + 5 
print(f"Chip Opportunity GWs (Absolute): WC1={TARGET_GW_WC1_OPPORTUNITY}, TC={TARGET_GW_TC}, FH={TARGET_GW_FH}, WC2={TARGET_GW_WC2_OPPORTUNITY}, BB={TARGET_GW_BB}")


# --- Configure Solver (Default Threads) ---
solver_to_use = pulp.PULP_CBC_CMD(
    msg=True, # Show solver output
    timeLimit=SOLVER_TIME_LIMIT # Pass timeLimit if set
)
print(f"--- Using Solver: {solver_to_use.name} (Default Threads) ---")
if SOLVER_TIME_LIMIT:
    print(f"--- Solver Time Limit per Subproblem: {SOLVER_TIME_LIMIT}s ---")
else:
    print("--- Solver Time Limit per Subproblem: None ---")


# --- Check Versions ---
print(f"--- Running on Python {sys.version} ---")
print(f"--- PuLP Version: {pulp.__version__} ---")

# --- 0. Setup & Data Loading ---
print("\n--- 0. Setup & Data Loading ---")
try:
    allesesonger = pd.read_csv(CSV_FILE_PATH)
    print(f"Raw CSV '{CSV_FILE_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Raw CSV file '{CSV_FILE_PATH}' not found.")
    sys.exit()
except Exception as e:
    print(f"ERROR: Failed to load raw CSV file. Reason: {e}")
    sys.exit()

print("Initial raw data shape:", allesesonger.shape)

# --- Data Pre-processing and Filtering ---
print("\n--- Pre-processing Raw Data ---")
essential_input_cols = ['player_id', 'GW', 'name', 'position', 'team', 'actual_total_points', 'value'] # Adjusted
missing_cols = [col for col in essential_input_cols if col not in allesesonger.columns]
if missing_cols:
    print(f"ERROR: Missing essential columns in the raw CSV: {missing_cols}")
    sys.exit()

# Ensure GW and value are numeric
try:
    allesesonger['GW'] = pd.to_numeric(allesesonger['GW'])
    allesesonger['value'] = pd.to_numeric(allesesonger['value'])
    allesesonger['value'] = allesesonger['value'].fillna(50.0) # Fill NA before potential scaling
    # --- SCALING --- Uncomment if your value needs scaling
    # allesesonger['value'] = allesesonger['value'] * 10
    # print("Value scaled by 10")
except ValueError as e:
    print(f"ERROR: Could not convert 'GW' or 'value' column to numeric: {e}")
    sys.exit()

# Add after line 73 (after filling NA values)
# --- DOUBLE GAMEWEEK HANDLING ---
print("Checking for double gameweeks and resolving...")
try:
    # Find players with multiple appearances in the same gameweek
    player_gw_counts = allesesonger.groupby(['player_id', 'GW']).size().reset_index(name='appearances')
    double_gw_entries = player_gw_counts[player_gw_counts['appearances'] > 1]
    
    if not double_gw_entries.empty:
        print(f"Found {len(double_gw_entries)} player-gameweek combinations with multiple appearances")
        
        # Check for potential inconsistencies in double gameweeks
        inconsistencies = []
        for _, row in double_gw_entries.iterrows():
            player_id, gw = row['player_id'], row['GW']
            player_gw_data = allesesonger[(allesesonger['player_id'] == player_id) & (allesesonger['GW'] == gw)]
            
            # Check if position or name is inconsistent
            if player_gw_data['position'].nunique() > 1:
                inconsistencies.append(f"Player ID {player_id} has different positions in GW {gw}")
            
            # Check if team is different (this could be valid for mid-season transfers)
            if player_gw_data['team'].nunique() > 1:
                teams = ", ".join(player_gw_data['team'].unique())
                player_name = player_gw_data['name'].iloc[0]
                print(f"NOTE: Player {player_name} (ID {player_id}) appears for multiple teams in GW {gw}: {teams}")
        
        if inconsistencies:
            print("WARNING: Found inconsistencies in double gameweek data:")
            for issue in inconsistencies:
                print(f"  - {issue}")
        
        # Perform the aggregation with necessary handling
        print("Aggregating double gameweeks...")
        # Define which columns to sum and which to keep first value
        sum_cols = ['actual_total_points', 'actual_total_points'] if 'actual_total_points' in allesesonger.columns else ['actual_total_points']
        first_cols = ['name', 'position', 'team', 'value']
        
        # Create aggregation dictionary
        agg_dict = {col: 'sum' for col in sum_cols}
        agg_dict.update({col: 'first' for col in first_cols if col in allesesonger.columns})
        
        # Group and aggregate
        allesesonger = allesesonger.groupby(['player_id', 'GW'], as_index=False).agg(agg_dict)
        print(f"Data shape after double gameweek aggregation: {allesesonger.shape}")
    else:
        print("No double gameweeks found in the data")
        
except Exception as e:
    print(f"WARNING: Error during double gameweek processing: {e}")
    print("Continuing with original data without double gameweek aggregation")

# Filter Gameweeks for the entire run horizon
data_load_start_gw = START_GAMEWEEK - 1 if START_GAMEWEEK > 1 else START_GAMEWEEK
data_full_range_raw = allesesonger[(allesesonger['GW'] >= data_load_start_gw) & (allesesonger['GW'] <= MAX_GAMEWEEK)].copy()
if data_full_range_raw.empty:
    print(f"ERROR: No raw data found for GW range {data_load_start_gw}-{MAX_GAMEWEEK}.")
    sys.exit()
print(f"Filtered raw data for GW {data_load_start_gw}-{MAX_GAMEWEEK}. Shape: {data_full_range_raw.shape}")

# Clean categorical columns
for col in ['team', 'position', 'name']:
    if col in data_full_range_raw.columns:
        data_full_range_raw[col] = data_full_range_raw[col].astype(str).str.strip()


# --- 1. Data Cleaning, Aggregation & FULL Set Definition ---
print("\n--- 1. Data Cleaning, Aggregation & FULL Set Definition ---")
# Define initial FULL sets from raw filtered data
T_setofgameweeks_full = sorted(data_full_range_raw['GW'].unique())
P_setofplayers_initial = sorted(data_full_range_raw['player_id'].unique())
C_setofteams_initial = sorted(data_full_range_raw['team'].dropna().unique())

n_T_full = len(T_setofgameweeks_full)
n_P_initial = len(P_setofplayers_initial)
if n_T_full == 0 or n_P_initial == 0: print("ERROR: Initial GW or Player set empty."); sys.exit()
print(f"Initial FULL sets: {n_T_full} GWs, {n_P_initial} Players, {len(C_setofteams_initial)} Teams.")

# Create all player-GW combinations for merging
player_gw_combos_full = pd.MultiIndex.from_product([P_setofplayers_initial, T_setofgameweeks_full],
                                                     names=['player_id', 'GW'])
data_complete_full = pd.DataFrame(index=player_gw_combos_full).reset_index()

# Merge and fill missing data
print("Merging and filling missing data...")
data_merged_full = pd.merge(data_complete_full, data_full_range_raw, on=['player_id', 'GW'], how='left', suffixes=('', '_discard'))
data_merged_full = data_merged_full[[col for col in data_merged_full.columns if not col.endswith('_discard')]]
data_merged_full.sort_values(by=['player_id', 'GW'], inplace=True)
essential_info_cols = ['name', 'position', 'team']
for col in essential_info_cols: data_merged_full[col] = data_merged_full.groupby('player_id')[col].transform(lambda x: x.ffill().bfill())
data_merged_full['actual_total_points'] = data_merged_full['actual_total_points'].fillna(0)
data_merged_full['value'] = data_merged_full.groupby('player_id')['value'].transform(lambda x: x.ffill().bfill())
data_merged_full['value'] = data_merged_full['value'].fillna(50.0)

# --- AGGREGATION FOR DGWs --- # <<<< THIS BLOCK IS KEPT
print("Aggregating data for Double Gameweeks...")
sum_cols = ['actual_total_points', 'minutes', 'goals_scored', 'assists', 'bonus', 'bps', 'saves', 'penalties_saved'] # Add/remove as needed
first_cols = ['value', 'name', 'position', 'team']
group_cols = ['player_id', 'GW']
agg_funcs = {}
for col in sum_cols:
    if col in data_merged_full.columns: agg_funcs[col] = 'sum'
for col in first_cols:
    if col in data_merged_full.columns: agg_funcs[col] = 'first'
data_aggregated = data_merged_full.groupby(group_cols, as_index=False).agg(agg_funcs)
print(f"Data shape after aggregation: {data_aggregated.shape}")

# --- Post-Aggregation Cleaning & Filtering ---
data_cleaned_full = data_aggregated.copy() # <<<< USE AGGREGATED DATA
print("Filtering invalid rows post-aggregation...")
initial_rows_agg = data_cleaned_full.shape[0]
data_cleaned_full = data_cleaned_full.dropna(subset=essential_info_cols).copy()
data_cleaned_full['team'] = data_cleaned_full['team'].astype(str)
C_setofteams_initial_str = [str(team) for team in C_setofteams_initial]
data_cleaned_full = data_cleaned_full[data_cleaned_full['team'].isin(C_setofteams_initial_str)]
rows_after_filter_agg = data_cleaned_full.shape[0]
print(f"Removed {initial_rows_agg - rows_after_filter_agg} rows post-aggregation (NaNs, invalid teams).")
if data_cleaned_full.empty: print("ERROR: No data remaining after aggregation/cleaning."); sys.exit()
print(f"Cleaned aggregated data shape: {data_cleaned_full.shape}")

# --- Define FINAL FULL Sets and Parameters (using data_cleaned_full) ---
print("Defining final sets and parameters...")
# --- Re-check START_GAMEWEEK against the actual cleaned data ---
T_setofgameweeks_full = sorted(data_cleaned_full['GW'].unique()) # Update based on cleaned data
if START_GAMEWEEK not in T_setofgameweeks_full:
     available_gws_after_start = [gw for gw in T_setofgameweeks_full if gw >= START_GAMEWEEK]
     if not available_gws_after_start: print(f"ERROR: START_GAMEWEEK {START_GAMEWEEK} not in cleaned data range ({min(T_setofgameweeks_full)}-{max(T_setofgameweeks_full)})."); sys.exit()
     actual_start_gw = min(available_gws_after_start); print(f"Warning: START_GAMEWEEK {START_GAMEWEEK} adjusted to actual start {actual_start_gw}."); START_GAMEWEEK = actual_start_gw
     T_setofgameweeks_full = sorted([gw for gw in T_setofgameweeks_full if gw >= START_GAMEWEEK])
     if not T_setofgameweeks_full: print("ERROR: No gameweeks left after adjusting START_GAMEWEEK."); sys.exit()

p = sorted(data_cleaned_full['player_id'].unique())
l = list(range(1, 4))
final_player_info = data_cleaned_full.drop_duplicates(subset=['player_id'], keep='first')
player_name_map = final_player_info.set_index('player_id')['name'].to_dict()
pos_map = pd.Series(final_player_info['position'].values, index=final_player_info['player_id'])
C_setofteams_final = sorted(final_player_info['team'].unique())
Pgk = sorted([p_ for p_ in p if pos_map.get(p_) == "GK"])
Pdef = sorted([p_ for p_ in p if pos_map.get(p_) == "DEF"])
Pmid = sorted([p_ for p_ in p if pos_map.get(p_) == "MID"])
Pfwd = sorted([p_ for p_ in p if pos_map.get(p_) == "FWD"])
P_not_gk = sorted([p_ for p_ in p if p_ not in Pgk])
P_c_all = data_cleaned_full[data_cleaned_full['player_id'].isin(p)].groupby('team')['player_id'].unique().apply(list).to_dict()
P_c = {team: sorted(players) for team, players in P_c_all.items() if team in C_setofteams_final and players}
C_setofteams = sorted(list(P_c.keys()))
n_P = len(p); n_C = len(C_setofteams); n_L = len(l)
print(f"Sets defined for run: {len(T_setofgameweeks_full)} GWs ({min(T_setofgameweeks_full)}-{max(T_setofgameweeks_full)}), {n_P} Players, {n_C} Teams.")
print(f"  Positions: {len(Pgk)} GK, {len(Pdef)} DEF, {len(Pmid)} MID, {len(Pfwd)} FWD")

# Define GW subsets for WC logic
season_num = math.ceil(START_GAMEWEEK / 38)
gw1_of_this_season = (season_num - 1) * 38 + 1
mid_season_split_gw = gw1_of_this_season + 19 -1
T_FH_overall = [t_ for t_ in T_setofgameweeks_full if t_ <= mid_season_split_gw]
T_SH_overall = [t_ for t_ in T_setofgameweeks_full if t_ > mid_season_split_gw]
print(f"  Season {season_num} Halves (Absolute GWs): FH <= {mid_season_split_gw}, SH > {mid_season_split_gw}")

# Define Parameters
R_penalty = 4; MK = 2; MD = 5; MM = 5; MF = 3; MC = 3; E = 11; EK = 1
ED = 3; EM = 2; EF = 1; BS = 1000.0 # Scaled budget
phi = (MK + MD + MM + MF) - E; phi_K = MK - EK
Q_bar = 2; Q_under_bar = 1
epsilon = 0.1; kappa = {1: 0.01, 2: 0.005, 3: 0.001}
M_transfer = MK + MD + MM + MF
M_budget = BS + M_transfer * 200
M_alpha = M_transfer + Q_bar
M_q = Q_bar + 1
epsilon_q = 0.1
print("Parameters defined.")

# Prepare Coefficient Data Structures
print("Preparing full coefficient matrices...")
try:
    # Use data_cleaned_full which is now aggregated
    points_matrix_df = data_cleaned_full.pivot(index='player_id', columns='GW', values='actual_total_points')
    value_matrix_df = data_cleaned_full.pivot(index='player_id', columns='GW', values='value')
    # Reindex using the final player set 'p' and the relevant gameweeks 'T_setofgameweeks_full'
    points_matrix_df = points_matrix_df.reindex(index=p, columns=T_setofgameweeks_full, fill_value=0.0)
    value_matrix_df = value_matrix_df.reindex(index=p, columns=T_setofgameweeks_full)
    # Forward/backward fill values within the reindexed matrix
    for player_id in p: value_matrix_df.loc[player_id] = value_matrix_df.loc[player_id].ffill().bfill()
    value_matrix_df.fillna(50.0, inplace=True) # Final fallback
    print("Full coefficient matrices ready.")
except Exception as e: print(f"ERROR creating full pivot tables: {e}"); sys.exit()

# --- Initialize State Variables ---
master_results = []
previous_squad_dict = {}
previous_budget = BS
previous_ft = 1
used_chips_tracker = {'wc1': False, 'wc2': False, 'bb': False, 'tc': False, 'fh': False}

# --- Rolling Horizon Loop ---
print("\n--- Starting Rolling Horizon ---")
for current_gw in range(START_GAMEWEEK, MAX_GAMEWEEK + 1):
    if current_gw not in T_setofgameweeks_full:
        print(f"Skipping Gameweek {current_gw} as it's not in the loaded data range.")
        continue

    print(f"\n{'='*15} Solving for Gameweek {current_gw} {'='*15}")
    loop_start_time = time.time()

    # 1. Define Sub-Horizon Gameweeks
    t_sub = sorted([gw for gw in T_setofgameweeks_full if gw >= current_gw and gw < current_gw + SUB_HORIZON_LENGTH])
    if not t_sub: print(f"Sub-horizon empty for GW {current_gw}."); break
    n_T_sub = len(t_sub); print(f"Sub-horizon GWs: {t_sub}"); t1_sub = t_sub[0]
    if t1_sub not in points_matrix_df.columns or t1_sub not in value_matrix_df.columns: print(f"ERROR: Missing coefficient data for GW {t1_sub}. Stopping."); break
    if not all(player_id in points_matrix_df.index for player_id in p): missing_p = [pid for pid in p if pid not in points_matrix_df.index]; print(f"ERROR: Players {missing_p} missing. Stopping."); break

    # 2. Create New Model Instance
    model = pulp.LpProblem(f"FPL_Opt_GW{current_gw}_Sub{SUB_HORIZON_LENGTH}", pulp.LpMaximize)

    # 3. Define Variables for Sub-Horizon
    print("Defining variables for sub-horizon...")
    var_start = time.time()
    x = pulp.LpVariable.dicts("Squad", (p, t_sub), cat='Binary')
    x_freehit = pulp.LpVariable.dicts("Squad_FH", (p, t_sub), cat='Binary')
    y = pulp.LpVariable.dicts("Lineup", (p, t_sub), cat='Binary')
    f = pulp.LpVariable.dicts("Captain", (p, t_sub), cat='Binary')
    h = pulp.LpVariable.dicts("ViceCaptain", (p, t_sub), cat='Binary')
    is_tc = pulp.LpVariable.dicts("TripleCaptainChipActive", (p, t_sub), cat='Binary')
    u = pulp.LpVariable.dicts("TransferOut", (p, t_sub), cat='Binary')
    e = pulp.LpVariable.dicts("TransferIn", (p, t_sub), cat='Binary')
    lambda_var = pulp.LpVariable.dicts("Aux_LineupInSquad", (p, t_sub), cat='Binary')
    g = {}
    if P_not_gk and l: g = pulp.LpVariable.dicts("Substitution", (P_not_gk, t_sub, l), cat='Binary')
    w = pulp.LpVariable.dicts("WildcardChipActive", t_sub, cat='Binary')
    b = pulp.LpVariable.dicts("BenchBoostChipActive", t_sub, cat='Binary')
    r = pulp.LpVariable.dicts("FreeHitChipActive", t_sub, cat='Binary')
    v = pulp.LpVariable.dicts("RemainingBudget", t_sub, lowBound=0, cat='Continuous')
    q = pulp.LpVariable.dicts("FreeTransfersAvailable", t_sub, lowBound=0, upBound=Q_bar, cat='Integer')
    alpha = pulp.LpVariable.dicts("PenalizedTransfers", t_sub, lowBound=0, upBound=M_alpha, cat='Integer')
    ft_carried_over_nonneg = pulp.LpVariable.dicts("FT_Carry", t_sub, lowBound=0)
    print(f"Variables defined in {time.time() - var_start:.2f}s")

    # 4. Filter Parameters for Sub-Horizon
    points_sub = points_matrix_df.loc[p, t_sub]
    value_sub = value_matrix_df.loc[p, t_sub]

    # 5. Define Objective for Sub-Horizon
    print("Defining objective function...")
    obj_start = time.time()
    points_from_lineup = pulp.lpSum(points_sub.loc[p_, t_] * y[p_][t_] for p_ in p for t_ in t_sub)
    points_from_captain = pulp.lpSum(points_sub.loc[p_, t_] * f[p_][t_] for p_ in p for t_ in t_sub)
    points_from_vice = pulp.lpSum(epsilon * points_sub.loc[p_, t_] * h[p_][t_] for p_ in p for t_ in t_sub)
    points_from_tc = pulp.lpSum(2 * points_sub.loc[p_, t_] * is_tc[p_][t_] for p_ in p for t_ in t_sub)
    points_from_subs = 0
    if g: points_from_subs = pulp.lpSum(kappa[l_] * points_sub.loc[p_ngk][t_] * g[p_ngk][t_][l_] for p_ngk in P_not_gk for t_ in t_sub for l_ in l)
    transfer_penalty = pulp.lpSum(R_penalty * alpha[t_] for t_ in t_sub)
    objective = (points_from_lineup + points_from_captain + points_from_vice +
                 points_from_tc + points_from_subs - transfer_penalty)
    model += objective, "Total_Expected_Points_Sub"
    print(f"Objective defined in {time.time() - obj_start:.2f}s")

    # 6. Define Constraints for Sub-Horizon
    print("Adding constraints...")
    cons_start = time.time()

    # --- Gamechips (with Timing Restrictions) ---
    print("  Adding Gamechip constraints (with timing)...")
    # <<< This block correctly implements chip timing >>>
    wc1_available = not used_chips_tracker['wc1']
    wc2_available = not used_chips_tracker['wc2']
    tc_available = not used_chips_tracker['tc']
    bb_available = not used_chips_tracker['bb']
    fh_available = not used_chips_tracker['fh']
    for t_ in t_sub:
        wc1_target_is_none = TARGET_GW_WC1_OPPORTUNITY is None; wc2_target_is_none = TARGET_GW_WC2_OPPORTUNITY is None
        tc_target_is_none = TARGET_GW_TC is None; bb_target_is_none = TARGET_GW_BB is None; fh_target_is_none = TARGET_GW_FH is None
        # WC1
        is_in_first_half = t_ <= mid_season_split_gw
        if is_in_first_half:
            if not wc1_available or (not wc1_target_is_none and t_ != TARGET_GW_WC1_OPPORTUNITY): model += w[t_] == 0, f"WC1_NotAvailableOrNotTarget_{t_}"
        else: model += w[t_] == 0, f"WC1_OnlyInFirstHalf_{t_}"
        # WC2
        is_in_second_half = t_ > mid_season_split_gw
        if is_in_second_half:
             if not wc2_available or (not wc2_target_is_none and t_ != TARGET_GW_WC2_OPPORTUNITY): model += w[t_] == 0, f"WC2_NotAvailableOrNotTarget_{t_}"
        else: model += w[t_] == 0, f"WC2_OnlyInSecondHalf_{t_}"
        # TC
        if not tc_available or (not tc_target_is_none and t_ != TARGET_GW_TC): model += pulp.lpSum(is_tc[p_][t_] for p_ in p) == 0, f"TC_NotAvailableOrNotTarget_{t_}"
        else: model += pulp.lpSum(is_tc[p_][t_] for p_ in p) <= 1, f"TC_Possible_{t_}"
        # BB
        if not bb_available or (not bb_target_is_none and t_ != TARGET_GW_BB): model += b[t_] == 0, f"BB_NotAvailableOrNotTarget_{t_}"
        # FH
        if not fh_available or (not fh_target_is_none and t_ != TARGET_GW_FH): model += r[t_] == 0, f"FH_NotAvailableOrNotTarget_{t_}"
        # One Chip Per Week
        model += w[t_] + pulp.lpSum(is_tc[p_][t_] for p_ in p) + b[t_] + r[t_] <= 1, f"GC_OneChipPerWeek_{t_}"
    # Overall Chip Limits
    t_sub_in_fh_overall = [t for t in t_sub if t <= mid_season_split_gw]; t_sub_in_sh_overall = [t for t in t_sub if t > mid_season_split_gw]
    if t_sub_in_fh_overall: model += pulp.lpSum(w[t_] for t_ in t_sub_in_fh_overall) <= (1 if wc1_available else 0), "WC_Limit_FH_Sub"
    if t_sub_in_sh_overall: model += pulp.lpSum(w[t_] for t_ in t_sub_in_sh_overall) <= (1 if wc2_available else 0), "WC_Limit_SH_Sub"


    # --- Squad, FH Squad, Lineup, Captain, Subs (Iterate over t_sub) ---
    print("  Adding Squad, Lineup, Captain, Sub constraints...")
    # ... (These constraints remain the same - Correct) ...
    squad_size_total = MK + MD + MM + MF
    for t_ in t_sub:
        # Regular Squad (4.8 - 4.12)
        if Pgk: model += pulp.lpSum(x[p_gk][t_] for p_gk in Pgk) == MK, f"Squad_GK_{t_}"
        if Pdef: model += pulp.lpSum(x[p_def][t_] for p_def in Pdef) == MD, f"Squad_DEF_{t_}"
        if Pmid: model += pulp.lpSum(x[p_mid][t_] for p_mid in Pmid) == MM, f"Squad_MID_{t_}"
        if Pfwd: model += pulp.lpSum(x[p_fwd][t_] for p_fwd in Pfwd) == MF, f"Squad_FWD_{t_}"
        for c_team in C_setofteams:
            players_in_team = P_c.get(c_team, [])
            if players_in_team: model += pulp.lpSum(x[p_tm][t_] for p_tm in players_in_team) <= MC, f"Squad_TeamLimit_{c_team}_{t_}"
        # Free Hit Squad (4.13 - 4.17)
        if Pgk: model += pulp.lpSum(x_freehit[p_gk][t_] for p_gk in Pgk) == MK * r[t_], f"FH_Squad_GK_{t_}"
        if Pdef: model += pulp.lpSum(x_freehit[p_def][t_] for p_def in Pdef) == MD * r[t_], f"FH_Squad_DEF_{t_}"
        if Pmid: model += pulp.lpSum(x_freehit[p_mid][t_] for p_mid in Pmid) == MM * r[t_], f"FH_Squad_MID_{t_}"
        if Pfwd: model += pulp.lpSum(x_freehit[p_fwd][t_] for p_fwd in Pfwd) == MF * r[t_], f"FH_Squad_FWD_{t_}"
        model += pulp.lpSum(x_freehit[p_][t_] for p_ in p) == squad_size_total * r[t_], f"FH_Squad_TotalSize_{t_}"
        for c_team in C_setofteams:
            players_in_team = P_c.get(c_team, [])
            if players_in_team: model += pulp.lpSum(x_freehit[p_tm][t_] for p_tm in players_in_team) <= MC * r[t_], f"FH_Squad_TeamLimit_{c_team}_{t_}"
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
        # Explicit Transfer Balance
        model += pulp.lpSum(e[p_][t_] for p_ in p) == pulp.lpSum(u[p_][t_] for p_ in p), f"Transfer_Balance_{t_}"


    # --- Linking Constraints & Evolution ---
    print("  Adding Linking and Evolution constraints...")
    t1_sub = t_sub[0] # First GW of this sub-problem

    # Link q[t1_sub] to previous state
    model += q[t1_sub] == previous_ft, f"FT_Link_{t1_sub}"

    # Link budget and squad for t1_sub
    if current_gw == START_GAMEWEEK:
        # Constraint 4.33: Initial budget limit (Use <= to allow saving budget)
        model += v[t1_sub] + pulp.lpSum(value_sub.loc[p_, t1_sub] * x[p_][t1_sub]) <= BS, f"Budget_Initial_{t1_sub}"
        # No transfers possible in GW1, force e=u=0
        model += pulp.lpSum(e[p_][t1_sub] for p_ in p) == 0, f"No_Transfers_In_StartGW"
        model += pulp.lpSum(u[p_][t1_sub] for p_ in p) == 0, f"No_Transfers_Out_StartGW"
    else:
        # Link budget based on previous week's end budget (Constraint 4.34)
        sales_value_t1 = pulp.lpSum(value_sub.loc[p_, t1_sub] * u[p_][t1_sub] for p_ in p)
        purchase_cost_t1 = pulp.lpSum(value_sub.loc[p_, t1_sub] * e[p_][t1_sub] for p_ in p)
        model += v[t1_sub] == previous_budget + sales_value_t1 - purchase_cost_t1, f"Budget_Link_{t1_sub}"
        # Link squad based on previous week's squad (Constraint 4.35)
        for p_ in p:
            model += x[p_][t1_sub] == previous_squad_dict.get(p_, 0) - u[p_][t1_sub] + e[p_][t1_sub], f"Squad_Link_{p_}_{t1_sub}"

    # Penalized Transfers for t1_sub
    model += alpha[t1_sub] >= pulp.lpSum(e[p_][t1_sub] for p_ in p) - q[t1_sub], f"PenalizedTransfers_Calc_{t1_sub}"
    model += alpha[t1_sub] <= M_alpha * (1 - w[t1_sub]), f"PenalizedTransfers_WC_Override_{t1_sub}"
    model += alpha[t1_sub] <= M_alpha * (1 - r[t1_sub]), f"PenalizedTransfers_FH_Override_{t1_sub}"

    # Evolution constraints for t > t1_sub within the sub-horizon
    for gw_idx in range(n_T_sub - 1):
        t_curr_sub = t_sub[gw_idx + 1]
        t_prev_sub = t_sub[gw_idx]

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
        model += ft_carried_over_nonneg[t_prev_sub] <= Q_bar - Q_under_bar, f"FT_Carry_Cap_{t_prev_sub}"

        chip_active_prev = w[t_prev_sub] + r[t_prev_sub]
        q_normal_calc = ft_carried_over_nonneg[t_prev_sub] + Q_under_bar

        model += q[t_curr_sub] <= q_normal_calc + M_q * chip_active_prev, f"FT_Evo_Normal_Upper_{t_curr_sub}"
        model += q[t_curr_sub] >= q_normal_calc - M_q * chip_active_prev, f"FT_Evo_Normal_Lower_{t_curr_sub}"
        model += q[t_curr_sub] <= Q_under_bar + M_q * (1 - chip_active_prev), f"FT_Evo_Chip_Reset_Upper_{t_curr_sub}"
        model += q[t_curr_sub] >= Q_under_bar - M_q * (1 - chip_active_prev), f"FT_Evo_Chip_Reset_Lower_{t_curr_sub}"

        # Constraint 4.42: No penalized transfers if max FT carried to next
        model += alpha[t_prev_sub] + M_alpha * q[t_curr_sub] <= M_alpha * Q_bar, f"Transfer_NoPaidIfMaxFT_{t_prev_sub}"


    print(f"Constraints added in {time.time() - cons_start:.2f}s")

    # 7. Solve Sub-Problem
    print(f"[{time.strftime('%H:%M:%S')}] Starting solve for GW {current_gw} (Solver Internal Limit: {SOLVER_TIME_LIMIT}s)...")
    print(f"[{time.strftime('%H:%M:%S')}] Solver output (msg=True) should appear below:")
    solve_start_time = time.time()
    solver_exception = None
    # --- Direct solve call ---
    try:
        print("Solver starting...")
        status = model.solve(solver_to_use)
        print("Solver finished.")
    except Exception as e:
        solver_exception = e
        print(f"Solver encountered an error: {e}")
        status = pulp.LpStatusUndefined

    solve_time = time.time() - solve_start_time

    if solver_exception: print("\n--- Solver Error ---"); print(f"Error: {solver_exception}"); status = -1000
    else: status = status

    if status is None: print("\nERROR: Solver status not captured."); status = -1000

    print(f"\n--- Solve Complete for GW {current_gw} ---")
    print(f"Final Solver Status Code: {status}")
    status_str = pulp.LpStatus.get(status, 'Unknown Status')
    print(f"Final Solver Status: {status_str}")
    print(f"Sub-problem Solve Time: {solve_time:.2f} seconds")
    time_limit_hit = SOLVER_TIME_LIMIT is not None and solve_time >= SOLVER_TIME_LIMIT * 0.98

    # 8. Extract Results and Update State
    objective_value_extracted = None; use_fallback = False
    try:
        if model.objective is not None: objective_value_extracted = model.objective.value()
    except AttributeError: pass
    solution_truly_optimal = (status == pulp.LpStatusOptimal)
    solution_acceptable_timeout = (status == pulp.LpStatusNotSolved and time_limit_hit and objective_value_extracted is not None)
    solution_acceptable = solution_truly_optimal or solution_acceptable_timeout

    if not solution_acceptable:
        print(f"!!! Solver failed/timed out without acceptable solution for GW {current_gw} (Status: {status_str}). Implementing FALLBACK. !!!")
        use_fallback = True
        try:
             lp_filename = f"failed_model_gw{current_gw}.lp"; model.writeLP(lp_filename)
             print(f"Model written to {lp_filename} for debugging.")
        except Exception as write_err: print(f"Could not write LP file: {write_err}")

    # --- Implement decision or fallback ---
    if not use_fallback:
        t1_sub = t_sub[0] # Gameweek to implement decisions for
        try:
            print(f"Extracting results for GW {current_gw} (t={t1_sub})...")
            gw_results = {'gameweek': current_gw}
            def get_var_val(var, default=0.0):
                try: val = var.varValue; return val if val is not None else default
                except AttributeError: return default

            # --- Initialize previous_squad_dict in the first GW ---
            if current_gw == START_GAMEWEEK:
                 previous_squad_dict = {p_: 1 for p_ in p if get_var_val(x[p_][t1_sub]) > 0.9}
                 print(f"Initialized previous_squad_dict with {len(previous_squad_dict)} players for GW{current_gw+1}")


            # Extract decisions for t1_sub
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
            gw_results['q_start'] = previous_ft # FT available at the start of this GW
            gw_results['objective_value'] = objective_value_extracted # Total objective for sub-horizon
            wc_active_gw = get_var_val(w[t1_sub]) > 0.9; bb_active_gw = get_var_val(b[t1_sub]) > 0.9; fh_active_gw = get_var_val(r[t1_sub]) > 0.9
            chip_name_display = None
            if wc_active_gw: chip_name_display = 'WC'
            if bb_active_gw: chip_name_display = 'BB'
            if fh_active_gw: chip_name_display = 'FH'
            if tc_active_gw and tc_player_id is not None: tc_player_name = player_name_map.get(tc_player_id, f"ID:{tc_player_id}"); chip_name_display = f'TC_{tc_player_name}'
            gw_results['chip_played'] = chip_name_display
            gw_results['budget_start'] = previous_budget # Store starting budget

            # --- Calculate and Store Weekly Objective Contributions ---
            weekly_objectives = {}
            for t_calc in t_sub:
                points_lineup_t = sum(get_var_val(y[p_][t_calc]) * points_sub.loc[p_, t_calc] for p_ in p)
                points_cap_t = sum(get_var_val(f[p_][t_calc]) * points_sub.loc[p_, t_calc] for p_ in p)
                points_vice_t = sum(get_var_val(h[p_][t_calc]) * epsilon * points_sub.loc[p_, t_calc] for p_ in p)
                points_tc_t = sum(get_var_val(is_tc[p_][t_calc]) * 2 * points_sub.loc[p_, t_calc] for p_ in p)
                points_subs_t = 0
                if g: points_subs_t = sum(kappa[l_] * get_var_val(g[p_ngk][t_calc][l_]) * points_sub.loc[p_ngk, t_calc] for p_ngk in P_not_gk for l_ in l)
                alpha_t_val = get_var_val(alpha[t_calc])
                penalty_t = R_penalty * alpha_t_val
                weekly_obj_val = points_lineup_t + points_cap_t + points_vice_t + points_tc_t + points_subs_t - penalty_t
                weekly_objectives[t_calc] = weekly_obj_val
                print(f"  GW {t_calc} Calculated Objective Contribution: {weekly_obj_val:.4f} (Penalty: {penalty_t:.1f})")
            # Store the objective for the first week (the one being implemented)
            gw_results['objective_gw'] = weekly_objectives.get(t1_sub, None)

            master_results.append(gw_results)

            # --- State Update Logic ---
            chip_played_this_gw = gw_results['chip_played']
            next_ft_value = previous_ft # Default
            try:
                if chip_played_this_gw == 'WC' or chip_played_this_gw == 'FH':
                    next_ft_value = Q_under_bar
                else: # BB, TC, or None
                    q_current_val = gw_results['q_start']
                    alpha_current_val = gw_results['alpha']
                    e_current_sum = len(gw_results['transfers_in'])
                    ft_used_current_eff = max(0, e_current_sum - alpha_current_val)
                    ft_carry_current = max(0, q_current_val - ft_used_current_eff)
                    next_ft_value = min(Q_bar, math.floor(ft_carry_current + Q_under_bar))
            except Exception as ft_err: print(f"Warning: Error calculating next FT: {ft_err}. Using default: {next_ft_value}")

            # Update persistent state (Squad/Budget)
            if fh_active_gw:
                print(f"FH played in GW {current_gw}. State (Squad/Budget) for next GW reverts.")
                previous_budget = gw_results['budget_start'] # Revert budget
            else:
                previous_squad_dict = {p_: 1 for p_ in gw_results['squad']}
                previous_budget = gw_results['budget_end'] # Update budget normally

            previous_ft = next_ft_value # Update FT for the *next* iteration

            # Update Chip Tracker
            if chip_played_this_gw:
                 if chip_played_this_gw == 'WC':
                     if current_gw <= mid_season_split_gw and not used_chips_tracker['wc1']: used_chips_tracker['wc1'] = True; print(f"--- WC1 activated in GW {current_gw} ---")
                     elif current_gw > mid_season_split_gw and not used_chips_tracker['wc2']: used_chips_tracker['wc2'] = True; print(f"--- WC2 activated in GW {current_gw} ---")
                 elif chip_played_this_gw == 'BB' and not used_chips_tracker['bb']: used_chips_tracker['bb'] = True; print(f"--- BB activated in GW {current_gw} ---")
                 elif chip_played_this_gw == 'FH' and not used_chips_tracker['fh']: used_chips_tracker['fh'] = True; print(f"--- FH activated in GW {current_gw} ---")
                 elif chip_played_this_gw.startswith('TC_') and not used_chips_tracker['tc']: used_chips_tracker['tc'] = True; print(f"--- {chip_played_this_gw} activated ---")

            print(f"End of GW {current_gw}: Budget={previous_budget:.1f}, Next FT={previous_ft}")

        except Exception as extract_err:
            print(f"!!! Error extracting results for GW {current_gw}: {extract_err} !!!")
            import traceback; traceback.print_exc()
            print("Stopping rolling horizon due to extraction error."); break

    else: # --- Handle Fallback Case ---
        print(f"Applying Fallback for GW {current_gw}: No transfers made.")
        next_ft_value = min(Q_bar, math.floor(previous_ft + Q_under_bar))
        gw_results = {
            'gameweek': current_gw, 'squad': sorted(list(previous_squad_dict.keys())),
            'lineup': [], 'captain': [], 'vice_captain': [],
            'transfers_in': [], 'transfers_out': [], 'budget_end': previous_budget,
            'alpha': 0, 'q_start': previous_ft, 'objective_value': None,
            'chip_played': 'FALLBACK_NO_TRANSFERS', 'budget_start': previous_budget,
            'objective_gw': None # No objective calculated in fallback
        }
        master_results.append(gw_results)
        previous_ft = next_ft_value # State update for FT
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
    # --- Add weekly objective to the summary printout ---
    def map_ids_to_names(id_list):
        if not isinstance(id_list, (list, tuple, set)): return id_list
        if not id_list: return []
        return sorted([player_name_map.get(p_id, f"ID:{p_id}") for p_id in id_list])
    id_list_columns = ['squad', 'lineup', 'captain', 'vice_captain', 'transfers_in', 'transfers_out']
    print("\nConverting player IDs to names in results DataFrame...")
    results_df_named = results_df.copy()
    for col in id_list_columns:
        if col in results_df_named.columns:
            results_df_named[col] = results_df_named[col].apply(map_ids_to_names)
    print("ID to Name conversion complete.")

    # Add this after the line "results_df_named = results_df.copy()" in the final results processing section

    # Calculate actual points if available in the source data
    if 'actual_total_points' in data_full_range_raw.columns:
        print("\nCalculating actual points from results...")
        # Create pivot table for actual points lookup
        actual_points_matrix = data_full_range_raw.pivot(index='player_id', columns='GW', values='actual_total_points')
        
        # Add actual points columns to results
        results_df_named['actual_squad_points'] = 0
        results_df_named['actual_lineup_points'] = 0
        results_df_named['actual_captain_points'] = 0  # This will store ONLY the bonus points
        results_df_named['actual_total_points'] = 0    # New field for the correct total
        
        for idx, row in results_df_named.iterrows():
            gw = row['gameweek']
            squad_ids = results_df.loc[idx, 'squad']  
            lineup_ids = results_df.loc[idx, 'lineup']
            captain_ids = results_df.loc[idx, 'captain']
            
            # Calculate base points for all players
            squad_actual = sum(actual_points_matrix.loc[p_id, gw] if p_id in actual_points_matrix.index and gw in actual_points_matrix.columns and not pd.isna(actual_points_matrix.loc[p_id, gw]) else 0 for p_id in squad_ids)
            lineup_actual = sum(actual_points_matrix.loc[p_id, gw] if p_id in actual_points_matrix.index and gw in actual_points_matrix.columns and not pd.isna(actual_points_matrix.loc[p_id, gw]) else 0 for p_id in lineup_ids)
            
            # Calculate captain BONUS points (not total)
            captain_actual_bonus = 0
            if captain_ids:
                captain_id = captain_ids[0]
                if captain_id in actual_points_matrix.index and gw in actual_points_matrix.columns and not pd.isna(actual_points_matrix.loc[captain_id, gw]):
                    captain_base_pts = actual_points_matrix.loc[captain_id, gw]
                    
                    # Only calculate the BONUS points (1x or 2x)
                    if row['chip_played'] and row['chip_played'].startswith('TC_'):
                        captain_actual_bonus = captain_base_pts * 2  # 2x bonus for TC
                    else:
                        captain_actual_bonus = captain_base_pts  # 1x bonus for regular captain
            
            # Add bench points if Bench Boost was used
            bench_pts = 0
            if row['chip_played'] == 'BB':
                bench_players = [p_id for p_id in squad_ids if p_id not in lineup_ids]
                bench_pts = sum(actual_points_matrix.loc[p_id, gw] if p_id in actual_points_matrix.index and gw in actual_points_matrix.columns and not pd.isna(actual_points_matrix.loc[p_id, gw]) else 0 for p_id in bench_players)
                lineup_actual += bench_pts  # Add bench to lineup when BB active
            
            # Store the values
            results_df_named.at[idx, 'actual_squad_points'] = squad_actual
            results_df_named.at[idx, 'actual_lineup_points'] = lineup_actual
            results_df_named.at[idx, 'actual_captain_points'] = captain_actual_bonus
            results_df_named.at[idx, 'actual_total_points'] = lineup_actual + captain_actual_bonus

    # Print summary per GW
    for index, row in results_df_named.iterrows():
        gw = row['gameweek']; print(f"\n--- GW {gw} Summary ---")
        print(f"  Chip Played: {row['chip_played'] if row['chip_played'] else 'None'}")
        transfers_in_str = ', '.join(map(str, row['transfers_in']))
        transfers_out_str = ', '.join(map(str, row['transfers_out']))
        print(f"  Transfers In ({len(row['transfers_in'])}): {transfers_in_str}")
        print(f"  Transfers Out ({len(row['transfers_out'])}): {transfers_out_str}")
        print(f"  FT Available (Start): {row['q_start']}")
        print(f"  Penalized Transfers (Hits): {row['alpha']}")
        cap_name = row['captain'][0] if row['captain'] else 'None'; vice_name = row['vice_captain'][0] if row['vice_captain'] else 'None'
        print(f"  Captain: {cap_name}"); print(f"  Vice-Captain: {vice_name}")
        print(f"  Budget End: {row['budget_end']:.1f}")
        obj_val_str = f"{row['objective_value']:.2f}" if row['objective_value'] is not None else "N/A"; print(f"  Objective Value (Sub-problem Total): {obj_val_str}")
        # --- Print the calculated weekly objective for this GW ---
        obj_gw_str = f"{row['objective_gw']:.2f}" if row['objective_gw'] is not None else "N/A (Fallback or Error)"; print(f"  Objective Value (GW {gw} Contribution): {obj_gw_str}")
        if 'actual_lineup_points' in results_df_named.columns:
            print(f"  Actual Points (Squad): {row['actual_squad_points']:.1f}")
            print(f"  Actual Points (Lineup): {row['actual_lineup_points']:.1f}")
            print(f"  Captain Bonus Points: {row['actual_captain_points']:.1f}")
            print(f"  Actual Total Points: {row['actual_total_points']:.1f}")


    # Save results
    try:
        output_csv_name = f"Squad Selection t-auto-actual, W{START_GAMEWEEK}-{MAX_GAMEWEEK},SHL{SUB_HORIZON_LENGTH}.csv"
        results_df_to_save = results_df_named.copy() # Use named df for saving
        for col in id_list_columns:
             if col in results_df_to_save.columns:
                 results_df_to_save[col] = results_df_to_save[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
        # Add the weekly objective column before saving
        results_df_to_save['objective_gw'] = results_df['objective_gw'] # Add the numeric column from original results
        results_df_to_save.to_csv(output_csv_name, index=False); print(f"\nResults saved to {output_csv_name}")
    except Exception as save_e: print(f"\nERROR: Could not save results to CSV. Reason: {save_e}")

print("\n--- Script Finished ---")