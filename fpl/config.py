"""Central configuration: paths and season parameters for the FPL pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETT_DIR = ROOT / "Datasett"
MASTER_DATASET_PATH = DATASETT_DIR / "master_dataset.csv"

MODELS_DIR = ROOT / "fpl" / "models"
PREDICTIONS_PATH = ROOT / "fpl" / "predictions_latest.csv"
SQUAD_OUTPUT_DIR = ROOT / "fpl" / "squad_selections"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
GITHUB_API_SEASONS_URL = "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data"

GWS_PER_SEASON = 38

# The ONE definition of the production forecaster's combination strategy, consumed as the
# default by BOTH backtest predictions (fpl.model.predict) and the live weekly run
# (fpl.run_week) - the two paths must never disagree about what "the production model" is
# (that skew was audit finding A1: live ran an untuned 12-member NNLS blend while every
# documented number came from tuned single:catboost). "single:<model>" puts weight 1.0 on
# that member; other options: "nnls" | "top_k" | "ridge" (see fpl.model.ensemble).
# fpl.model.train's combination bake-off warns when its empirical winner disagrees with
# this constant - update it here, deliberately, not per-callsite.
PRODUCTION_WEIGHT_STRATEGY = "single:catboost"

# Hyperparameter tuning must never validate on gameweeks the standing MILP backtest is run
# on, or the headline realized-points number stops being out-of-sample (audit finding A2).
# 152 = last GW before the 2024-25-season GW1-31 evaluation window (GW153-183). Like every
# global-GW constant this is season-ORDINAL: re-derive it if DEFAULT_START_SEASON changes.
TUNING_TRAIN_MAX_GW = 152

# MILP solver settings for fpl/milp/optimize.py, benchmarked 2026-07-16 on the
# GW153-183 standard backtest (see RESEARCH_LOG.md). MILP_SOLVER: "highs" (via the
# highspy package) or "cbc" (PuLP's bundled COIN-OR CBC) - both prove optimality,
# so squads are identical; HiGHS is ~20% faster overall with a better worst
# gameweek (7.7s vs 12.7s). MILP_THREADS 0 = solver default (8 threads measured
# no faster - the branch-and-bound tree here is too small to parallelize).
# MILP_GAP_REL 0 = prove full optimality. Do NOT loosen the gap for speed:
# gap 0.001 saved ~20% wall time but changed squads and cost 40 realized points
# on the benchmark window.
MILP_SOLVER = "highs"
MILP_THREADS = 0
MILP_GAP_REL = 0.0

# The solver's free-transfer BANKING POLICY (Q_bar in the Kristiansen formulation) and FTs
# gained per gameweek (Q_under_bar). NOTE: 2 is deliberately BELOW the site's banking cap
# (FPL allows 5 since 2024-25) - banking less than allowed is legal play, and the tighter
# policy measured better (TODO 3.4, RESEARCH_LOG 2026-07-20): on GW153-183, cap 5 scored
# 1951 vs 2086 at horizon 3 and 1886 vs 1977 at horizon 5 - each a leaning-negative tie,
# consistently negative in sign tests. Mechanism: forecasts are noisy, so a predicted-
# indifferent deferral is realized-costly on average; cap 2's use-it-or-lose-it pressure
# is accidentally protective (same regularization-beats-flexibility pattern as
# single-CatBoost-vs-blends and MILP_GAP_REL=0). A live squad holding MORE than 2 banked
# FTs is still honored: optimize.py bounds the FT state at max(this, --initial-ft).
MILP_MAX_FREE_TRANSFERS = 2
MILP_FT_PER_GW = 1

# Extended back from the original 2022-23 thesis scope to get more history per player
# (see RESEARCH_LOG.md). 2020-21 is the earliest season where `position`/`team` are still
# present directly in vaastav's merged_gw.csv (older seasons need a players_raw.csv join
# fetch.py doesn't do yet) and BPS/ICT-index/influence/creativity/threat go back even
# further (to 2016-17) - NOT the limiting factor. What IS lost for 2020-21/2021-22 rows:
# the Opta expected-goals family (expected_goals/expected_assists/expected_goal_involvements/
# expected_goals_conceded) and `starts`, both only present from 2022-23 onward - those
# columns are NaN for the older two seasons, which LightGBM (this pipeline's main model)
# handles natively; linear models fall back to imputing 0 there via SimpleImputer.
# Override by passing an explicit `seasons` list.
DEFAULT_START_SEASON = "2020-21"

# Manual name/team corrections carried over from the original R data pipeline
# (Datasett/R-Script 1 fetching data.r) — vaastav's raw data has inconsistent
# player names across seasons.
NAME_CORRECTIONS = {
    "Mitoma Kaoru": "Kaoru Mitoma",
    "Tomiyasu Takehiro": "Takehiro Tomiyasu",
    "Endo Wataru": "Wataru Endo",
    "Kim Ji-Soo": "Ji-Soo Kim",
    "Olu Aina": "Ola Aina",
    "Dominic Solanke-Mitchell": "Dominic Solanke",
    "Kaine Kesler-Hayden": "Kaine Kesler Hayden",
    "Adama Traore Diarra": "Adama Traore",
    "Omari Giraud-Hutchinson": "Omari Hutchinson",
    "Joe Gomez": "Joseph Gomez",
    "Rodrigo 'Rodri' Hernandez": "Rodrigo Hernandez",
    "Yehor Yarmoliuk": "Yegor Yarmoliuk",
    "Michale Olakigbe": "Michael Olakigbe",
    "Djordje Petrovic": "Dorde Petrovic",
    "Joshua King": "Josh King",
    "Ben Brereton": "Ben Brereton Diaz",
    "Jaden Philogene": "Jaden Philogene-Bidace",
    "Carlos Alcaraz": "Carlos Alcaraz Duran",
    "Luis Sinisterra": "Luis Sinisterra Lucumi",
    "Joe Aribo": "Joe Ayodele-Aribo",
    "Joseph Hodge": "Joe Hodge",
    "Tom Cannon": "Thomas Cannon",
    "Yerson Mosquera": "Yerson Mosquera Valdelamar",
    "Max Kinsey": "Max Kinsey-Wellings",
    "Alexandre Moreno Lopera": "Alex Moreno Lopera",
    "Joe Whitworth": "Joseph Whitworth",
}

TEAM_NAME_CORRECTIONS = {
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    "Man Utd": "Man United",
    "Leicester City": "Leicester",
}

ONFIELD_POSITIONS = ["GK", "DEF", "MID", "FWD"]

# player_id is the FPL `code` (players_raw.csv / bootstrap-static) - a permanent per-person
# integer, currently 6-7 digits. Rows whose element id is missing from players_raw.csv get
# a name-factorized fallback id starting at this offset, far above any real code, so
# fallback ids can never collide with (or be mistaken for) genuine FPL codes.
FALLBACK_PLAYER_ID_OFFSET = 100_000_000
