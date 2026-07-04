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
