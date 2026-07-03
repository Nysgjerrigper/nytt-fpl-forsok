import requests
import time
import matplotlib.pyplot as plt

PREDICTED_GW31_POINTS = 2045  # Your model's total points up to GW31
GW_CUTOFF = 31
MAX_PAGES = 30  # Top 1500 players (each page = 50 users)
SLEEP = 1

def get_league_standings(page):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings={page}"
    return requests.get(url).json()['standings']['results']

def get_total_players():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    return requests.get(url).json()['total_players']

def get_user_gw_points(entry_id, gw=GW_CUTOFF):
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/history/"
    data = requests.get(url).json()
    for gw_data in data['current']:
        if gw_data['event'] == gw:
            return gw_data['total_points']
    return None

def build_gw31_rank_table(predicted_points, max_pages):
    results = []
    rank = 1
    for page in range(1, max_pages + 1):
        print(f"Page {page}")
        players = get_league_standings(page)
        for p in players:
            entry_id = p['entry']
            time.sleep(SLEEP)
            try:
                gw_points = get_user_gw_points(entry_id)
                if gw_points is not None:
                    results.append((rank, gw_points))
                    if gw_points <= predicted_points:
                        return rank, results
                    rank += 1
            except:
                continue
    return None, results

def plot_rank_vs_points(data, predicted):
    ranks, points = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.plot(points, ranks, lw=2, label="Actual leaderboard (GW31)")
    plt.gca().invert_yaxis()
    for r, p in data:
        if p <= predicted:
            plt.scatter(p, r, color='red', label=f'Your model ({predicted} pts → #{r})')
            break
    plt.xlabel("Points after GW31")
    plt.ylabel("Overall Rank")
    plt.title("FPL Rank vs Total Points (after GW31)")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Run it ===
total_players = get_total_players()
approx_rank, rank_data = build_gw31_rank_table(PREDICTED_GW31_POINTS, MAX_PAGES)

print(f"\n📊 Estimated rank after GW31 with {PREDICTED_GW31_POINTS} points: #{approx_rank}")
print(f"🌍 Total players: {total_players:,}")

plot_rank_vs_points(rank_data, PREDICTED_GW31_POINTS)
