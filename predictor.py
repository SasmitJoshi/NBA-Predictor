import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from nba_api.stats.endpoints import LeagueGameLog

seasons = ['2022-23', '2023-24', '2024-25']
all_games = []

for season in seasons:
    games = LeagueGameLog(season=season, season_type_all_star='Regular Season').get_data_frames()[0]
    games['SEASON'] = season
    all_games.append(games)

all_games_df = pd.concat(all_games, ignore_index=True)
all_games_df.to_csv('nba_games_2022_to_2025.csv', index=False)

