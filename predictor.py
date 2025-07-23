import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from nba_api.stats.endpoints import LeagueGameLog
import re

def get_location(row):
    team = row['TEAM_ABBREVIATION']
    matchup = row['MATCHUP']

    if re.search(fr"{team}\s+vs\.?", matchup):
        return 'Home'
    elif re.search(fr"{team}\s+@", matchup):
        return 'Away'
    else:
        return 'Unknown'

seasons = ['2022-23', '2023-24', '2024-25']

def get_games(seasons):
    all_games = []
    for season in seasons:
        games = LeagueGameLog(season=season, season_type_all_star='Regular Season').get_data_frames()[0]
        games['SEASON'] = season
        all_games.append(games)

    all_games_df = pd.concat(all_games, ignore_index=True)
    all_games_df.to_csv('nba_games_2022_to_2025.csv', index=False)


def get_opponent(row):
    team = row['TEAM_ABBREVIATION']
    matchup = row['MATCHUP']

    matchup_clean = matchup.replace('.', '').strip()

    if ' vs ' in matchup_clean:
        home_team, away_team = matchup_clean.split(' vs ')
    elif ' @ ' in matchup_clean:
        away_team, home_team = matchup_clean.split(' @ ')
    else:
        return None

    if team == home_team:
        return away_team
    elif team == away_team:
        return home_team
    else:
        return None

# Read the CSV File
games = pd.read_csv('nba_games_2022_to_2025.csv')

# Pre-process the data
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

games = games[['GAME_DATE', 'SEASON', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL']]

games['VENUE'] = games.apply(get_location, axis=1)
games['OPPONENT'] = games.apply(get_opponent, axis=1)

games['VENUE_CODE'] = games['VENUE'].astype('category').cat.codes
games['OPPONENT_CODE'] = games['OPPONENT'].astype('category').cat.codes

games["GOAL"] = (games["WL"] == "W")

print(games)

# Train model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = games[games['GAME_DATE'] < '2023-01-01']
test = games[games['GAME_DATE'] > '2023-01-01']

predictors = ['VENUE_CODE', 'OPPONENT_CODE']

rf.fit(train[predictors], train['GOAL'])
RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)
predictions = rf.predict(test[predictors])

accuracy = accuracy_score(test['GOAL'], predictions)
print(accuracy)

combined = pd.DataFrame(dict(actual=test['GOAL'], prediction=predictions))
print(pd.crosstab(index=combined['actual'], columns=combined['prediction']))

print(precision_score(test['GOAL'], predictions))