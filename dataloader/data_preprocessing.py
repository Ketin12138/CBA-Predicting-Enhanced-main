import pandas as pd

def load_data(filepath):
    if isinstance(filepath, pd.DataFrame):
        return filepath  
    else:
        return pd.read_excel(filepath)  


def calculate_weighted_rolling_averages(data, num_days):
    rolling_data = data.copy()

    columns_to_avg = ['PTS', 'OREB', 'DREB', 'REB', 'AS', 'ST', 'BS', 'FOUL', 'TO', '2P', '2PA', '3P', '3PA', 'FT', 'FTA', 'FG', 'FGA']
    percent_columns = ['2P%', '3P%', 'FT%', 'FG%', 'eFG%', 'TS%']
    
    for col in columns_to_avg + percent_columns:
        rolling_data[f'avg_{col}'] = 0.0

    team_stats = {}

    for idx, row in rolling_data.iterrows():
        team = row['TEAM']

        if team not in team_stats:
            team_stats[team] = []

        recent_games = team_stats[team][-num_days:]
        num_games = len(recent_games)
        
        actual_weights_sum = sum(range(1, num_games + 1))

        for col in columns_to_avg:
            if num_games > 0:
                weighted_sum = sum(game[col] * (i + 1) for i, game in enumerate(recent_games))
                rolling_data.at[idx, f'avg_{col}'] = weighted_sum / actual_weights_sum

        total_fga = sum(game['FGA'] * (i + 1) for i, game in enumerate(recent_games))
        total_fta = sum(game['FTA'] * (i + 1) for i, game in enumerate(recent_games))
        total_2pa = sum(game['2PA'] * (i + 1) for i, game in enumerate(recent_games))
        total_3pa = sum(game['3PA'] * (i + 1) for i, game in enumerate(recent_games))

        if total_2pa > 0:
            rolling_data.at[idx, 'avg_2P%'] = sum(game['2P'] * (i + 1) for i, game in enumerate(recent_games)) / total_2pa
        if total_3pa > 0:
            rolling_data.at[idx, 'avg_3P%'] = sum(game['3P'] * (i + 1) for i, game in enumerate(recent_games)) / total_3pa
        if total_fta > 0:
            rolling_data.at[idx, 'avg_FT%'] = sum(game['FT'] * (i + 1) for i, game in enumerate(recent_games)) / total_fta
        if total_fga > 0:
            fg_total = sum(game['FG'] * (i + 1) for i, game in enumerate(recent_games))
            rolling_data.at[idx, 'avg_FG%'] = fg_total / total_fga
            rolling_data.at[idx, 'avg_eFG%'] = (sum((game['2P'] + 1.5 * game['3P']) * (i + 1) for i, game in enumerate(recent_games))) / total_fga
            rolling_data.at[idx, 'avg_TS%'] = sum(game['PTS'] * (i + 1) for i, game in enumerate(recent_games)) / (2 * (total_fga + 0.44 * sum(game['FT'] * (i + 1) for i, game in enumerate(recent_games))))

        game_data = {col: row[col] for col in columns_to_avg}
        team_stats[team].append(game_data)

    return rolling_data

def calculate_rate(home_value, away_value):
    denominator = home_value + away_value
    return home_value / denominator.where(denominator != 0, other=0.5)

def calculate_seasonal_averages(data):

    columns_to_avg = ['PTS', 'OREB', 'DREB', 'REB', 'AS', 'ST', 'BS', 'FOUL', 
                      'TO', '2P', '2PA', '3P', '3PA', 'FT', 'FTA', 'FG', 'FGA', 'ORTG']
    percent_columns = ['2P%', '3P%', 'FT%', 'FG%', 'eFG%', 'TS%']
    
    for col in columns_to_avg + percent_columns:
        data[f'avg_{col}'] = 0.0

    team_stats = {}

    for idx, row in data.iterrows():
        team = row['TEAM']
        season = row['SEASON']

        if team not in team_stats:
            team_stats[team] = {}
        if season not in team_stats[team]:
            team_stats[team][season] = []

        season_games = team_stats[team][season]
        num_games = len(season_games)

        for col in columns_to_avg:
            if num_games > 0:
                avg_value = sum(game[col] for game in season_games) / num_games
                data.at[idx, f'avg_{col}'] = avg_value

        total_fga = sum(game['FGA'] for game in season_games)
        total_fta = sum(game['FTA'] for game in season_games)
        total_2pa = sum(game['2PA'] for game in season_games)
        total_3pa = sum(game['3PA'] for game in season_games)

        if total_2pa > 0:
            data.at[idx, 'avg_2P%'] = sum(game['2P'] for game in season_games) / total_2pa
        if total_3pa > 0:
            data.at[idx, 'avg_3P%'] = sum(game['3P'] for game in season_games) / total_3pa
        if total_fta > 0:
            data.at[idx, 'avg_FT%'] = sum(game['FT'] for game in season_games) / total_fta
        if total_fga > 0:
            fg_total = sum(game['FG'] for game in season_games)
            data.at[idx, 'avg_FG%'] = fg_total / total_fga
            data.at[idx, 'avg_eFG%'] = sum((game['2P'] + 1.5 * game['3P']) for game in season_games) / total_fga
            data.at[idx, 'avg_TS%'] = sum(game['PTS'] for game in season_games) / (2 * (total_fga + 0.44 * sum(game['FT'] for game in season_games)))

        game_data = {col: row[col] for col in columns_to_avg}
        team_stats[team][season].append(game_data)

    columns_to_keep = ['SEASON', 'ROUND', 'NUMBER', 'HOME&AWAY', 'TEAM', 'RESULT1']
    columns_to_avg_new = [f'avg_{col}' for col in columns_to_avg + percent_columns]
    columns_to_retain = columns_to_keep + columns_to_avg_new

    return data[columns_to_retain]

def preprocess_data(data):
    return data[~(data['SEASON'] == '2020-2021') & 
                ~((data['SEASON'] == '2021-2022') & (data['ROUND'] == 1)) & 
                ~((data['SEASON'] == '2022-2023') & (data['ROUND'] == 1)) & 
                ~((data['SEASON'] == '2023-2024') & (data['ROUND'] == 1))]

def merge_home_away_data(data):
    Home_Data = data[data['HOME&AWAY'] == 'HOME'].copy()
    Away_Data = data[data['HOME&AWAY'] == 'AWAY'].copy()
    merged_data = pd.merge(Home_Data, Away_Data, on=['SEASON', 'ROUND', 'NUMBER'], suffixes=('_home', '_away'))
    return merged_data
