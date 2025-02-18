# Elo 计算公式
def calculate_elo(elo_x, elo_y, result_x, k=32):
    Ex = 1 / (1 + 10 ** ((elo_y - elo_x) / 400))
    Ey = 1 / (1 + 10 ** ((elo_x - elo_y) / 400))
    new_elo_x = elo_x + k * (result_x - Ex)
    new_elo_y = elo_y + k * ((1 - result_x) - Ey)
    return new_elo_x, new_elo_y

def reset_elo(elo, base_elo=1000, ratio=0.5):
    return elo * (1 - ratio) + base_elo * ratio

def calculate_recent_elo_change(elo_history, team, n=5):
    history = elo_history[team]
    return sum(history[-n:]) if len(history) >= n else sum(history)

def calculate_elo_ratings(data):
    teams = data['TEAM'].unique()
    elo = {team: 1000 for team in teams}  
    elo_history = {team: [] for team in teams}  

    current_season = None
    for i in range(0, len(data), 2):
        season = data.at[i, 'SEASON']

        if current_season != season:
            current_season = season
            for team in teams:
                elo[team] = reset_elo(elo[team])
                elo_history[team] = []

        home_team = data.at[i, 'TEAM']
        away_team = data.at[i + 1, 'TEAM']
        
        home_result = data.at[i, 'RESULT1']
        
        elo_home = elo[home_team]
        elo_away = elo[away_team]

        data.at[i, 'HOME_ELO'] = elo_home
        data.at[i + 1, 'HOME_ELO'] = elo_home
        data.at[i, 'AWAY_ELO'] = elo_away
        data.at[i + 1, 'AWAY_ELO'] = elo_away

        new_elo_home, new_elo_away = calculate_elo(elo_home, elo_away, home_result)

        elo[home_team] = new_elo_home
        elo[away_team] = new_elo_away

        elo_change_home = new_elo_home - elo_home
        elo_change_away = new_elo_away - elo_away
        elo_history[home_team].append(elo_change_home)
        elo_history[away_team].append(elo_change_away)

        recent_elo_change_home = calculate_recent_elo_change(elo_history, home_team)
        recent_elo_change_away = calculate_recent_elo_change(elo_history, away_team)

        data.at[i, 'HOME_RECENT_ELO_CHANGE'] = recent_elo_change_home
        data.at[i + 1, 'HOME_RECENT_ELO_CHANGE'] = recent_elo_change_home
        data.at[i, 'AWAY_RECENT_ELO_CHANGE'] = recent_elo_change_away
        data.at[i + 1, 'AWAY_RECENT_ELO_CHANGE'] = recent_elo_change_away

    return data
