import pandas as pd
from dataloader.data_prepocessing import (load_data, calculate_weighted_rolling_averages,
                             preprocess_data, merge_home_away_data, calculate_seasonal_averages,calculate_rate)
from dataloader.four_factors import (calculate_home_offense_tp, calculate_home_offense_orp,
                          calculate_home_offense_ftr, calculate_home_defense_tp,
                          calculate_home_defense_orp, calculate_home_defense_ftr,
                          calculate_home_offense_rating,calculate_home_defense_rating)
from dataloader.defenseofense import calculate_offense, calculate_defense
from dataloader.Elo import calculate_elo_ratings

def construct_weighted_feature_data(filepath):
    Data = load_data(filepath)
    avg_data = calculate_weighted_rolling_averages(Data, num_days=15)  
    avg_data = preprocess_data(avg_data)  
    merged_data = merge_home_away_data(avg_data)  
    
    return merged_data

def construct_seasonol_feature_data(filepath):
    Data = load_data(filepath)
    Basis_data = calculate_seasonal_averages(Data) 
    Basis_data = merge_home_away_data(Basis_data)  
    Basis_data = preprocess_data(Basis_data)  
    
    
    rate_columns = {
    'avg_REB_Percent': ('avg_REB_home', 'avg_REB_away'),
    'avg_AS_Percent': ('avg_AS_home', 'avg_AS_away'),
    'avg_ST_Percent': ('avg_ST_home', 'avg_ST_away'),
    'avg_BS_Percent': ('avg_BS_home', 'avg_BS_away'),
    'avg_FOUL_Percent': ('avg_FOUL_home', 'avg_FOUL_away'),
    'avg_TO_Percent': ('avg_TO_home', 'avg_TO_away'),
    'avg_FG%_Percent': ('avg_FG%_home', 'avg_FG%_away'),
    'avg_eFG%_Percent': ('avg_eFG%_home', 'avg_eFG%_away'),
    'avg_TS%_Percent': ('avg_TS%_home', 'avg_TS%_away')}
    
    for new_col, (home_col, away_col) in rate_columns.items():
        Basis_data[new_col] = calculate_rate(Basis_data[home_col], Basis_data[away_col])

    Basis_data['avg_ORTG'] = Basis_data['avg_ORTG_home']
    Basis_data['avg_DRTG'] = Basis_data['avg_ORTG_away']

    selected_columns = ['SEASON', 'ROUND', 'NUMBER', 'TEAM_home', 'RESULT1_home'] + list(rate_columns.keys()) + ['avg_ORTG', 'avg_DRTG']

    seasonal_data = Basis_data[selected_columns]
    return seasonal_data

def construct_FourFactors(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['avg_home_offense_tp'] = calculate_home_offense_tp(merged_data['avg_TO_home'], merged_data['avg_2PA_home'], merged_data['avg_3PA_home'],
                                                                   merged_data['avg_FTA_home'], merged_data['avg_OREB_home'])
    merged_data['avg_home_offense_orp'] = calculate_home_offense_orp(merged_data['avg_OREB_home'], merged_data['avg_DREB_away'])
    merged_data['avg_home_offense_ftr'] = calculate_home_offense_ftr(merged_data['avg_FTA_home'], merged_data['avg_2PA_home'], merged_data['avg_3PA_home'])

    merged_data['avg_home_defense_tp'] = calculate_home_defense_tp(merged_data['avg_TO_away'], merged_data['avg_2PA_away'], merged_data['avg_3PA_away'],
                                                               merged_data['avg_FTA_away'], merged_data['avg_OREB_away'])
    merged_data['avg_home_defense_orp'] = calculate_home_defense_orp(merged_data['avg_OREB_away'], merged_data['avg_DREB_home'])
    merged_data['avg_home_defense_ftr'] = calculate_home_defense_ftr(merged_data['avg_FTA_away'], merged_data['avg_2PA_away'], merged_data['avg_3PA_away'])

    merged_data['avg_home_offense_rating'] = calculate_home_offense_rating(merged_data['avg_eFG%_home'], merged_data['avg_home_offense_tp'], merged_data['avg_home_offense_orp'], merged_data['avg_home_offense_ftr'])
    merged_data['avg_home_defense_rating'] = calculate_home_defense_rating(merged_data['avg_eFG%_away'], merged_data['avg_home_defense_tp'], merged_data['avg_home_defense_orp'], merged_data['avg_home_defense_ftr'])

    selected_columns = ['SEASON', 'ROUND', 'NUMBER', 'TEAM_home', 'RESULT1_home',
                        'avg_home_offense_rating', 'avg_home_defense_rating']
    
    return merged_data[selected_columns]

def construct_FourFactors_detailed(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['avg_home_offense_tp'] = calculate_home_offense_tp(merged_data['avg_TO_home'], merged_data['avg_2PA_home'], merged_data['avg_3PA_home'],
                                                               merged_data['avg_FTA_home'], merged_data['avg_OREB_home'])
    merged_data['avg_home_offense_orp'] = calculate_home_offense_orp(merged_data['avg_OREB_home'], merged_data['avg_DREB_away'])
    merged_data['avg_home_offense_ftr'] = calculate_home_offense_ftr(merged_data['avg_FTA_home'], merged_data['avg_2PA_home'], merged_data['avg_3PA_home'])

    merged_data['avg_home_defense_tp'] = calculate_home_defense_tp(merged_data['avg_TO_away'], merged_data['avg_2PA_away'], merged_data['avg_3PA_away'],
                                                               merged_data['avg_FTA_away'], merged_data['avg_OREB_away'])
    merged_data['avg_home_defense_orp'] = calculate_home_defense_orp(merged_data['avg_OREB_away'], merged_data['avg_DREB_home'])
    merged_data['avg_home_defense_ftr'] = calculate_home_defense_ftr(merged_data['avg_FTA_away'], merged_data['avg_2PA_away'], merged_data['avg_3PA_away'])

    merged_data['avg_home_offense_rating'] = calculate_home_offense_rating(merged_data['avg_eFG%_home'], merged_data['avg_home_offense_tp'], merged_data['avg_home_offense_orp'], merged_data['avg_home_offense_ftr'])
    merged_data['avg_home_defense_rating'] = calculate_home_defense_rating(merged_data['avg_eFG%_away'], merged_data['avg_home_defense_tp'], merged_data['avg_home_defense_orp'], merged_data['avg_home_defense_ftr'])
    
    selected_columns = ['SEASON', 'ROUND', 'NUMBER', 'TEAM_home', 'RESULT1_home',
                        'avg_eFG%_home','avg_home_offense_tp','avg_home_offense_orp','avg_home_offense_ftr',
                        'avg_eFG%_away','avg_home_defense_tp','avg_home_defense_orp','avg_home_defense_ftr']
    
    return merged_data[selected_columns]

def construct_DefenseOfense(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['avg_OFFENSE'] = calculate_offense(merged_data['avg_2P_home'],merged_data['avg_3P_home'],merged_data['avg_FT_home'],
                                           merged_data['avg_2PA_home'],merged_data['avg_3PA_home'],merged_data['avg_FTA_home'],
                                           merged_data['avg_OREB_home'],merged_data['avg_AS_home'],merged_data['avg_FOUL_away'],
                                           merged_data['avg_TO_home'],merged_data['avg_BS_away'])
    
    merged_data['avg_DEFENSE'] = calculate_defense(merged_data['avg_2P_away'],merged_data['avg_3P_away'],merged_data['avg_FT_away'],
                                           merged_data['avg_2PA_away'],merged_data['avg_3PA_away'],merged_data['avg_FT_away'],
                                           merged_data['avg_DREB_home'],merged_data['avg_ST_home'],merged_data['avg_BS_home'],
                                           merged_data['avg_OREB_away'],merged_data['avg_FOUL_home'])
    
    selected_columns = ['SEASON','ROUND','NUMBER','TEAM_home','RESULT1_home','avg_OFFENSE','avg_DEFENSE']
    
    return merged_data[selected_columns]

def construct_DefenseOfense_detailed(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['avg_OFFENSE'] = calculate_offense(merged_data['avg_2P_home'],merged_data['avg_3P_home'],merged_data['avg_FT_home'],
                                           merged_data['avg_2PA_home'],merged_data['avg_3PA_home'],merged_data['avg_FTA_home'],
                                           merged_data['avg_OREB_home'],merged_data['avg_AS_home'],merged_data['avg_FOUL_away'],
                                           merged_data['avg_TO_home'],merged_data['avg_BS_away'])
    
    merged_data['avg_DEFENSE'] = calculate_defense(merged_data['avg_2P_away'],merged_data['avg_3P_away'],merged_data['avg_FT_away'],
                                           merged_data['avg_2PA_away'],merged_data['avg_3PA_away'],merged_data['avg_FT_away'],
                                           merged_data['avg_DREB_home'],merged_data['avg_ST_home'],merged_data['avg_BS_home'],
                                           merged_data['avg_OREB_away'],merged_data['avg_FOUL_home'])
    
    selected_columns = ['SEASON','ROUND','NUMBER','TEAM_home','RESULT1_home','avg_2P_home', 
                        'avg_2PA_home', 'avg_3P_home', 'avg_3PA_home','avg_FT_home', 'avg_FTA_home', 
                        'avg_DREB_home', 'avg_OREB_home', 'avg_REB_home', 'avg_AS_home', 'avg_ST_home', 
                        'avg_TO_home', 'avg_BS_home', 'avg_BS_away', 'avg_FOUL_home', 'avg_FOUL_away']
    
    return merged_data[selected_columns]

def construct_Elo_features(filepath):
    Data = load_data(filepath)
    elo_data = calculate_elo_ratings(Data)
    elo_data = elo_data[elo_data['HOME&AWAY'] == 'HOME']
    selected_columns = ['SEASON', 'ROUND', 'NUMBER', 'HOME_ELO', 'AWAY_ELO', 'HOME_RECENT_ELO_CHANGE', 'AWAY_RECENT_ELO_CHANGE']
    
    return elo_data[selected_columns]
    
    
def construct_baseline_data(filepath, model_type):
    Data = load_data(filepath)
    
    if model_type == "FourFactors":
        return construct_FourFactors(Data)
    elif model_type == "FourFactors_detailed":
        return construct_FourFactors_detailed(Data)
    elif model_type == "DefenseOfense":
        return construct_DefenseOfense(Data)
    elif model_type == "DefenseOfense_detailed":
        return construct_DefenseOfense_detailed(Data)
    else:
        raise ValueError("Invalid baseline model type!")

def construct_enhanced_data(filepath, model_type):
    baseline_data = construct_baseline_data(filepath, model_type)
    seasonal_data = construct_seasonol_feature_data(filepath)
    elo_data = construct_Elo_features(filepath)

    enhanced_data1 = pd.merge(seasonal_data, elo_data, on=['SEASON', 'ROUND', 'NUMBER'], how='left')
    enhanced_data = pd.merge(enhanced_data1, baseline_data, on=['SEASON', 'ROUND', 'NUMBER','TEAM_home','RESULT1_home'], how='left')

    return enhanced_data

