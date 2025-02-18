def calculate_offense(
    home_2FGM, home_3FGM, home_FTM, 
    home_2FGA, home_3FGA, home_FTA, 
    home_OReb, home_Assists, home_Fouls_Rv, 
    home_Turnovers, home_Blocks_Ag):
    return home_2FGM+home_3FGM+home_FTM-(home_2FGA-home_2FGM+home_3FGA-home_3FGM+home_FTA-home_FTM)+home_OReb+home_Assists+home_Fouls_Rv-home_Turnovers-home_Blocks_Ag


def calculate_defense(
    away_2FGM, away_3FGM, away_FTM, 
    away_2FGA, away_3FGA, away_FTA, 
    home_DReb, home_Steals, home_Blocks_Fv, 
    away_OReb, home_Fouls_Com):
    return -(away_2FGM+away_3FGM+away_FTM)+(away_2FGA-away_2FGM+away_3FGA-away_3FGM+away_FTA-away_FTM)+home_DReb+home_Steals+home_Blocks_Fv-away_OReb-home_Fouls_Com
