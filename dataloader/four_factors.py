def calculate_home_offense_tp(
    home_Turnovers,home_2FGA,home_3FGA,
    home_FTA,home_OReb):
    return home_Turnovers/(home_2FGA+home_3FGA+(0.44*home_FTA)-home_OReb+home_Turnovers)

def calculate_home_offense_orp(
    home_OReb,away_DReb):
    return (home_OReb)/(home_OReb+away_DReb)

def calculate_home_offense_ftr(
    home_FTA,home_2FGA,home_3FGA):
    return (home_FTA)/(home_2FGA+home_3FGA)

def calculate_home_defense_tp(
    away_Turnovers,away_2FGA,away_3FGA,
    away_FTA,away_OReb):
    return away_Turnovers/(away_2FGA+away_3FGA+(0.44*away_FTA)-away_OReb+away_Turnovers)

def calculate_home_defense_orp(
     away_OReb,home_DReb):
     return (away_OReb)/(away_OReb+home_DReb)
    
def calculate_home_defense_ftr(
     away_FTA,away_2FGA,away_3FGA):
     return (away_FTA)/(away_2FGA+away_3FGA)

def calculate_home_offense_rating(
     home_offense_efgp,home_offense_tp,home_offense_orp,home_offense_ftr):
     return (0.4*home_offense_efgp)+(0.25*home_offense_tp)+(0.2*home_offense_orp)+(0.15*home_offense_ftr)
    
def calculate_home_defense_rating(
     home_defense_efgp,home_defense_tp,home_defense_orp,home_defense_ftr):
     return (0.4*home_defense_efgp)+(0.25*home_defense_tp)+(0.2*home_defense_orp)+(0.15*home_defense_ftr)