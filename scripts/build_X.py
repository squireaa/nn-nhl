import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from getter_methods import *


def generate_col_names(num_needed: int) -> list:
    return ['Column' + str(i+1) for i in range(num_needed)]

ou_X_df = pd.DataFrame()

every_my_id = list()
start_year = 18
end_year = 23
for year in range(start_year, end_year):
    file_path = f"nn-nhl/data/adv_20{year}.csv"
    df = pd.read_csv(file_path)

    num_rows = len(df)
    for i in range(num_rows):
        team_abv = get_three_letter_code(df['Team'].iloc[i])
        date = df['Date'].iloc[i]
        every_my_id.append(f"{team_abv}-{date}")
        

def get_main_metrics(my_id: str) -> list:
    to_return = []
    to_return.extend(get_last_n(my_id, 1))
    to_return.extend(get_last_n(my_id, 3))
    to_return.extend(get_last_n(my_id, 5))
    to_return.extend(get_last_n(my_id, 10))
    to_return.extend([get_rest_days(my_id)])
    to_return.extend([get_year(my_id)])
    return to_return

def build_ml():
    col_names = generate_col_names(502)
    ml_X_df = pd.DataFrame(columns=col_names)
    ml_index = 0

    num_of_ids = len(every_my_id)
    for i in range(0, num_of_ids, 2):
        try:
            team1_id = every_my_id[i]
            team2_id = every_my_id[i + 1]

            team1_metrics = get_main_metrics(team1_id)
            team2_metrics = get_main_metrics(team2_id)

            team1_open_ml = get_open_ml(team1_id)
            team2_open_ml = get_open_ml(team2_id)
            team1_close_ml = get_close_ml(team1_id)
            team2_close_ml = get_close_ml(team2_id)

            ml_row_1 = team1_metrics + team2_metrics + team1_open_ml
            ml_row_2 = team1_metrics + team2_metrics + team1_close_ml
            ml_row_3 = team2_metrics + team1_metrics + team2_open_ml
            ml_row_4 = team2_metrics + team1_metrics + team2_close_ml

            ml_X_df.loc[ml_index] = ml_row_1
            ml_index += 1
            ml_X_df.loc[ml_index] = ml_row_2
            ml_index += 1
            ml_X_df.loc[ml_index] = ml_row_3
            ml_index += 1
            ml_X_df.loc[ml_index] = ml_row_4
            ml_index += 1
        except:
            pass

        print(f"{int(i / 2)}/{int(num_of_ids / 2)}")

    ml_X_df = ml_X_df.dropna()
    ml_X_df.to_csv("ml_X.csv")

def build_pl():
    col_names = generate_col_names(503)
    pl_X_df = pd.DataFrame(columns=col_names)
    pl_index = 0

    num_of_ids = len(every_my_id)
    for i in range(0, num_of_ids, 2):
        try:
            team1_id = every_my_id[i]
            team2_id = every_my_id[i + 1]

            team1_metrics = get_main_metrics(team1_id)
            team2_metrics = get_main_metrics(team2_id)

            team1_odds = get_puck_line_odds(team1_id)
            team2_odds = get_puck_line_odds(team2_id)

            team1_line = get_puck_line(team1_id)
            team2_line = get_puck_line(team2_id)

            pl_row_1 = team1_metrics + team2_metrics + team1_odds + team1_line
            pl_row_2 = team2_metrics + team1_metrics + team2_odds + team2_line

            pl_X_df.loc[pl_index] = pl_row_1
            pl_index += 1
            pl_X_df.loc[pl_index] = pl_row_2
            pl_index += 1
        except:
            pass

        print(f"{int(i / 2)}/{int(num_of_ids / 2)}")

    pl_X_df = pl_X_df.dropna()
    pl_X_df.to_csv("pl_X.csv")


def build_ou():
    col_names = generate_col_names(504)
    ou_X_df = pd.DataFrame(columns=col_names)
    ou_index = 0

    num_of_ids = len(every_my_id)
    for i in range(0, num_of_ids, 2):
        try:
            team1_id = every_my_id[i]
            team2_id = every_my_id[i + 1]

            team1_metrics = get_main_metrics(team1_id)
            team2_metrics = get_main_metrics(team2_id)

            team1_open_odds = get_open_ou_odds(team1_id)
            team2_open_odds = get_open_ou_odds(team2_id)
            team1_close_odds = get_close_ou_odds(team1_id)
            team2_close_odds = get_close_ou_odds(team2_id)

            team1_open_line = get_open_ou_line(team1_id)
            team2_open_line = get_open_ou_line(team2_id)
            team1_close_line = get_close_ou_line(team1_id)
            team2_close_line = get_close_ou_line(team2_id)

            ou_row_1 = team1_metrics + team2_metrics + team1_open_odds + team1_open_line
            ou_row_2 = team1_metrics + team2_metrics + team2_open_odds + team2_open_line
            ou_row_3 = team2_metrics + team1_metrics + team1_close_odds + team1_close_line
            ou_row_4 = team2_metrics + team1_metrics + team2_close_odds + team2_close_line

            ou_X_df.loc[ou_index] = ou_row_1
            ou_index += 1
            ou_X_df.loc[ou_index] = ou_row_2
            ou_index += 1
            ou_X_df.loc[ou_index] = ou_row_3
            ou_index += 1
            ou_X_df.loc[ou_index] = ou_row_4
            ou_index += 1
        except:
            pass

        print(f"{int(i / 2)}/{int(num_of_ids / 2)}")

    ou_X_df = ou_X_df.dropna()
    ou_X_df.to_csv("ou_X.csv")


build_ou()