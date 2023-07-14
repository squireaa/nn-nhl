import pandas as pd
import os
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from getter_methods import *


def generate_col_names(num_needed: int) -> list:
        return ['Column' + str(i+1) for i in range(num_needed)]


def get_main_metrics(my_id: str) -> list:
    to_return = []
    to_return.extend(get_last_n(my_id, 1))
    to_return.extend(get_last_n(my_id, 3))
    to_return.extend(get_last_n(my_id, 5))
    to_return.extend(get_last_n(my_id, 10))
    to_return.extend([get_rest_days(my_id)])
    to_return.extend([get_year(my_id)])
    return to_return


def build_ml(every_my_id:list, output_name:str) -> None:
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
    ml_X_df.to_csv(output_name)


def build_pl(every_my_id:list, output_name:str) -> None:
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
    pl_X_df.to_csv(output_name)


def build_ou(every_my_id:list, output_name:str) -> None:
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
    ou_X_df.to_csv(output_name)


def format_data(input_name:str, output_name:str) -> None:
    df = pd.read_csv(input_name)

    # removes columns with an abundance of NaN
    columns_to_drop = ["GF%", "SCGF%", "HDGF%", "HDSH%", "HDSV%", "MDGF%", "MDSH%", "MDSV%", "LDGF%"]
    for column in columns_to_drop:
        df = df.drop(column, axis=1)
    
    # converts to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # uses datetype type to convert to the proper format of string
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # replaces the few straggling NaN's with 0.0, wont affect long-term results
    df = df.replace("-", 0.0)
    df.to_csv(output_name)

    # I still need something to change the last column name to "Outcome"


from typing import Callable

def build_X(input_name:str, output_name:str, which_X: Callable[[list, str], None]) -> None:
    df = pd.read_csv(input_name)
    num_rows = len(df)
    every_my_id = list()
    for i in range(num_rows):
        team_abv = get_three_letter_code(df['Team'].iloc[i])
        date = df['Date'].iloc[i]
        every_my_id.append(f"{team_abv}-{date}")
        
    which_X(every_my_id, output_name)


# build_X("data/adv_2023.csv", "new_pl_X.csv", build_pl)
# build_X("data/adv_2023.csv", "new_ou_X.csv", build_ou)

format_data("new_pl_X.csv", "2023_pl_X")
format_data("new_ou_X.csv", "2023_ou_X")