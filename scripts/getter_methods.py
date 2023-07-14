import pandas as pd
import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# dictionary to convert full names into their abbreviation
team_dict = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "St Louis Blues": "STL",

    "Pittsburgh": "PIT",
    "TampaBay": "TBL",
    "SeattleKraken": "SEA",
    "Vegas": "VGK",
    "NYRangers": "NYR",
    "Washington": "WSH",
    "Montreal": "MTL",
    "Toronto": "TOR",
    "Vancouver": "VAN",
    "Edmonton": "EDM",
    "Chicago": "CHI",
    "Colorado": "COL",
    "Winnipeg": "WPG",
    "Anaheim": "ANA",
    "Ottawa": "OTT",
    "Buffalo": "BUF",
    "Florida": "FLA",
    "NYIslanders": "NYI",
    "Carolina": "CAR",
    "Dallas": "DAL",
    "Arizona": "ARI",
    "Columbus": "CBJ",
    "Detroit": "DET",
    "NewJersey": "NJD",
    "Philadelphia": "PHI",
    "Minnesota": "MIN",
    "Boston": "BOS",
    "St.Louis": "STL",
    "Calgary": "CGY",
    "SanJose": "SJS",
    "Nashville": "NSH",
    "LosAngeles": "LAK"
}

# function needed to deal with the accent in Montreal in some dfs
def get_three_letter_code(full_name: str):
    try:
        return team_dict[full_name]
    except:
        if full_name[:5] == "Montr":
            return team_dict["Montreal Canadiens"]
        else:
            pass


def add_lists(list1, list2):
    result = []
    for item1, item2 in zip(list1, list2):
        if not isinstance(item1, str) and not isinstance(item2, str):
            result.append(item1 + item2)
    return result


def remove_strings(lst):
    return [x for x in lst if not isinstance(x, str)]


# create dataframes for all the archeived odds
def get_odds(my_id:str) -> pd.Series:
    year = int(my_id[4:8])
    if int(my_id[9:11]) >= 10:
        year += 1
    filepath = f"data/{year}odds.csv"
    df = pd.read_csv(filepath)
    date = f"{my_id[9:11]}{my_id[12:]}"
    if date[0] == str(0):
        date = date[1:]
    df = df[df["Date"] == int(date)]
    for index, row in df.iterrows():
        if get_three_letter_code(row["Team"]) == my_id[:3]:
            return row
    raise ValueError("Your my_id is incorrect.")


def get_dict(my_id:str) -> dict:
    team_dict = {}
    year = int(my_id[4:8])
    if int(my_id[9:11]) >= 10:
        year += 1
    filepath = f"data/adv_{year}.csv"
    df = pd.read_csv(filepath)
    df_groups = [group for _, group in df.groupby('Team')]

    for group in df_groups:
        abbreviation = get_three_letter_code(group['Team'].iloc[0])
        team_dict[abbreviation] = group

    return team_dict


def get_opp_score(my_id: str) -> int:
    year = int(my_id[4:8])
    if int(my_id[9:11]) >= 10:
        year += 1
    filepath = f"data/{year}odds.csv"
    df = pd.read_csv(filepath)
    date = f"{my_id[9:11]}{my_id[12:]}"
    if date[0] == str(0):
        date = date[1:]
    df = df[df["Date"] == int(date)].reset_index()
    for index, row in df.iterrows():
        if get_three_letter_code(row["Team"]) == my_id[:3]:
            if (index % 2) == 0:
                return int(df["Final"].iloc[index + 1])
            return int(df["Final"].iloc[index - 1])
    raise ValueError("Something went wrong.")


def get_open_ml(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    l1.append(int(odds.loc["Open"]))
    if int(odds.loc["Final"]) > get_opp_score(my_id):
        l1.append(1)
    else:
        l1.append(0)
    return l1


def get_close_ml(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    l1.append(int(odds.loc["Close"]))
    if int(odds.loc["Final"]) > get_opp_score(my_id):
        l1.append(1)
    else:
        l1.append(0)
    return l1


# return [line, over/under (over is 1), result]
def get_open_ou_line(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    ou = float(odds.loc["OpenOU"])
    l1.append(ou)
    total = int(odds.loc["Final"]) + int(get_opp_score(my_id))
    if odds.loc["VH"] == "V":
        l1.append(1)
        if total > ou:
            l1.append(1)
        else:
            l1.append(0)
    else:
        l1.append(0)
        if total < ou:
            l1.append(1)
        else:
            l1.append(0)
    return l1


def get_open_ou_odds(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    l1.append(odds.loc["OpenOdds"])
    return l1


# return [line, over/under (over is 1), result]
def get_close_ou_line(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    ou = float(odds.loc["CloseOU"])
    l1.append(ou)
    total = int(odds.loc["Final"]) + int(get_opp_score(my_id))
    if odds.loc["VH"] == "V":
        l1.append(1)
        if total > ou:
            l1.append(1)
        else:
            l1.append(0)
    else:
        l1.append(0)
        if total < ou:
            l1.append(1)
        else:
            l1.append(0)
    return l1


def get_close_ou_odds(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    l1.append(odds.loc["CloseOdds"])
    return l1


def get_puck_line(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    pl = float(odds.loc["PL"])
    l1.append(pl)
    if (int(odds.loc["Final"]) + float(odds.loc["PL"])) > int(get_opp_score(my_id)):
        l1.append(1)
    else:
        l1.append(0)
    return l1


def get_puck_line_odds(my_id: str) -> list:
    l1 = []
    odds = get_odds(my_id)
    l1.append(odds.loc["PLOdds"])
    return l1



def get_last_n(my_id: str, how_many: int) -> list:
    # define which dataset to look in
    season = int(my_id[4:8])

    # account for games occuring in oct-dec being different than the calendar year
    october = 10
    if int(my_id[9:11]) >= october:
        season += 1

    # create a df with only the relevant data to the team and the year
    working_dict = get_dict(my_id)
    working_df = working_dict[my_id[:3]]

    # figure out what game number the given game is
    working_df = working_df.reset_index()
    dates_list = working_df['Date'].tolist()

    if "/" in dates_list[0]:
        dates_list = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates_list]

    index = dates_list.index(my_id[4:]) - 1
    if index == -1:
        return [None] * 62

    # average the values of the last n games
    start = working_df.iloc[index].values.tolist()
    added_list = add_lists(start, [0] * len(start))
    num_to_divide_by = 1
    for i in [index - j for j in range(1, how_many)]:
        if i < 0:
            continue
        num_to_divide_by += 1
        added_list = add_lists(remove_strings(working_df.iloc[i]), added_list)

    # return the list of averaged values
    return [round(x / num_to_divide_by, 4) for x in added_list]


def get_rest_days(my_id: str) -> int:
    # define which dataset to look in
    season = int(my_id[4:8])

    # account for games occuring in oct-dec being different than the calendar year
    if int(my_id[9:11]) >= 10:
        season += 1

    # create a df with only the relevant data to the team and the year
    working_dict = get_dict(my_id)
    working_df = working_dict[my_id[:3]]

    # figure out what game number the given game is
    working_df = working_df.reset_index()
    index = working_df['Date'].tolist().index(my_id[4:])
    if index < 0:
        raise ValueError("Can't do this with first game of the season")
    last_game = working_df['Date'].iloc[index - 1]
    this_game = working_df['Date'].iloc[index]

    date_format = "%Y-%m-%d"
    date1 = datetime.strptime(last_game, date_format)
    date2 = datetime.strptime(this_game, date_format)

    delta = date2 - date1
    return delta.days


def get_year(my_id: str) -> int:
    return int(my_id[4:8])