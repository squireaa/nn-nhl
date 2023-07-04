import pandas as pd
from datetime import datetime
from random import randint
from time import sleep
import os


def get_live_metrics(date:str) -> None:
    year = int(date[:4])
    if int(date[5:7]) >= 10:
        year += 1
    date = datetime.strptime(date, "%Y-%m-%d")
    # creates a df based on the url given
    url = f"https://www.hockey-reference.com/leagues/NHL_{year}_games.html"
    dfs = pd.read_html(url)
    df = dfs[0]

    output_path = "nf_adv_{0}.csv".format(year)

    # defines the url needed for the specific date
    nhl_url = f"https://www.naturalstattrick.com/teamtable.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=ev&score=all&rate=y&team=all&loc=B&gpf=410&fd={date.strftime('%Y-%m-%d')}&td={date.strftime('%Y-%m-%d')}"

    # saves the advanced data to a df
    nhl_dfs = pd.read_html(nhl_url, index_col=0)
    nhl_df = nhl_dfs[0]

    # adds a date column to the df and saves it
    nhl_df['Date'] = date

    # gets rid of the rows that had an invalid date
    nhl_df = nhl_df[nhl_df["GP"] != 82]
    nhl_df.to_csv(output_path, mode='a', header= not os.path.exists(output_path), index = False)

# get_live_metrics("2023-07-04")