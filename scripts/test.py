import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

df = pd.read_csv("data/adv_2023.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.to_csv("data/adv_2023.csv", index=False)

"""
how to format the data:
get rid of the columns:
    GF%
    SCGF%
    HDGF%
    HDSH%
    HDSV%
    MDGF%
    MDSH%
    MDSV%
    LDGF%
format the date to yyyy-mm-dd
replace the few extra "-" with 0's
make sure there is a blank index column at the very start


"""