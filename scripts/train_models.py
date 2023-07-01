import pandas as pd
import numpy as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
import keras
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

ml_df = pd.read_csv("data/ml_X.csv")
ml_X = ml_df.iloc[:, -1]
ml_y = ml_df.iloc[:, :-1]

print(ml_X)
print(ml_y)
ou_X = pd.read_csv("data/ou_X.csv")
pl_X = pd.read_csv("data/pl_X.csv")