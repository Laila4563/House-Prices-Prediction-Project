import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from utils.helpers import read_file
from utils.preprocessing import get_unique_values, fix_logically_missing_values, clean_missing_values

df = read_file('data/train.csv')

# df.info()

# get_unique_values(df)

fix_logically_missing_values(df)
clean_missing_values(df)