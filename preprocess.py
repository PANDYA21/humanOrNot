import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# read in data - part-1: heros information file
heros_info = pd.read_csv('./data/heroes_information.csv')
# heros_info.head()

# read in data - part-2: heros superpowers file
heros_power = pd.read_csv('./data/super_hero_powers.csv')
# heros_power.head()

# since only binary classification is required, create a feature that defines weather the hero is human or not
heros_info['isHuman'] = heros_info['Race'].apply(lambda i: 1 if i == 'Human' else 0)
heros_info.drop(['Unnamed: 0'], inplace=True, axis=1)
# heros_info.head()

# create model matrix (aka dummies in pyworld) for superpowers data
power_bool_cols = heros_power.columns.drop("hero_names")
heros_power_dummies = pd.get_dummies(heros_power, columns=power_bool_cols)
# heros_power_dummies.head()

# model-matrix for info dataframe
info_bool_cols = ['Gender', 'Eye color', 'Hair color', 'Skin color', 'Alignment']
heros_info_dummies = pd.get_dummies(heros_info, columns=info_bool_cols)
# heros_info_dummies.head()

# merge two dataframes
heros = pd.merge(heros_info_dummies, heros_power_dummies, left_on=['name'], right_on=['hero_names'], how='inner')
# heros.head()

# create input (X) and output (y) for training a model
# drop all non-numeric collumns from input and all rows with one or more NaN/NAs
heros = heros.dropna()
print('\nComplete Cases: ')
print(heros.shape)
X_columns_drop = ['name', 'Publisher', 'hero_names', 'Race', 'isHuman']
X, y = heros.drop(X_columns_drop, axis=1), heros['isHuman']

# split dataframes into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=35)
print('Trainig size: ')
print(X_train.shape)
print('Test size: ')
print(X_test.shape)
print('\n')

