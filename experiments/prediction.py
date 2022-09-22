# import sys
# sys.path.append('..')
# sys.path.append('.')

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from enum import Enum
# from cardinal.plotting import plot_confidence_interval
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
# from bench.data import BetterTransformer


# class ColumnType(Enum):
#     NUM = 1
#     CAT = 2
#     DROP = 3

# # Make writing easier
# NUM = ColumnType.NUM
# CAT = ColumnType.CAT
# DROP = ColumnType.DROP


metrics = [
    ('accuracy','accuracy_test.csv'),
    ('agreement','agreement_test.csv'),
    ('trustscore','test_trustscore.csv'),
    ('violation','test_violation.csv'),
    ('exploration','soft_exploration.csv'),
    ('closest','this_closest.csv')
]

# Import and join data

dataset_id = 1471
df = pd.read_csv('results_{}/'.format(dataset_id)+'accuracy_test.csv')
df.rename(columns = {'value':'accuracy'}, inplace = True)

for i, (metric_name, filename) in enumerate(metrics):
    df_new = pd.read_csv('results_{}/'.format(dataset_id)+filename)
    df['{}'.format(metric_name)]=df_new['value'].values
    # df = pd.merge(df,df_new, on=['seed', 'method', 'n_iter', 'dataset'], how='inner', suffixes=('_test', '_{}'.format('agreement'))) 

X_transformer = ColumnTransformer(transformers =[ 
    ('num', StandardScaler(), ['agreement', 'trustscore', 'violation', 'exploration', 'closest']), 
    ('cat', OneHotEncoder(), ['dataset', 'method'])
], remainder ='drop') 
y_transformer = ColumnTransformer(transformers =[('num', StandardScaler(), ['accuracy'])], remainder ='drop')  



# GET SCORES
y_df = pd.DataFrame({'accuracy': df['accuracy'].values})   # y = df['accuracy']
X_df = df.drop(labels=['seed', 'n_iter', 'accuracy'], axis=1)

# print(X_df.head())

X = X_transformer.fit_transform(X_df)
y = y_transformer.fit_transform(y_df)
y = y.reshape((y.shape[0]))
# print(X.shape, y.shape)


# kf = KFold(n_splits=5, shuffle=True, random_state=0)
# for i, (ind_learn, ind_test) in enumerate(kf.split(X, y)):
# for i in range(5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     # print('Iteration {}'.format(i))

#     # X_train, X_test = X[ind_learn], X[ind_test]
#     # y_train, y_test = y[ind_learn], y[ind_test]

#     model = MLPRegressor(max_iter=2000)

#     model.fit(X_train, y_train)
#     print('[SCORE] iter {}'.format(i),model.score(X_test,y_test))




# PLOTS
df_train, df_test = train_test_split(df, test_size=0.2)

# y_df_train = pd.DataFrame({'accuracy': df_train['accuracy'].values, 'seed': df_train['seed'].values, 'n_iter': df_train['n_iter'].values})
# y_df_test = pd.DataFrame({'accuracy': df_test['accuracy'].values, 'seed': df_test['seed'].values, 'n_iter': df_test['n_iter'].values})
y_df_train = df_train.drop(labels=['method', 'agreement','trustscore', 'violation', 'exploration', 'closest'], axis=1)
y_df_test = df_test.drop(labels=['method', 'agreement','trustscore', 'violation', 'exploration', 'closest'], axis=1)

X_df_train = df_train.drop(labels=['accuracy'], axis=1) #'seed', 'n_iter',
X_df_test = df_test.drop(labels=['accuracy'], axis=1) #'seed', 'n_iter',
print("train", X_df_train.shape, y_df_train.shape)
print("test", X_df_test.shape, y_df_test.shape)

# fit transformers
_ = X_transformer.fit(X_df_train)
_ = y_transformer.fit(y_df_train)

for seed in range(10):
    X_df_train_seed = X_df_train[X_df_train['seed'] == seed]
    y_df_train_seed = y_df_train[y_df_train['seed'] == seed]
    X_df_test_seed = X_df_test[X_df_test['seed'] == seed]
    y_df_test_seed = y_df_test[y_df_test['seed'] == seed]

    X_train = X_transformer.transform(X_df_train_seed)
    y_train = y_transformer.transform(y_df_train_seed).reshape((y_df_train_seed.shape[0]))
    X_test = X_transformer.transform(X_df_test_seed)
    y_test = y_transformer.transform(y_df_test_seed).reshape((y_df_test_seed.shape[0]))

    model = MLPRegressor(max_iter=2000)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    plt.plot(X_df_test_seed['n_iter'].values, prediction, label='prediction')
    plt.plot(X_df_test_seed['n_iter'].values, y_df_test_seed['accuracy'].values, label='true')
    plt.legend()
    plt.show()
