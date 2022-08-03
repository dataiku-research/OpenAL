from sklearn.base import TransformerMixin, clone
from enum import Enum
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import inspect, sys
import openml

import time
import datetime
import pandas as pd

from tensorflow.keras.datasets import mnist, cifar10, cifar100


# https://gist.github.com/devanshuDesai/d3bdc9270395490cae3b690632445e0e
def transform_date_string_to_timestamp(date_col):
    """Convert date strings "01/12/2011" to timestamp format"""
    # date_col['Date'] = date_col['Date'].apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), "%d/%m/%Y").timetuple()) if x is not None else None)
    for sample in date_col:
        sample[0] = time.mktime(datetime.datetime.strptime(str(sample[0]), "%d/%m/%Y").timetuple()) if sample[0] is not None else None
    return date_col

def transform_time_string_to_timestamp(time_col):
    """Convert time strings "18:11" to timestamp format"""
    # time_col['Time'] =  time_col['Time'].apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), "%H:%M").timetuple()) if x is not None else None)
    for sample in time_col:
        sample[0] = time.mktime(datetime.datetime.strptime(str(sample[0]), "%H:%M").timetuple()) if sample[0] is not None else None
    return time_col

def transform_date2_string_to_timestamp(date_col):
    """Convert date strings "2016-04-29T18:38:08Z" to timestamp format"""
    # date_col['Date'] = date_col['Date'].apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), "%d/%m/%Y").timetuple()) if x is not None else None)
    for sample in date_col:
        sample[0] = time.mktime(datetime.datetime.strptime(str(sample[0]), "%Y-%m-%dT%H:%M:%SZ").timetuple()) if sample[0] is not None else None
    return date_col

# def transform_flatten_MNIST(X):
#     return X.astype('float32').reshape((X.shape[0], -1)) / 255.0

# def transform_CIFAR10(X):
#     #TODO 32x32 RGB image
#     return X

# def transform_CIFAR100(X):
#     #TODO 
#     return X


class ColumnType(Enum):
    NUM = 1
    CAT = 2
    DROP = 3
    DATE = 4
    TIME = 5
    DATE2 = 6

# Make writing easier
NUM = ColumnType.NUM
CAT = ColumnType.CAT
DROP = ColumnType.DROP
DATE = ColumnType.DATE
TIME = ColumnType.TIME
DATE2 = ColumnType.DATE2


class BetterTransformer(TransformerMixin):

    def __init__(self, column_types, numeric_transformer=StandardScaler(), category_transformer=OneHotEncoder(sparse=False)):
        self.column_types = column_types
        self.numeric_transformer = numeric_transformer
        self.category_transformer = category_transformer

    def fit(self, X, full_X=None):
        assert(len(self.column_types) == X.shape[1])
        transformers = []
        for i, type in enumerate(self.column_types):
            if type == NUM:
                transformer = clone(self.numeric_transformer)
                transformers.append(('num_{}'.format(i), transformer, [i]))
            elif type == CAT:
                if full_X is None:
                    # transformer = self.category_transformer()
                    transformer = Pipeline([
                                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                                    ('OneHotEncoder', OneHotEncoder(sparse=False))
                                    ])
                                    #clone(self.category_transformer)
                else:
                    transformer = Pipeline([
                                        ('Imputer', SimpleImputer(strategy='most_frequent')),   #TODO : stratégie discutable
                                        ('OneHotEncoder', OneHotEncoder(sparse=False, categories=[np.unique(full_X[:, i][~pd.isna(full_X[:, i])])]))
                                        ]) 
                                        #clone(self.category_transformer(categories=[np.unique(full_X[:, i])]))
                transformers.append(('cat_{}'.format(i), transformer, [i]))
            elif type == DATE:
                transformer = Pipeline([('convert_date', FunctionTransformer(transform_date_string_to_timestamp)),
                                        ('Imputer', SimpleImputer(strategy='median')),
                                        ('Scaler', StandardScaler())])
                transformers.append(('date_{}'.format(i), transformer, [i]))
            elif type == TIME:
                transformer = Pipeline([('convert_date', FunctionTransformer(transform_time_string_to_timestamp)),
                                        ('Imputer', SimpleImputer(strategy='median')),
                                        ('Scaler', StandardScaler())]) 
                transformers.append(('date_{}'.format(i), transformer, [i]))
            elif type == DATE2:
                # TODO   .weekday()
                transformer = Pipeline([('convert_date2', FunctionTransformer(transform_date2_string_to_timestamp)),
                                        ('Scaler', StandardScaler())]) 
                transformers.append(('date_{}'.format(i), transformer, [i]))
            elif type == DROP:
                continue
            else:
                raise NotImplementedError('Unknow type {}'.format(type))
        self.transformer_ = ColumnTransformer(transformers)
        self.transformer_.fit(X)
        return self
    
    def transform(self, X):
        return self.transformer_.transform(X)

    def inverse_transform(self, X):
        Xt = np.empty((X.shape[0], len(self.transformer.transformers_)))
        for i, (name, transformer, _) in enumerate(self.transformer_.transformers_):
            Xt[:, i] = transformer.inverse_transform(X[self.transformer_.output_indices[name]])

        return Xt


def preprocess_1461(data):
    types = [NUM, CAT, CAT, CAT, CAT, NUM, CAT, CAT, CAT, NUM, CAT, NUM, NUM, NUM, NUM, CAT]
    data['V14'].replace(-1, data['V14'].max() + 1, inplace=True)
    best_model = lambda seed: RandomForestClassifier(max_depth=8, random_state=seed)
    return data, types, best_model


def preprocess_1471(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM]
    best_model = lambda seed: MLPClassifier(alpha=0.0001, hidden_layer_sizes=(100,), solver='adam', max_iter=2000, random_state=seed)
    return data, types, best_model


def preprocess_1502(data):
    types = [NUM, NUM, NUM]
    best_model = lambda seed: RandomForestClassifier(max_depth=32, n_estimators=50, random_state=seed)
    return data, types, best_model


def preprocess_1590(data):
    types = [NUM, CAT, NUM, CAT, NUM, CAT, CAT, CAT, CAT, CAT, NUM, NUM, NUM, CAT]
    best_model = lambda seed: GradientBoostingClassifier(max_depth=3, n_estimators=20, random_state=seed)
    return data, types, best_model


def preprocess_40922(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM]
    best_model = lambda seed: RandomForestClassifier(max_depth=32, random_state=seed)
    return data, types, best_model


def preprocess_41138(data):
    types = [NUM] * 170
    best_model = lambda seed: RandomForestClassifier(max_depth=8, random_state=seed)
    transformer = BetterTransformer(
        types,
        numeric_transformer=Pipeline([('Imputer', SimpleImputer(strategy='median')), ('Scaler', StandardScaler())]))
    return data, types, best_model, transformer


def preprocess_41162(data):
    #Problème : problème d'encoding d'une valeur numérique (col 0) qui apparait dans le train mais pas dans le test ... PAS NORMAL CAR NUMERIQUE
    types = [NUM, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, NUM, CAT, CAT, CAT, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, CAT, CAT, CAT, CAT, CAT, NUM, CAT, NUM]    # 3ème et 4ème features ajoutées en catégorielles
    best_model = lambda seed: GradientBoostingClassifier(max_depth=3, n_estimators=100, random_state=seed)
    transformer = BetterTransformer(
        types,
        numeric_transformer=Pipeline([('Imputer', SimpleImputer(strategy='median')), ('Scaler', StandardScaler())]))
    return data, types, best_model, transformer


def preprocess_42395(data):
    types = [DROP] + ([NUM] * 200)
    best_model = lambda seed: GradientBoostingClassifier(max_depth=3, n_estimators=20, random_state=seed)
    return data, types, best_model


def preprocess_42803(data):
    types = [DROP, CAT,  CAT, CAT, CAT, NUM, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, NUM, CAT, NUM, CAT,    NUM, CAT, NUM, NUM, NUM, NUM, CAT, CAT, CAT, CAT, DATE, CAT, TIME,      NUM        , DROP, CAT, NUM, CAT, CAT, CAT, CAT, CAT, NUM, CAT, CAT, CAT, CAT, CAT, NUM, NUM, CAT, CAT, DROP, CAT, CAT, CAT, CAT, NUM, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT]  #1rst DROP replace NUM, 2nd DROP replace NUM (Local_Authority_(Highway)), 3rd DROP replace NUM (LSOA_of_Accident_Location), NUM (31) (Local_Authority_(District) CAT -> NUM)
    best_model = lambda seed: GradientBoostingClassifier(max_depth=8, n_estimators=100, random_state=seed)
    transformer = BetterTransformer(
        types,
        numeric_transformer = Pipeline([('Imputer', SimpleImputer(strategy='median')), ('Scaler', StandardScaler())]))
    return data, types, best_model, transformer


def preprocess_43439(data):
    #TODO : extract hours and day of week more
    # .weekday()

    # Adding new features to the dataset (extract hours and day of week more)
    # date_col = data['date']
    # weekday_col = transform_date2_string_to_timestamp(date_col)

    types = [DROP, CAT, DATE2, DATE2, CAT, CAT, CAT, CAT, CAT, CAT, CAT, CAT] 
    best_model = lambda seed: GradientBoostingClassifier(max_depth=8, n_estimators=20, random_state=seed)
    return data, types, best_model


def preprocess_43551(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, CAT]
    best_model = lambda seed: GradientBoostingClassifier(max_depth=3, n_estimators=20, random_state=seed)
    return data, types, best_model

def preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    X = X.astype('float32').reshape((X.shape[0], -1)) / 255.0

    best_model = lambda seed: MLPClassifier(random_state=seed)    #TODO
    # transformer = Pipeline([('MNIST preprocessing', FunctionTransformer(transform_flatten_MNIST))]) #TODO class transformer for images

    return X, y, best_model#, transformer

def preprocess_cifar10():
    best_model = lambda seed: MLPClassifier(random_state=seed)    #TODO
    folder_path = "/data.nfs/data_al/cifar10/"

    # Embeddings from ImageNet
    X = np.load(folder_path+'cifar_embeddings.npy')
    y = np.load(folder_path+'cifar_target.npy')

    # Transformation to fit with the script (will be transformed later)
    y = np.argmax(y,axis=1)

    return X, y, best_model#, transformer

def preprocess_cifar10_simclr():
    best_model = lambda seed: MLPClassifier(random_state=seed)
    folder_path = "/data.nfs/data_al/cifar10/"

    # Embeddings from contrastive learning
    X = np.load(folder_path+'simclr_embed.npy')
    y = np.load(folder_path+'simclr_labels.npy')

    return X, y, best_model

def preprocess_cifar100():
    best_model = lambda seed: MLPClassifier(random_state=seed)
    folder_path = "/data.nfs/data_al/cifar100/"

    # Embeddings from ImageNet
    X = np.load(folder_path+'cifar_embeddings.npy')
    y = np.load(folder_path+'cifar_target.npy')

    # Transformation to fit with the script (will be transformed later)
    y = np.argmax(y,axis=1)

    return X, y, best_model#, transformer

def preprocess_cifar100_simclr():
    best_model = lambda seed: MLPClassifier(max_iter=2000, random_state=seed)
    folder_path = "/data.nfs/data_al/cifar100/"

    # Embeddings from contrastive learning
    X = np.load(folder_path+'simclr_embed.npy')
    y = np.load(folder_path+'simclr_labels.npy')

    return X, y, best_model


def get_openml(dataset_id):
    funcs = {name:obj for name, obj in inspect.getmembers(sys.modules[__name__]) 
                if (inspect.isfunction(obj) and name.startswith('preprocess'))}

    func_name = 'preprocess_{}'.format(dataset_id)
    if not func_name in funcs:
        raise ValueError('No preprocessing found for dataset {}'.format(dataset_id))
    
    func = funcs[func_name]
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    preproc = func(X)
    if len(preproc) == 3:
        X, types, best_model = preproc
        transformer = BetterTransformer(types)
    else:
        X, types, best_model, transformer = preproc
    y = LabelEncoder().fit_transform(y)

    # print(X)

    return X.values, y, transformer, best_model


def get_image_dataset(dataset_id):
    funcs = {name:obj for name, obj in inspect.getmembers(sys.modules[__name__]) 
                if (inspect.isfunction(obj) and name.startswith('preprocess'))}

    func_name = 'preprocess_{}'.format(dataset_id)
    if not func_name in funcs:
        raise ValueError('No preprocessing found for dataset {}'.format(dataset_id))
    
    func = funcs[func_name]

    preproc = func()
    X, y, best_model = preproc

    y = LabelEncoder().fit_transform(y)

    return X, y, best_model


def get_dataset(dataset_id):
    if dataset_id in ['mnist','cifar10','cifar100','cifar10_simclr','cifar100_simclr']:
        return get_image_dataset(dataset_id)
    else:
        return get_openml(int(dataset_id))
