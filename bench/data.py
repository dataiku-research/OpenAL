from sklearn.base import TransformerMixin, clone
from enum import Enum
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import inspect, sys
import openml


class ColumnType(Enum):
    NUM = 1
    CAT = 2
    DROP = 3

# Make writing easier
NUM = ColumnType.NUM
CAT = ColumnType.CAT
DROP = ColumnType.DROP


def BetterTransformer(TransformerMixin):

    def __init__(self, column_types, numeric_transformer=StandardScaler(), category_transformer=OneHotEncoder(sparse=False)):
        self.column_types = column_types
        self.numeric_transformer = numeric_transformer
        self.category_transformer = category_transformer

    def fit(self, X, y=None):
        assert(len(self.column_types) == X.shape[1])
        transformers = []
        for i, type in enumerate(self.column_types):
            if type == NUM:
                transformer = clone(self.numeric_transformer)
                transformer.fit(X[:, i])
                transformers.append('num_{}'.format(i), transformer, i)
            elif type == CAT:
                transformer = clone(self.category_transformer)
                transformer.fit(X[:, i])
                transformers.append('cat_{}'.format(i), transformer, i)
            elif type == DROP:
                continue
            else:
                raise NotImplementedError('Unknow type {}'.format(type))
        self.transformer_ = ColumnTransformer(transformers)
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
    data['pdays'].replace(-1, data['pdays'].max() + 1, inplace=True)
    return data, types


def preprocess_1471(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM]
    return data, types


def preprocess_1502(data):
    types = [NUM, NUM, NUM]
    return data, types


def preprocess_40922(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM]
    return data, types


def preprocess_43551(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, CAT]
    return data, types


def preprocess_1590(data):
    types = [NUM, CAT, NUM, CAT, NUM, CAT, CAT, CAT, CAT, CAT, NUM, NUM, NUM, CAT]
    return data, types


def preprocess_41138(data):
    types = [NUM] * 170
    return data, types


def preprocess_42395(data):
    types = [DROP] + ([NUM] * 200)
    return data, types


def get_openml(dataset_id):
    funcs = {name:obj for name, obj in inspect.getmembers(sys.modules[__name__]) 
                if (inspect.isfunction(obj) and name.startwith('preprocess'))}

    func_name = 'preprocess_{}'.format(dataset_id)
    if not func_name in funcs:
        raise ValueError('No preprocessing found for dataset {}'.format(dataset_id))
    
    func = funcs[func_name]
    dataset = openml.datasets.get_dataset(43439)
    X, y, _, _ = dataset.get_data(dataset_format='dataframe')
    X, types = func(X)
    transformer = BetterTransformer(types)

    return X, y, transformer