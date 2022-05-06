from sklearn.base import TransformerMixin, clone
from enum import Enum
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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


class BetterTransformer(TransformerMixin):

    def __init__(self, column_types, numeric_transformer=StandardScaler(), category_transformer=OneHotEncoder(sparse=False)):
        self.column_types = column_types
        self.numeric_transformer = numeric_transformer
        self.category_transformer = category_transformer

    def fit(self, X):
        assert(len(self.column_types) == X.shape[1])
        transformers = []
        for i, type in enumerate(self.column_types):
            if type == NUM:
                transformer = clone(self.numeric_transformer)
                transformers.append(('num_{}'.format(i), transformer, [i]))
            elif type == CAT:
                transformer = clone(self.category_transformer)
                transformers.append(('cat_{}'.format(i), transformer, [i]))
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
    best_model = RandomForestClassifier(max_depth=8)
    return data, types, best_model


def preprocess_1471(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM]
    best_model = MLP(alpha=0.0001, hidden_layer_sizes=(100,), solver='adam', n_iter=1000)
    return data, types, best_model


def preprocess_1502(data):
    types = [NUM, NUM, NUM]
    best_model = RandomForestClassifier(max_depth=32, n_estimators=50)
    return data, types, best_model


def preprocess_40922(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM]
    best_model = RandomForestClassifier(max_depth=32)
    return data, types, best_model


def preprocess_43551(data):
    types = [NUM, NUM, NUM, NUM, NUM, NUM, NUM, NUM, CAT]
    best_model = GBDTClassifier(max_depth=3, n_estimators=20)
    return data, types, best_model


def preprocess_1590(data):
    types = [NUM, CAT, NUM, CAT, NUM, CAT, CAT, CAT, CAT, CAT, NUM, NUM, NUM, CAT]
    best_model = GBDTClassifier(max_depth=3, n_estimators=20)
    return data, types, best_model


def preprocess_41138(data):
    types = [NUM] * 170
    return data, types


def preprocess_42395(data):
    types = [DROP] + ([NUM] * 200)
    best_model = GBDTClassifier(max_depth=3, n_estimators=20)
    return data, types, best_model


def get_openml(dataset_id):
    funcs = {name:obj for name, obj in inspect.getmembers(sys.modules[__name__]) 
                if (inspect.isfunction(obj) and name.startswith('preprocess'))}

    func_name = 'preprocess_{}'.format(dataset_id)
    if not func_name in funcs:
        raise ValueError('No preprocessing found for dataset {}'.format(dataset_id))
    
    func = funcs[func_name]
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    X, types, best_model = func(X)
    transformer = BetterTransformer(types)
    y = LabelEncoder().fit_transform(y)

    return X.values, y, transformer, best_model
