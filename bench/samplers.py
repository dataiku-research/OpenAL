
from cardinal.clustering import MiniBatchKMeansSampler
from cardinal.zhdanov2019 import TwoStepKMeansSampler
import numpy as np

from cardinal.uncertainty import MarginSampler
from cardinal.clustering import IncrementalMiniBatchKMeansSampler
import copy
from tensorflow.keras.models import Sequential, Model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder


class TwoStepIncrementalMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected])
        selected = selected[new_selected]
        
        return selected


class TwoStepMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            MiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       ) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected])
        selected = selected[new_selected]
        
        return selected




class AutoEmbedder:

    ALLOWED = (MLPClassifier, Sequential, RandomForestClassifier, GradientBoostingClassifier)

    def __init__(self, model, **kwargs):

        if not isinstance(model, self.ALLOWED):
            raise TypeError('Expecting a model among ' + str(self.ALLOWED))

        if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
            if not 'X' in kwargs:
                raise ValueError('You must provide a calibration dataset for RF')
            if isinstance(model, RandomForestClassifier):
                estimators = model.estimators_
            elif isinstance(model, GradientBoostingClassifier):
                estimators = model.estimators_.reshape(-1)
                
            n_leaves = [0] + [i.get_n_leaves() for i in estimators]
            X = kwargs['X']

            self.shifts = np.cumsum(n_leaves)
            data = model.apply(X).reshape([X.shape[0], -1])
            self.ohe = OneHotEncoder().fit(data)
            transformed = self.ohe.transform(data)
            n_components = min(transformed.shape[1] - 1, int(np.log(data.shape[1]) / (0.25 ** 2)))
            self.pca = TruncatedSVD(n_components=n_components)
            self.pca.fit(transformed)

        self.model = model

    def __call__(self, X):
        if isinstance(self.model, MLPClassifier):
            return _forward_pass_partial(self.model, X, self.model.n_layers_ - 2)
        elif isinstance(self.model, Sequential):
            m = Model(inputs=self.model.input,
                      outputs=self.model.get_layer(index=len(self.model.layers) -1).output)
            m.compile()
            return m.predict(X)
        elif isinstance(self.model, RandomForestClassifier) or isinstance(self.model, GradientBoostingClassifier):
            return self.pca.transform(self.ohe.transform(self.model.apply(X).reshape((X.shape[0], -1))))
        else:
            raise ValueError('Model is not supported')
            
