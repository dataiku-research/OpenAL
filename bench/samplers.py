from .trustscore import TrustScore
from cardinal.clustering import KMeansSampler, KCentroidSampler, MiniBatchKMeansSampler
from cardinal.utils import ActiveLearningSplitter
from cardinal.zhdanov2019 import TwoStepKMeansSampler
import numpy as np
from cardinal.base import ScoredQuerySampler, BaseQuerySampler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from cardinal.uncertainty import MarginSampler, margin_score, EntropySampler, ConfidenceSampler, ScoredQuerySampler, check_proba_estimator, _get_probability_classes
from cardinal.clustering import IncrementalMiniBatchKMeansSampler
import copy
from torch import tensor
from sklearn.metrics import confusion_matrix
from cardinal.typeutils import RandomStateType, check_random_state
from tensorflow.keras.models import Sequential, Model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer


class TwoStepIncrementalMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       fixed_cluster_centers=None) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected], fixed_cluster_centers=fixed_cluster_centers)
        selected = selected[new_selected]
        
        return selected


class TwoStepDoubtfulIncrementalMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.classifier = classifier
        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def fit(self, X, y):
        super().fit(X, y)
        y_pred = self.classifier.predict(X)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=1)
        self.good_centers = X[y_pred == y]

    def select_samples(self, X: np.array,
                       fixed_cluster_centers=None) -> np.array:
        margin = self.sampler_list[0]
        selected = margin.select_samples(X)
        threshold = margin.sample_scores_[selected].min()
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected], fixed_cluster_centers=self.good_centers)
        selected = selected[new_selected]
        
        return selected


class TwoStepNoBetaIncrementalMiniBatchKMeansSampler(BaseQuerySampler):
    def __init__(self, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.classifier = classifier
        self.sampler = IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X = X
        self.y = y

    def select_samples(self, X: np.array,
                       fixed_cluster_centers=None) -> np.array:
        margins = margin_score(self.classifier, X)
        fc_margins = margin_score(self.classifier, fixed_cluster_centers)
        threshold = fc_margins.max()
        mask = margins > threshold
        if mask.sum() <= self.batch_size:
            print('Beta: 1')
            return np.argsort(margins)[-self.batch_size:]
        print('Beta: {}'.format(mask.sum() / self.batch_size))
        new_selected = self.sampler.select_samples(
            X[mask], sample_weight=margins[mask], fixed_cluster_centers=fixed_cluster_centers)
        selected = np.arange(X.shape[0])[mask][new_selected]
        
        return selected


def to_classes(data):
    if len(data.shape) == 2:
        return np.argmax(data, axis=1)
    return data


class AdaptiveSampler2:
    def __init__(self, batch_size, exploration_samplers, exploitation_samplers):
        self.exploration_samplers = exploration_samplers
        self.exploitation_samplers = exploitation_samplers
        self.batch_size = batch_size

    def fit(self, X, y):
        for sampler in self.exploration_samplers:
            sampler.fit(X, y)
        for sampler in self.exploitation_samplers:
            sampler.fit(X, y)
        return self        

    def select_samples(self, splitter, clf, X, y):

        X_test = X[splitter.test]
        y = to_classes(y)

        knn = KNeighborsClassifier(1)
        knn.fit(X[splitter.selected], y[splitter.selected])

        def select_samples(sampler):
            if isinstance(sampler, TwoStepIncrementalMiniBatchKMeansSampler):
                return sampler.select_samples(X[splitter.non_selected], fixed_cluster_centers=X[splitter.selected])
            return sampler.select_samples(X[splitter.non_selected])

        exploration_batches = [select_samples(sampler) for sampler in self.exploration_samplers]
        exploitation_batches = [select_samples(sampler) for sampler in self.exploitation_samplers]

        r_agreements = []
        r_exploration = []
        for batch in exploration_batches:
            idx = splitter.dereference_batch_indices(batch)
            
            r_agreements.append((to_classes(clf.predict(X[idx])) == knn.predict(X[idx])).sum() / idx.shape[0])

            splitter_batch = copy.deepcopy(splitter)
            knn_batch = KNeighborsClassifier(1)
            splitter_batch.add_batch(batch)
            knn_batch.fit(X[splitter_batch.selected], y[splitter_batch.selected])
            assert(splitter_batch.selected.sum() > splitter.selected.sum())
            r_exploration.append((knn.predict(X_test) == knn_batch.predict(X_test)).sum() / X_test.shape[0])

        i_agreements = []
        i_exploration = []
        for batch in exploitation_batches:
            idx = splitter.dereference_batch_indices(batch)
            i_agreements.append((to_classes(clf.predict(X[idx])) == knn.predict(X[idx])).sum() / idx.shape[0])

            splitter_batch = copy.deepcopy(splitter)
            knn_batch = KNeighborsClassifier(1)
            splitter_batch.add_batch(batch)
            knn_batch.fit(X[splitter_batch.selected], y[splitter_batch.selected])
            assert(splitter_batch.selected.sum() > splitter.selected.sum())
            i_exploration.append((knn.predict(X_test) == knn_batch.predict(X_test)).sum() / X_test.shape[0])

        # Best explorer
        best_r_explorer = np.argmin(r_exploration)
        best_i_explorer = np.argmin(i_exploration)

        # Best exploiter
        best_r_exploiter = np.argmax(r_agreements)
        best_i_exploiter = np.argmax(i_agreements)

        print(r_exploration, i_exploration)
        print(r_agreements, i_agreements)

        # First, are exploitation good enough? This is a heuristic, we should find a better rule
        if i_agreements[best_i_exploiter] < 0.25:
            # Here, we take the best explorer, no matter the method
            if i_exploration[best_i_explorer] > r_exploration[best_r_explorer]:
                print('A exploit', best_i_explorer)
                return exploitation_batches[best_i_explorer]
            print('A explore', best_r_explorer)
            return exploration_batches[best_r_explorer]
        if  1. - (r_exploration[best_r_explorer] / i_exploration[best_i_explorer]) < 0.01:
            print('B exploit', best_i_exploiter)
            return exploitation_batches[best_i_exploiter]
        else:
            print('B explore', best_r_explorer)
            return exploration_batches[best_r_explorer]

        print('C exploit', best_i_exploiter)
        return exploitation_batches[best_i_exploiter]


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


class TwoStepRandomSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            MyRandomSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       selected=None) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected])
        selected = selected[new_selected]
        
        return selected


class Experimental(ScoredQuerySampler):
    def __init__(self, classifier, batch_size: int,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True

    def fit(self, X: np.array, y: np.array) -> 'ConfidenceSampler':
        """Fit the estimator on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        self.X_train = X
        self.y_train = y
        return self

    def score_samples(self, X: np.array) -> np.array:
        """Selects the samples to annotate from unlabeled data.
        Args:
            X: shape (n_samples, n_features), Samples to evaluate.
        Returns:
            The score of each sample according to lowest confidence estimation.
        """
        u_scores = margin_score(self.classifier_, X)
        dist = pairwise_distances(self.X_train, X)
        d = get_min_dist_per_class(dist, self.y_train)
        # Normalize rows to sum 1
        d = d / np.linalg.norm(d, ord=1, axis=1, keepdims=True)
        d = (1 - d)
        d_scores = margin_score('precomputed', d)
        scores = (u_scores + d_scores) / 2.
        return scores


class AdaptiveSampler3:
    def __init__(self, batch_size, samplers):
        self.samplers = samplers
        self.batch_size = batch_size

    def fit(self, X, y):
        for sampler in self.samplers:
            sampler.fit(X, y)
        return self        

    def select_samples(self, splitter, clf, X, y):

        X_test = X[splitter.test]

        knn = KNeighborsClassifier(1)
        knn.fit(X[splitter.selected], y[splitter.selected])

        def select_samples(sampler):
            if isinstance(sampler, TwoStepIncrementalMiniBatchKMeansSampler):
                return sampler.select_samples(X[splitter.non_selected], fixed_cluster_centers=X[splitter.selected])
            return sampler.select_samples(X[splitter.non_selected])

        batches = [select_samples(sampler) for sampler in self.samplers]

        agreements = []
        exploration = []
        for batch in batches:
            idx = splitter.dereference_batch_indices(batch)
            agreements.append((clf.predict(X[idx]) == knn.predict(X[idx])).sum())
            exploration.append(exploration_score(X[splitter.selected], X_test, X_batch=X[idx]))

        best_explorer, second_best_explorer = np.argsort(exploration)[:2]
        best_exploiter = np.argmax(agreements)

        print('Exploration', exploration)
        print('Agreement', agreements)

        # First, are exploitation good enough? This is a heuristic, we should find a better rule
        if agreements[best_exploiter] < self.batch_size / 4:
            # Here, we take the best explorer, no matter the method
            print('A', best_explorer)
            return batches[best_explorer]
        if  1. - (exploration[best_explorer] / exploration[second_best_explorer]) < 0.01:
            print('B', best_explorer)
            return batches[best_explorer]

        print('C', best_exploiter)
        return batches[best_exploiter]


class AdaptiveSampler4:
    def __init__(self, batch_size, samplers):
        self.samplers = samplers
        self.batch_size = batch_size

    def fit(self, X, y):
        for sampler in self.samplers:
            sampler.fit(X, y)
        return self        

    def select_samples(self, splitter, clf, X, y):

        X_test = X[splitter.test]

        knn = KNeighborsClassifier(1)
        knn.fit(X[splitter.selected], y[splitter.selected])

        def select_samples(sampler):
            if isinstance(sampler, TwoStepIncrementalMiniBatchKMeansSampler):
                return sampler.select_samples(X[splitter.non_selected], fixed_cluster_centers=X[splitter.selected])
            return sampler.select_samples(X[splitter.non_selected])

        batches = [select_samples(sampler) for sampler in self.samplers]

        agreements = []
        exploration = []
        for batch in batches:
            idx = splitter.dereference_batch_indices(batch)
            agreements.append((clf.predict(X[idx]) == knn.predict(X[idx])).sum())
            exploration.append(exploration_score(X[splitter.selected], X_test, X_batch=X[idx]))

        best_explorer, second_best_explorer = np.argsort(exploration)[:2]
        best_exploiter = np.argmax(agreements)

        print('Exploration', exploration)
        print('Agreement', agreements)

        # First, are exploitation good enough? This is a heuristic, we should find a better rule
        if agreements[best_exploiter] < self.batch_size / 4:
            # Here, we take the best explorer, no matter the method
            print('A', best_explorer)
            return batches[best_explorer]
        if  1. - (exploration[best_explorer] / exploration[second_best_explorer]) < 0.01:
            print('B', best_explorer)
            return batches[best_explorer]

        print('C', best_exploiter)
        return batches[best_exploiter]

        
class BatchBALDSampler:

    def __init__(self, n_classes, batch_size, clf, n_estimators, n_samples=None):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.clf = clf
        if n_samples is None:
            n_samples = int(40000 / n_classes)
        self.n_samples = n_samples
        if hasattr(clf, 'estimators_'):
            self.n_estimators = len(clf.estimators_)

    def fit(self, X: np.array, y: np.array) -> 'BatchBALDSampler':
        pass

    def select_samples(self, X, y=None):
        pred_N_K_C = np.empty((X.shape[0], self.n_estimators, self.n_classes))

        if self.clf.__class__.__name__ == 'Sequential':
            from keras.layers import Dropout
            # Check that there is a dropout layer
            for layer in self.clf.layers:
                if isinstance(layer, Dropout):
                    break
            else:
                raise ValueError('Cannot use BatchBALD without dropout layer')
            
            for i in range(self.n_estimators):
                pred_N_K_C[:, i, :] = self.clf(X, training=True)  # training=True enables dropout
        
        elif hasattr(self.clf, 'estimators_'):
            # sklearn's RandomForest or other
            for i in range(self.n_estimators):
                pred_N_K_C[:, i, :] = self.clf.estimators_[i].predict_proba(X)
        else:
            raise ValueError('Model not supported')

        from batchbald_redux.batchbald import get_batchbald_batch
        candidates = get_batchbald_batch(tensor(pred_N_K_C), self.batch_size, self.n_samples)
        return np.asarray(candidates.indices)


class InformedConfidenceSampler(ConfidenceSampler):
    def score_samples(self, X: np.array, max_confidence:np.array) -> np.array:
        score = super(InformedConfidenceSampler, self).score_samples(X)
        return np.clip(score - max_confidence, 0, None)

    def select_samples(self, X: np.array, max_confidence:np.array) -> np.array:
        sample_scores = self.score_samples(X, max_confidence)
        self.sample_scores_ = sample_scores
        index = np.argsort(sample_scores)[-self.batch_size:]
        return index

class InformedMarginSampler(MarginSampler):
    def score_samples(self, X: np.array, max_margin:np.array) -> np.array:
        score = super(InformedMarginSampler, self).score_samples(X)
        return np.clip(score - max_margin, 0, None)

    def select_samples(self, X: np.array, max_margin:np.array) -> np.array:
        sample_scores = self.score_samples(X, max_margin)
        self.sample_scores_ = sample_scores
        index = np.argsort(sample_scores)[-self.batch_size:]
        return index

class InformedEntropySampler(EntropySampler):
    def score_samples(self, X: np.array, max_entropy:np.array) -> np.array:
        score = super(InformedEntropySampler, self).score_samples(X)
        return np.clip(score - max_entropy, 0, None)

    def select_samples(self, X: np.array, max_entropy:np.array) -> np.array:
        sample_scores = self.score_samples(X, max_entropy)
        self.sample_scores_ = sample_scores
        index = np.argsort(sample_scores)[-self.batch_size:]
        return index

class InformedIWkmeans(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            InformedMarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array, max_margin:np.array,
                       fixed_cluster_centers=None) -> np.array:
        selected = self.sampler_list[0].select_samples(X, max_margin)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected], fixed_cluster_centers=fixed_cluster_centers)
        selected = selected[new_selected]
        
        return selected

class KappaSampler1(ScoredQuerySampler):

    def __init__(self, classifier, knn_classifier, batch_size: int,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.classifier_ = classifier
        self.knn_classifier = knn_classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.array, y: np.array) -> 'KappaSampler1':
        """Fit the estimator on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        return self

    def score_samples(self, X: np.array) -> np.array:

        knn_preds = _get_probability_classes(self.knn_classifier, X)
        clf_preds = _get_probability_classes(self.classifier_, X)

        # I take as uncertainty score the difference in prediction between 2 max probas
        a = knn_preds.max(axis=1) - clf_preds[np.arange(X.shape[0]), np.argmax(knn_preds, axis=1)]
        b = clf_preds.max(axis=1) - knn_preds[np.arange(X.shape[0]), np.argmax(clf_preds, axis=1)]
        s = (np.abs(a) + np.abs(b)) / 2.

        return s


class KappaSampler2(ScoredQuerySampler):

    def __init__(self, classifier, batch_size: int,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.array, y: np.array) -> 'KappaSampler2':
        """Fit the estimator on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        self._nn = NearestNeighbors(n_neighbors=1).fit(X)
        self._X_preds = _get_probability_classes(self.classifier_, X)
        return self


    def score_samples(self, X: np.array) -> np.array:
        clf_preds = _get_probability_classes(self.classifier_, X)
        knn_preds = self._X_preds[self._nn.kneighbors(X)[1][:, 0]]

        # I take as uncertainty score the difference in prediction between 2 max probas
        a = knn_preds.max(axis=1) - clf_preds[np.arange(X.shape[0]), np.argmax(knn_preds, axis=1)]
        b = clf_preds.max(axis=1) - knn_preds[np.arange(X.shape[0]), np.argmax(clf_preds, axis=1)]
        s = (np.abs(a) + np.abs(b)) / 2.

        return s


class KappaSampler3(ScoredQuerySampler):

    def __init__(self, classifier, batch_size: int,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.array, y: np.array) -> 'KappaSampler3':
        """Fit the estimator on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        return self


    def score_samples(self, X: np.array) -> np.array:
        clf_proba = _get_probability_classes(self.classifier_, X)
        clf_preds = np.argmax(clf_proba, axis=1)
        clf_preds_proba = np.max(clf_proba, axis=1)
        nn = NearestNeighbors(n_neighbors=20).fit(X)
        kn = nn.kneighbors(X)[1]
        max_preds = clf_preds_proba[kn].max(axis=1)

        return max_preds - clf_preds


class ErrorSampler1(BaseQuerySampler):

    def __init__(self, classifier, batch_size: int,
                 assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.array, y: np.array) -> 'ErrorSampler1':
        """Fit the estimator on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        self._cm = confusion_matrix(np.argmax(y, axis=1), _get_probability_classes(self.classifier_, X).argmax(axis=1))
        return self


    def select_samples(self, X: np.array) -> np.array:
        # We want to select samples proportionally to the errors of the model
        # First we nullify the diagonal of well classified samples in the CM
        cm = self._cm.copy()
        np.fill_diagonal(cm, -1)
        # Get the sorted list of weights and their indices
        cm_w = np.sort(cm, axis=None)
        # We take the argsort of the matrix, find 2d indices, transpose and revert them
        cm_i = np.asarray(np.unravel_index(np.argsort(cm, axis=None), cm.shape)).T
        # Let's remove the first ones that correspond to the diagonals and in general those below 0
        cm_i = cm_i[cm_w > 0].tolist()
        cm_w = cm_w[cm_w > 0]

        # Normalize, cumsum, in order to know how many samples we want from each bucket
        cm_w = np.cumsum(cm_w / cm_w.sum() * self.batch_size).astype(int)
        # Now for each bucket, we can select up to number in the array so if a
        # bucket does not have enough samples, the next one can fill it in

        # Let's sort the current unlabeled samples in buckets
        clf_proba = _get_probability_classes(self.classifier_, X)
        # Get the 2 most probable classes for each sample
        top_ranks = np.argsort(-clf_proba)[:, :2]
        # Compute margin score
        scores = margin_score(self.classifier_, X)
        
        all_selected = []
        all_scores = []
        for b_rank, b_nsamples in zip(cm_i, cm_w):
            # We select the samples in the unlebled pool that have the same
            # top ranks. Note that this could be sped up a bit using unique
            # and return inverse but the code would be overly complicated wrt gain
            to_select = b_nsamples - len(all_selected)
            b_ind = (top_ranks == b_rank).all(axis=1).nonzero()[0]
            b_scores = scores[b_ind]
            b_selected = b_ind[np.argsort(b_scores[-to_select:])]
            all_selected.extend(b_selected.tolist())
            all_scores.extend(scores[b_selected])

        # We may still end up with not enough samples. Let's fill them with top margin
        if len(all_selected) < self.batch_size:
            to_select = self.batch_size - len(all_selected)
            print('Filling {}'.format(to_select))
            mask = np.ones(X.shape[0], dtype=bool)
            mask[all_selected] = False
            b_scores = scores[mask]
            b_selected = np.argsort(b_scores[-to_select:])
            all_selected.extend(np.where(mask)[0][b_selected])
            all_scores.extend(b_scores[b_selected])

        all_selected = np.asarray(all_selected)
        all_scores = np.asarray(all_scores)

        # Rounding errors?
        if len(all_selected) > self.batch_size:
            all_selected = all_selected[-self.batch_size:]
            all_scores = all_scores[-self.batch_size:]

        self.sample_scores_ = all_scores

        return all_selected



class ErrorSampler2(ScoredQuerySampler):

    def __init__(self, classifier, batch_size: int,
                 strategy: str = 'top', assume_fitted: bool = False,
                 random_state: RandomStateType = None,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.random_state = random_state
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.array, y: np.array) -> 'ErrorSampler1':
        """Fit the estimator on labeled samples.
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        self.random_state = check_random_state(self.random_state)
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        if len(y.shape) == 2:
            y = y.argmax(axis=1)
        mistakes = X[_get_probability_classes(self.classifier_, X).argmax(axis=1) != y]
        if len(mistakes) == 0:
            self._knn = None  # Do random
        else:
            self._knn = NearestNeighbors(n_neighbors=1).fit(mistakes)
        return self


    def score_samples(self, X: np.array) -> np.array:
        if self._knn == None:
            print('Warning: No mistakes to fit on, doing random')
            return self.random_state.random(X.shape[0])
        distances, _ = self._knn.kneighbors(X)
        return 1 / distances[:, 0]


class ScoredSamplerAggregator(ScoredQuerySampler):
    """KMeans sampler using a margin uncertainty sampler as preselector
    """

    def __init__(self, sampler_list, fun_agg, batch_size: int,
                 strategy: str = 'top',):
        super().__init__(batch_size, strategy=strategy)

        if len(sampler_list) != fun_agg.__code__.co_argcount:
            raise ValueError('Agg function does not match samplers')

        self.sampler_list = sampler_list
        self.fun_agg = fun_agg

    def fit(self, X: np.array, y: np.array = None) -> 'ScoredSamplerAggregator':
        """Fits the first query sampler
        Args:
            X: Labeled samples of shape [n_samples, n_features].
            y: Labels of shape [n_samples].
        
        Returns:
            The object itself
        """
        for sampler in self.sampler_list:
            sampler.fit(X, y)
        return self

    def score_samples(self, X: np.array) -> np.array:
        scores = [sampler.score_samples(X) for sampler in self.sampler_list]
        scores = self.fun_agg(*scores)

        return scores


class TwoStepGenericSampler(BaseQuerySampler):
    """KMeans sampler using a margin uncertainty sampler as preselector
    """

    def __init__(self, preselection_sampler, batch_size: int, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            preselection_sampler,
            KMeansSampler(batch_size, **kmeans_args)
        ]

    def fit(self, X: np.array, y: np.array = None) -> 'TwoStepGenericSampler':
        """Fits the first query sampler
        Args:
            X: Labeled samples of shape [n_samples, n_features].
            y: Labels of shape [n_samples].
        
        Returns:
            The object itself
        """
        self.sampler_list[0].fit(X, y)
        return self

    def select_samples(self, X: np.array,
                       sample_weight: np.array = None) -> np.array:
        """Selects the using uncertainty preselection and KMeans sampler.
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            sample_weight: Weight of the samples of shape (n_samples),
                optional.
        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        selected = self.sampler_list[0].select_samples(X)
        kwargs = dict()
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight[selected]
        new_selected = self.sampler_list[1].select_samples(
            X[selected], **kwargs)
        selected = selected[new_selected]
        
        return selected

def predict(test, clf):
    # Predict with all fitted estimators.
    x = np.array(list(map(lambda e: e.predict_proba(test), clf.estimators_)))
    
    # Roll axis because BaaL expect [n_samples, n_classes, ..., n_estimations]
    x = np.rollaxis(x, 0, 3)
    return x


class BatchBALDSampler(BaseQuerySampler):

    def __init__(self, classifier, batch_size: int,
                 assume_fitted: bool = False,
                 random_state=0,
                 verbose: int = 0):
        from baal.active.heuristics import BatchBALD
        super().__init__(batch_size)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        self.bald = BatchBALD(batch_size, num_draw=10)
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.array, y: np.array):
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        return self

    def select_samples(self, X: np.array) -> np.array:
        probs = predict(X, self.classifier_)
        return self.bald(probs)



# Adapted from sklearn
# May not work in all cases

def _forward_pass_partial(model, X, n_layers):

    from sklearn.neural_network._multilayer_perceptron import ACTIVATIONS
    from sklearn.utils.extmath import safe_sparse_dot
    from sklearn.utils.validation import check_is_fitted
    
    if n_layers >= model.n_layers_ - 1:
        raise ValueError('Do not use this to perform full prediction.')

    check_is_fitted(model)
    # X = model._validate_data(X, accept_sparse=['csr', 'csc'], reset=False)

    # Initialize first layer
    activation = X

    # Forward propagate
    hidden_activation = ACTIVATIONS[model.activation]
    for i in range(n_layers):
        activation = safe_sparse_dot(activation, model.coefs_[i])
        activation += model.intercepts_[i]
        hidden_activation(activation)

    return activation


class AutoEmbedder:

    ALLOWED = (MLPClassifier, Sequential, RandomForestClassifier)

    def __init__(self, model, **kwargs):

        if not isinstance(model, self.ALLOWED):
            raise TypeError('Expecting a model among ' + str(self.ALLOWED))

        if isinstance(model, RandomForestClassifier):
            if not 'X' in kwargs:
                raise ValueError('You must provide a calibration dataset for RF')
            n_leaves = [0] + [i.get_n_leaves() for i in model.estimators_]
            self.shifts = np.cumsum(n_leaves)
            data = model.apply(kwargs['X'])
            self.ohe = OneHotEncoder().fit(data)
            self.pca = TruncatedSVD(n_components=int(np.log(data.shape[1]) / (0.25 ** 2)))
            self.pca.fit(self.ohe.transform(data))

        self.model = model

    def __call__(self, X):
        if isinstance(self.model, MLPClassifier):
            return _forward_pass_partial(self.model, X, self.model.n_layers_ - 2)
        elif isinstance(self.model, Sequential):
            m = Model(inputs=self.model.input,
                      outputs=self.model.get_layer(index=len(self.model.layers) -1).output)
            m.compile()
            return m.predict(X)
        elif isinstance(self.model, RandomForestClassifier):
            return self.pca.transform(self.ohe.transform(self.model.apply(X)))
        else:
            raise ValueError('An unexpected error has occured')
            

class TrustSampler(ScoredQuerySampler):

    def __init__(self, classifier, batch_size: int, n_neighbors: int=10,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        self.trust_scorer_ = TrustScore(k_filter=n_neighbors)
        self.n_neighbors = n_neighbors
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        from alibi.confidence import TrustScore

        if len(y.shape) == 2:
            n_classes = y.shape[1]
            y = np.argmax(y, axis=1)
        else:
           n_classes = len(np.unique(y))
        self.trust_scorer_.fit(X, y, classes=n_classes)
        self._max_k = np.unique(y, return_counts=True)[1].min() - 1
        return super().fit(X, y=y)

    def score_samples(self, X: np.array) -> np.array:
        predicted = _get_probability_classes(self.classifier_, X).argmax(axis=1)
        score = - self.trust_scorer_.score(X, predicted, k=min(self._max_k, self.n_neighbors))[0]
        return score


class TrustTestSampler(ScoredQuerySampler):

    def __init__(self, classifier, batch_size: int, X_test, n_neighbors: int=10,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.X_test = X_test
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        self.trust_scorer_ = TrustScore(k_filter=n_neighbors)
        self.n_neighbors = n_neighbors
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        from alibi.confidence import TrustScore

        if len(y.shape) == 2:
            n_classes = y.shape[1]
            y = np.argmax(y, axis=1)
        else:
            n_classes = len(np.unique(y))
        y_pred = _get_probability_classes(self.classifier_, self.X_test).argmax(axis=1)
        self.trust_scorer_.fit(self.X_test, y_pred, classes=n_classes)
        self._max_k = np.unique(y_pred, return_counts=True)[1].min() - 1
        return super().fit(X, y=y)

    def score_samples(self, X: np.array) -> np.array:
        predicted = _get_probability_classes(self.classifier_, X).argmax(axis=1)
        score = - self.trust_scorer_.score(X, predicted, k=min(self._max_k, self.n_neighbors))[0]
        return score


class ITrustSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            TrustSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       fixed_cluster_centers=None) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected], fixed_cluster_centers=fixed_cluster_centers)
        selected = selected[new_selected]
        
        return selected

class TrustHybridSampler(ScoredQuerySampler):

    def __init__(self, classifier, batch_size: int, n_neighbors: int=10,
                 strategy: str = 'top', assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size, strategy=strategy)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        self.trust_scorer_ = TrustScore(k_filter=n_neighbors)
        self.n_neighbors = n_neighbors
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        from alibi.confidence import TrustScore

        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        if len(y.shape) == 2:
            n_classes = y.shape[1]
            y = np.argmax(y, axis=1)
        else:
           n_classes = len(np.unique(y))
        self.trust_scorer_.fit(X, y, classes=n_classes)
        self._max_k = np.unique(y, return_counts=True)[1].min() - 1

        return super().fit(X, y=y)

    def score_samples(self, X: np.array) -> np.array:
        predicted = _get_probability_classes(self.classifier_, X).argmax(axis=1)
        t_score = self.trust_scorer_.score(X, predicted, k=min(self._max_k, self.n_neighbors))[0]
        m_score = margin_score(self.classifier_, X)
        # Normalize both scores
        trust_trans = QuantileTransformer()
        margin_trans = QuantileTransformer()
        trust_score = 1 - trust_trans.fit_transform(t_score[:, None])[:, 0]
        m_score = margin_trans.fit_transform(m_score[:, None])[:, 0]
        return trust_score * m_score


class TrustMixedSampler(BaseQuerySampler):

    def __init__(self, classifier, batch_size: int, ratio:float, n_neighbors: int=10,
                 assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.ratio = ratio
        self.verbose = verbose
        self.trust_scorer_ = TrustScore(k_filter=n_neighbors)
        self.n_neighbors = n_neighbors
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        from alibi.confidence import TrustScore

        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        if len(y.shape) == 2:
            n_classes = y.shape[1]
            y = np.argmax(y, axis=1)
        else:
           n_classes = len(np.unique(y))
        self.trust_scorer_.fit(X, y, classes=n_classes)
        self._max_k = np.unique(y, return_counts=True)[1].min() - 1

        return super().fit(X, y=y)

    def select_samples(self, X: np.array) -> np.array:
        t_samples = int(self.ratio * self.batch_size)
        m_samples = self.batch_size - t_samples

        predicted = _get_probability_classes(self.classifier_, X).argmax(axis=1)
        
        t_score = - self.trust_scorer_.score(X, predicted, k=min(self._max_k, self.n_neighbors))[0]
        m_score = margin_score(self.classifier_, X)
        
        m_index = np.argsort(m_score)[-m_samples:]
        # In order to complete the batch with trust score ones, we first put a bad score for
        # the already selected ones in trust score
        t_score[m_index] = t_score.min()
        t_index = np.argsort(t_score)[-t_samples:]

        index = np.hstack([m_index, t_index])
        assert(index.shape[0] == self.batch_size)
        
        return index


class MarginMixedSampler(BaseQuerySampler):

    def __init__(self, classifier, batch_size: int, ratio:float, n_neighbors: int=10,
                 assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.ratio = ratio
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        return super().fit(X, y=y)

    def select_samples(self, X: np.array) -> np.array:
        r_samples = int(self.ratio * self.batch_size)
        m_samples = self.batch_size - r_samples

        m_score = margin_score(self.classifier_, X)
        m_index = np.argsort(m_score)[-m_samples:]

        splitter = ActiveLearningSplitter(X.shape[0])
        splitter.add_batch(m_index)

        # Select randomly among non selected
        r_index = np.random.choice(np.arange(X.shape[0] - m_samples), size=r_samples, replace=False)
        splitter.add_batch(r_index)
        index = np.where(splitter.selected)[0]

        assert(index.shape[0] == self.batch_size)
        return index


class TrustParetoSampler(BaseQuerySampler):

    def __init__(self, classifier, batch_size: int, n_neighbors: int=10,
                 assume_fitted: bool = False,
                 verbose: int = 0):
        super().__init__(batch_size)
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        self.trust_scorer_ = TrustScore(k_filter=n_neighbors)
        self.n_neighbors = n_neighbors
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True
        else:
            check_proba_estimator(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        from alibi.confidence import TrustScore

        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        if len(y.shape) == 2:
            n_classes = y.shape[1]
            y = np.argmax(y, axis=1)
        else:
           n_classes = len(np.unique(y))
        self.trust_scorer_.fit(X, y, classes=n_classes)
        self._max_k = np.unique(y, return_counts=True)[1].min() - 1

        return super().fit(X, y=y)

    def select_samples(self, X: np.array) -> np.array:

        predicted = _get_probability_classes(self.classifier_, X).argmax(axis=1)
        
        t_score = - self.trust_scorer_.score(X, predicted, k=min(self._max_k, self.n_neighbors))[0]
        m_score = margin_score(self.classifier_, X)
        i_range = range(m_score.shape[0])
        
        scores = sorted(list(zip(m_score, t_score, i_range)), reverse=True)
        pareto_indices = []
        while len(pareto_indices) < self.batch_size:
            pareto_front = [scores[0]]
            pareto_index = [scores[0][2]]
            new_scores = []
            for mti in scores[1:]:
                if mti[1] >= pareto_front[-1][1]:
                    pareto_front.append(mti)
                    pareto_index.append(mti[2])
                else:
                    new_scores.append(mti)
            scores = new_scores
            pareto_indices.extend(pareto_index)
            
        index = np.asarray(pareto_indices)
        index = index[:self.batch_size]
        return index
