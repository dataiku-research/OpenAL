from functools import partial
import os
import sys


sys.path.append('..')
sys.path.append('.')

import itertools
from pathlib import Path
import importlib
import argparse
import io
from enum import Enum

import numpy as np
import pandas as pd
from datetime import datetime
from bench.csv_db import CsvDb
from bench.data import get_openml

from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

from cardinal.uncertainty import MarginSampler, ConfidenceSampler, EntropySampler, margin_score, confidence_score, entropy_score, _get_probability_classes
from cardinal.random import RandomSampler
from cardinal.clustering import KCentroidSampler, MiniBatchKMeansSampler, KCenterGreedy
from cardinal.utils import ActiveLearningSplitter


from bench.samplers import TwoStepIncrementalMiniBatchKMeansSampler, TwoStepMiniBatchKMeansSampler
from bench.trustscore import TrustScore
import bench.prose.datainsights as di



class Tee(io.StringIO):
    class Source(Enum):
        STDOUT = 1
        STDERR = 2

    def __init__(self, clone=Source.STDOUT, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._clone = clone

        if clone == Tee.Source.STDOUT:
            self._out = sys.stdout
        elif clone == Tee.Source.STDERR:
            self._out = sys.stderr
        else:
            raise ValueError("Clone has to be STDOUT or STDERR.")

    def write(self, *args, **kwargs):
        self._out.write(*args, **kwargs)
        return super().write(*args, **kwargs)

    def __enter__(self):
        if self._clone == Tee.Source.STDOUT:
            sys.stdout = self
        else:
            sys.stderr = self
        self.seek(io.SEEK_END)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clone == Tee.Source.STDOUT:
            sys.stdout = self._out
        else:
            sys.stderr = self._out
        self.seek(0)
        return False



parser = argparse.ArgumentParser()
parser.add_argument('-m', action="store", nargs='*', default=['all'], help='Metrics to print')
parser.add_argument('methods', nargs='*', help='Methods to print', default=[])

args = parser.parse_args()

metrics = set(args.m)
samplers_to_compute = args.methods

del args

def should_compute(metric):
    if 'all' in metrics:
        return True
    return metric in metrics


if len(samplers_to_compute) > 0:
    print('Will only compute samplers', samplers_to_compute)

cwd = Path.cwd()
cache_path = cwd / 'cache'
exp_path = cwd / '..' / 'exp'
database_path = './results/'
db = CsvDb('results')

dataset_name = cwd.stem


print('Cache path is {}'.format(cache_path))

# Load experiment configuration
# exp_module = importlib.import_module(dataset_name)
# exp_config = exp_module.get_config()

# start_size = exp_config['start_size']
# batches = exp_config['batches']
# two_step_beta = exp_config['two_step_beta']
# oracle_error = exp_config.get('oracle_error', None)
# metric = exp_config.get('metric', accuracy_score)

# data = exp_module.get_dataset()
X, y, transformer = get_openml(1461)
X = transformer.fit_transform(X)

get_clf = lambda: RandomForestClassifier(max_depth=8)
fit_clf = lambda clf, X, y: clf.fit(X, y)

n_classes = len(np.unique(y))

k_start = False

n_iter = 10
batch_size = int(.001 * X.shape[0])


start_size = batch_size
two_step_beta = 10
oracle_error = False


def get_min_dist_per_class(dist, labels):
    assert(dist.shape[0] == labels.shape[0])
    min_dist_per_class = np.zeros((dist.shape[1], n_classes))
    max_dist = np.max(dist)  # If a class is not represented we use this as max distance

    for ic in range(n_classes):
        mask_sample_of_class_ic = dist[labels == ic]
        if mask_sample_of_class_ic.shape[0] != 0:
            min_dist_per_class[:, ic] = np.min(dist[labels == ic], axis=0)
        else :
            min_dist_per_class[:, ic].fill(max_dist)
    
    return min_dist_per_class

model_cache = dict()


for seed in range(10):
    print(seed)
    methods = dict()
    _methods = {
        'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=int(seed)),
        'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'confidence': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'entropy': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'kmeans': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=int(seed)), batch_size=params['batch_size']),
        'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
        'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
        # 'batchbald': lambda params: BatchBALDSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        # 'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=X[splitter.train]), batch_size=params['batch_size']),
    }

    methods.update(_methods)

    if samplers_to_compute is None:
        samplers_to_compute = list(methods.keys())
    
    index = np.arange(X.shape[0])
    
    # precomputed_proba_path = Path('precomputed_proba') / (seed + ds)

    # if not precomputed_proba_path.exists():
    #     precomputed_proba_path.mkdir(parents=True)
    #     clf = get_clf()
    #     fit_clf(clf, X[splitter.test], y[splitter.test], **exp_config.get('full_dataset_fit_params', {}))
    #     y_proba = _get_probability_classes(clf, X)

    #     max_confidence = confidence_score('precomputed', y_proba)
    #     np.save(str(precomputed_proba_path / 'max_confidence.npy'), max_confidence)

    #     max_margin = margin_score('precomputed', y_proba)
    #     np.save(str(precomputed_proba_path / 'max_margin.npy'), max_margin)

    #     max_entropy = entropy_score('precomputed', y_proba)
    #     np.save(str(precomputed_proba_path / 'max_entropy.npy'), max_entropy)

    # max_confidence = np.load(str(precomputed_proba_path / 'max_confidence.npy'))
    # max_margin = np.load(str(precomputed_proba_path / 'max_margin.npy'))
    # max_entropy = np.load(str(precomputed_proba_path / 'max_entropy.npy'))

    for name in samplers_to_compute:
        print(name)
        
        # Capture the output for logging
        with Tee() as tee:

            splitter = ActiveLearningSplitter.train_test_split(X.shape[0], test_size=.2, random_state=int(seed), stratify=y)
            
            method = methods[name]

            # First, get at least one sample for each class
            one_per_class = np.unique(y[splitter.non_selected], return_index=True)[1]
            splitter.add_batch(one_per_class)
            
            if not k_start:
                first_index, _ = train_test_split(
                    np.arange(X[splitter.non_selected].shape[0]),
                    train_size=start_size - one_per_class.shape[0],
                    random_state=int(seed),
                    stratify=y[splitter.non_selected])
            else:
                start_sampler = MiniBatchKMeansSampler(start_size - one_per_class.shape[0], random_state=int(seed))
                start_sampler.fit(X[splitter.non_selected])
                first_index = start_sampler.select_samples(X[splitter.non_selected])
            splitter.add_batch(first_index, partial=True)

            classifier = get_clf()
            previous_predicted = None

            assert(splitter.selected.sum() == start_size)
            assert(splitter.current_iter == 0)

            for i in range(n_iter):

                fit_clf(classifier, X[splitter.selected], y[splitter.selected])
                predicted = _get_probability_classes(classifier, X)
        
                params = dict(batch_size=batch_size, clf=classifier, iter=i + 1, splitter=splitter)
                X_test = X[splitter.test]
                params['X_test'] = X_test

                sampler = method(params)
                sampler.fit(X[splitter.selected], y[splitter.selected])

                if name.startswith('iwkmeans'):
                    new_selected_index = sampler.select_samples(X[splitter.non_selected], fixed_cluster_centers=X[splitter.selected])
                elif name.startswith('iconfidence'):
                    new_selected_index = sampler.select_samples(X[splitter.non_selected], max_confidence[splitter.non_selected])
                elif name.startswith('imargin'):
                    new_selected_index = sampler.select_samples(X[splitter.non_selected], max_margin[splitter.non_selected])
                elif name.startswith('ientropy'):
                    new_selected_index = sampler.select_samples(X[splitter.non_selected], max_entropy[splitter.non_selected])
                elif name.startswith('iiwkmeans'):
                    new_selected_index = sampler.select_samples(X[splitter.non_selected], max_margin[splitter.non_selected], fixed_cluster_centers=X[splitter.selected])
                else:
                    new_selected_index = sampler.select_samples(X[splitter.non_selected])
                                
                splitter.add_batch(new_selected_index)

                assert(splitter.current_iter == (i + 1))
                assert(splitter.selected_at(i + 1).sum() == ((i + 1) * batch_size))
                assert(splitter.batch_at(i + 1).sum() == batch_size)

                config = dict(
                    seed=seed,
                    method=name,
                    n_iter=i,
                    dataset=dataset_name
                )

                selected = splitter.selected_at(i + 1)
                batch = splitter.batch_at(i + 1)

                # ================================================================================

                # Accuracy

                predicted_proba_test = predicted[splitter.test] 
                predicted_proba_selected = predicted[selected] 
                predicted_proba_batch = predicted[batch]

                predicted_test = np.argmax(predicted_proba_test, axis=1)
                predicted_selected = np.argmax(predicted_proba_selected, axis=1)
                predicted_batch = np.argmax(predicted_proba_batch, axis=1)
                
                db.upsert('accuracy_test', config, accuracy_score(y[splitter.test], predicted_test))
                db.upsert('accuracy_selected', config, accuracy_score(y[selected], predicted_selected))
                db.upsert('accuracy_batch', config, accuracy_score(y[batch], predicted_batch))

                # ================================================================================

                # Contradictions

                if previous_predicted is not None:
                    pre_predicted_test = previous_predicted[splitter.test]
                    pre_predicted_selected = previous_predicted[splitter.selected_at(i)]
                    pre_predicted_batch = previous_predicted[splitter.batch_at(i)]

                    db.upsert('contradiction_test', config, np.mean(np.argmax(pre_predicted_test, axis=1) != np.argmax(predicted_proba_test, axis=1)))
                    db.upsert('contradiction_selected', config, np.mean(np.argmax(pre_predicted_selected, axis=1) != np.argmax(predicted_proba_selected, axis=1)))
                    db.upsert('contradiction_batch', config, np.mean(np.argmax(pre_predicted_batch, axis=1) != np.argmax(predicted_proba_batch, axis=1)))

                # ================================================================================

                # Exploration

                post_selected = splitter.selected_at(i + 1)

                distance_matrix = pairwise_distances(X[selected], X[splitter.test])
                min_dist_per_class = get_min_dist_per_class(distance_matrix, predicted_selected)
                post_distance_matrix = pairwise_distances(X[post_selected], X[splitter.test])
                post_min_dist_per_class = get_min_dist_per_class(post_distance_matrix, predicted_selected)
                
                db.upsert('hard_exploration', config, np.mean(np.argmin(min_dist_per_class, axis=1) == np.argmin(post_min_dist_per_class, axis=1)).item())
                db.upsert('soft_exploration', config, np.mean(np.abs(min_dist_per_class - post_min_dist_per_class)).item())
                db.upsert('top_exploration', config, np.mean(np.min(min_dist_per_class, axis=1) - np.min(post_min_dist_per_class, axis=1)).item())

                # Batch Distance
                post_closest = post_distance_matrix.min(axis=1)
                closest = distance_matrix.min(axis=1)
                
                db.upsert('post_closest', config, np.mean(post_closest))
                db.upsert('this_closest', config, np.mean(closest))
                db.upsert('diff_closest', config, np.mean(closest) - np.mean(post_closest))


                # ================================================================================

                # Trustscore

                trustscorer = TrustScore()
                score = np.nan
                trustscorer.fit(X[selected], y[selected], classes=n_classes)
                max_k = np.unique(y[selected], return_counts=True)[1].min() - 1
                score = trustscorer.score(X[batch], predicted[batch], k=min(max_k, 10))[0].mean()
                db.upsert('batch_trustscore', config, score)

                np.random.seed(int(seed))
                idx_sel = np.random.choice(splitter.test.sum())

                score = trustscorer.score(X[splitter.test], predicted[splitter.test], k=min(max_k, 10))[0].mean()
                db.upsert('test_trustscore', config, score)
                score = trustscorer.score(X[selected], predicted[selected], k=min(max_k, 10))[0].mean()
                db.upsert('self_trustscore', config, score)

                # ================================================================================

                # Violations

                assertions = di.learn_assertions(pd.DataFrame(X[selected]), max_self_violation=1)
                if assertions.size() > 0:
                    score = assertions.evaluate(pd.DataFrame(X[splitter.test]), explanation=True).row_wise_violation_summary['violation'].sum()
                    db.upsert('test_violation', config, score)
                    score = assertions.evaluate(pd.DataFrame(X[selected]), explanation=True).row_wise_violation_summary['violation'].sum()
                    db.upsert('self_violation', config, score)
                    score = assertions.evaluate(pd.DataFrame(X[batch]), explanation=True).row_wise_violation_summary['violation'].sum()
                    db.upsert('batch_violation', config, score)

                else:
                    print('Assertions learning failed')
                    score = 0
                    db.upsert('test_violation', config,  score)
                    db.upsert('self_violation', config, score)
                    db.upsert('batch_violation', config, score)



                # ================================================================================

                # Agreement

                knn = KNeighborsClassifier()
                knn.fit(X[selected], y[selected])
                db.upsert('agreement_test', config,  np.mean(knn.predict(X[splitter.test]) == predicted_test))
                db.upsert('agreement_selected', config,  np.mean(knn.predict(X[selected]) == predicted_selected))
                db.upsert('agreement_batch', config,  np.mean(knn.predict(X[batch]) == predicted_batch))


        log_folder = Path('logs')
        log_folder.mkdir(exist_ok=True)
        with open(log_folder / '{}-{}-{}.log'.format(name, seed, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
            f.write(tee.read())
