from functools import partial
import os
import string
import sys
from sys import exit

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
from bench.data import get_openml, get_dataset

from Experiments.share_results import share_results

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

# Plots
from matplotlib import pyplot as plt
from cardinal.plotting import plot_confidence_interval

cwd = Path.cwd()

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


# def run_benchmark(new_sampler_generator, 
#                 datasets_ids:list(string)=['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist'], 
#                 sampler_name:string = 'my_custom_sampler'):

#     assert sampler_name is not None
#     assert dataset_id is not None
#     assert type(dataset_id) == str
#     assert dataset_id in ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist']

#     run(dataset_id, new_sampler_generator, sampler_name)


def run_benchmark(dataset_id, new_sampler_generator, sampler_name):
    db = CsvDb('user_results/results_{}/db'.format(dataset_id))

    # X, y, transformer, best_model = get_openml(dataset_id)
    preproc = get_dataset(dataset_id)
    if len(preproc) == 3:
        # DATA_TYPE = "image"
        X, y, best_model = preproc
    else:
        # DATA_TYPE = "tabular"
        X, y, transformer, best_model = preproc
        X = transformer.fit_transform(X)

    get_clf = lambda seed: best_model(seed=seed)
    fit_clf = lambda clf, X, y: clf.fit(X, y)

    n_classes = len(np.unique(y))

    k_start = False

    args = {
        "n_seed" : 10, 
        'n_iter' : 10, 
        'batch_size' : int(.001 * X.shape[0])
        }
    # n_iter = 10
    # n_seed = 2
    # batch_size = int(.01 * X.shape[0])


    start_size = args['batch_size']
    two_step_beta = 10
    # oracle_error = False


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


    # model_cache = dict()


    for seed in range(args['n_seed']):
        print('Iteration {}'.format(seed))
        methods = {
            # 'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=params['seed']),
            # 'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'confidence': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'entropy': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'kmeans': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=params['seed']), batch_size=params['batch_size']),
            # 'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
            # 'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
            # 'batchbald': lambda params: BatchBALDSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=X[splitter.train]), batch_size=params['batch_size']),

            # 'random-2': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=params['seed']),
            # 'margin-2': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'confidence-2': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'entropy-2': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'kmeans-2': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=params['seed']), batch_size=params['batch_size']),
            # 'wkmeans-2': lambda params: TwoStepMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        }

        # Add new sampler method in the evaluated methods
        methods[sampler_name] = new_sampler_generator


        for name_index, name in enumerate(methods):
            print(name)

            # Check if it has been computer already
            config = dict(
                seed=seed,
                method=name,
                n_iter=args['n_iter'] - 1,
                dataset=dataset_id
            )

            
            # Capture the output for logging
            with Tee() as tee:

                # splitter = ActiveLearningSplitter.train_test_split(X.shape[0], test_size=.2, random_state=int(seed), stratify=y)
                test_indexes = load_indexes(dataset_id, seed, type='test')
                mask = np.full(X.shape[0], -1, dtype=np.int8)
                mask[test_indexes] = -2
                splitter = ActiveLearningSplitter.from_mask(mask)   # [INFO] We instanciate the ActiveLearningSplitter with test sample indexes that have been registered and used in the previous benchmark (instead of using seeds)

                method = methods[name]

                # First, get at least one sample for each class
                # one_per_class = np.unique(y[splitter.non_selected], return_index=True)[1]
                one_per_class = load_indexes(dataset_id, seed, type='one_class')    # [INFO] We select the same first samples indexes (from each class) that have been registered and used in the previous benchmark (instead of using seeds)
                splitter.add_batch(one_per_class)
    
                if not k_start:
                    # first_index, _ = train_test_split(
                    #     np.arange(X[splitter.non_selected].shape[0]),
                    #     train_size=start_size - one_per_class.shape[0],
                    #     random_state=int(seed),
                    #     stratify=y[splitter.non_selected])
                    first_index = load_indexes(dataset_id, seed, type='random')     # [INFO] We select the same first samples indexes (randomly chosen for initialisation) that have been registered and used in the previous benchmark (instead of using seeds)

                else:
                    start_sampler = MiniBatchKMeansSampler(start_size - one_per_class.shape[0], random_state=int(seed))
                    start_sampler.fit(X[splitter.non_selected])
                    first_index = start_sampler.select_samples(X[splitter.non_selected])
                splitter.add_batch(first_index, partial=True)

                classifier = get_clf(seed)
                previous_predicted = None

                assert(splitter.selected.sum() == start_size)
                assert(splitter.current_iter == 0)

                for i in range(args['n_iter']):

                    fit_clf(classifier, X[splitter.selected], y[splitter.selected])
                    predicted = _get_probability_classes(classifier, X)
            
                    DYNAMIC_PARAMS = dict(batch_size=args['batch_size'], clf=classifier, seed=int(seed), iteration=i)  # HERE : PARAMETERS GIVEN TO THE SAMPLER AT EACH ITERATION             #iter=i + 1, splitter=splitter
                    # X_test = X[splitter.test]
                    # params['X_test'] = X_test

                    sampler = method(DYNAMIC_PARAMS)
                    sampler.fit(X[splitter.selected], y[splitter.selected])

                    new_selected_index = sampler.select_samples(X[splitter.non_selected])
                                    
                    splitter.add_batch(new_selected_index)

                    assert(splitter.current_iter == (i + 1))
                    assert(splitter.selected_at(i + 1).sum() == ((i + 1) * args['batch_size']))
                    assert(splitter.batch_at(i + 1).sum() == args['batch_size'])

                    config = dict(
                        seed=seed,
                        method=name,
                        n_iter=i,
                        dataset=dataset_id
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
                    # Metric skipped for dataset 43551 because of invalid shape error when calling trustscorer.score() (shape in axis 1: 0.)

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


                    # ================================================================================

            log_folder = Path('logs')
            log_folder.mkdir(exist_ok=True)
            with open(log_folder / '{}-{}-{}.log'.format(name, seed, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
                f.write(tee.read())
        

        #Saving indexes for reproducibility
        """
        Data types :
        "one per class" = 0
        "random" = 1
        "test" = 2
        """

                    # try:
                    #     df_to_save = pd.read_csv('user_results/results_{}/indexes.csv'.format(dataset_id)) 
                    #     #First unique indexes
                    #     df = pd.DataFrame([{'seed': int(seed), 'type': 0, 'index':index} for index in one_per_class])
                    #     df_to_save = pd.concat([df_to_save, df], ignore_index=True)
                    # except:
                    #     #First unique indexes
                    #     df_to_save = pd.DataFrame([{'seed': seed, 'type': 0, 'index':index} for index in one_per_class])
                
                    # # Randomly selected samples
                    # df = pd.DataFrame([{'seed': seed, 'type': 1, 'index':index} for index in first_index])
                    # df_to_save = pd.concat([df_to_save, df], ignore_index=True)
                    # # Test indexes
                    # df = pd.DataFrame([{'seed': seed, 'type': 2, 'index':index} for index, is_in_test_set in enumerate(splitter.test) if is_in_test_set])
                    # df_to_save = pd.concat([df_to_save, df], ignore_index=True)
                    
                    # df_to_save.to_csv('user_results/results_{}/indexes.csv'.format(dataset_id), index=False)


        # #TODO : use this only when runing initial benchmark (and not on client side) because it's very long
        # # We define a column 'index_id__not_used' in order to define different csv indexes for save indexes with the save type (if we don't do so, Csv db saves only one index from each type)
        # # One per class
        # for index_id, index in enumerate(one_per_class):
        #     dic = dict(seed=int(seed), type=0, index_id__not_used=index_id)
        #     db.upsert('indexes', dic,  int(index))
        # # Randomly selected samples
        # for index_id, index in enumerate(first_index):
        #     dic = dict(seed=int(seed), type=1, index_id__not_used=index_id)
        #     db.upsert('indexes', dic,  int(index))
        # # Test indexes
        # for index_id, (index, is_in_test_set) in enumerate(enumerate(splitter.test)):
        #     if is_in_test_set:
        #         dic = dict(seed=int(seed), type=2, index_id__not_used=index_id)
        #         db.upsert('indexes', dic,  int(index))


    # Plots results from saved csv
    plot_results(dataset_id, n_iter=args['n_iter'], n_seed=args['n_seed'])

    # Propose to merge current sampler results to benchmark resutls
    share_results(dataset_id)


def plot_results(dataset_id, n_iter, n_seed, show=False):

    x_data = np.arange(n_iter)

    metrics = [
        ('Accuracy','accuracy_test.csv'),
        ('Agreement','agreement_test.csv'),
        ('Trustscore','test_trustscore.csv'),
        ('Violation','test_violation.csv'),
        ('Exploration','soft_exploration.csv'), #TODO is it the important file to plot ?
        ('Closest','this_closest.csv')  #TODO is it the important file to plot ?
    ]

    # dataset_ids = [1461]    #[1461, 1471, 1502, 1590, 40922, 41138, 42395, 43439, 43551, 42803, 41162, 'cifar10', 'cifar10_simclr', 'mnist]
    # for dataset_id in dataset_ids:
    for i, (metric_name, filename) in enumerate(metrics):
            print('\n',metric_name, '\n')
        # try:

            # Plot new sampler results

            df = pd.read_csv(f'user_results/results_{dataset_id}/db/{filename}')
            # sampler_name = np.unique(df["method"].values)[0]    #TODO

            #Loop in case their are several samplers tested here
            method_names = np.unique(df["method"].values)
            for sampler_name in method_names:
                all_metric = []
                for seed in range(n_seed):
                    metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
                    all_metric.append(metric)
                    
                plt.figure(i, figsize=(15,10))
                plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name))
            

            # Plot other samplers results from the benchmark

            # plot_benchmark_sampler_results(i, dataset_id, filename, x_data, n_seed)
            df = pd.read_csv(f'Experiments/results_{dataset_id}/db/{filename}')
            method_names = np.unique(df["method"].values)

            for sampler_name in method_names:
                all_metric = []
                for seed in range(n_seed):
                    metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
                    all_metric.append(metric)
                
                plt.figure(i, figsize=(15,10))
                plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name))


            plt.xlabel('AL iteration')
            plt.ylabel(metric_name)
            plt.title('{} metric'.format(metric_name))
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'user_results/results_{dataset_id}/plot-'+metric_name+'.png')
            # plt.clf()

            # We only show the accuracy to the user, other metrics are still saved
            # if metric_name != 'Accuracy':
            #     plt.clf()
            
        # except:
        #     print("[ERROR] Problem occured when trying to plot {} metric values".format(metric_name))

    if show:
        plt.show()


def load_indexes(dataset_id, seed, type):
    """
    Instead of only seeding the random initialisation, we use the indexes that have been saved during the previous benchmark run.
    Hence, all the samplers will have the same random initialisation samples
    """
    if type == 'one_class':
        type = 0
    elif type == 'random':
        type = 1
    elif type == 'test':
        type = 2 
    else:
        exit(f'[ERROR] Can’t load indexes from type {type}')

    df = pd.read_csv(f'Experiments/results_{dataset_id}/db/indexes.csv')

    # return df.loc[(df["seed"] == seed) & (df["type"]== type)]['value'].values   # TODO When column name will be updated
    return df.loc[(df["seed"] == seed) & (df["type"]== type)]['index'].values   


def run_benchmark2(dataset_id):
    """
    Original benchmark run
    Used to find reproducibility problems only
    TODO : to delete after
    """
    db = CsvDb('user_results/results_{}'.format(dataset_id))

    # X, y, transformer, best_model = get_openml(dataset_id)
    preproc = get_dataset(dataset_id)
    if len(preproc) == 3:
        DATA_TYPE = "image"
        X, y, best_model = preproc
    else:
        DATA_TYPE = "tabular"
        X, y, transformer, best_model = preproc
        X = transformer.fit_transform(X)

    get_clf = lambda seed: best_model(seed=seed)    # Pas seedé avant
    fit_clf = lambda clf, X, y: clf.fit(X, y)

    n_classes = len(np.unique(y))

    k_start = False

    n_iter = 10
    n_seed = 10
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


    for seed in range(n_seed):
        print('Iteration {}'.format(seed))
        methods = {
            # 'random-2': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=int(seed)),
            'margin-2': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'confidence-2': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'entropy-2': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'kmeans-2': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=int(seed)), batch_size=params['batch_size']),
            # 'wkmeans-2': lambda params: TwoStepMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
            
            # 'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
            # 'batchbald': lambda params: BatchBALDSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
            # 'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=X[splitter.train]), batch_size=params['batch_size']),
        }

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


        for name_index, name in enumerate(methods):
            print(name)

            # Check if it has been computer already
            config = dict(
                seed=seed,
                method=name,
                n_iter=n_iter - 1,
                dataset=dataset_id
            )

            # v = db.get('accuracy_selected', config)
            # if v is not None and v.shape[0] > 0:
            #     continue
            
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

                classifier = get_clf(seed = seed)
                previous_predicted = None

                assert(splitter.selected.sum() == start_size)
                assert(splitter.current_iter == 0)

                for i in range(n_iter):

                    fit_clf(classifier, X[splitter.selected], y[splitter.selected])
                    predicted = _get_probability_classes(classifier, X)
            
                    params = dict(batch_size=batch_size, clf=classifier)    #iter=i + 1, splitter=splitter
                    X_test = X[splitter.test]
                    params['X_test'] = X_test

                    sampler = method(params)
                    sampler.fit(X[splitter.selected], y[splitter.selected])

                    new_selected_index = sampler.select_samples(X[splitter.non_selected])
                                    
                    splitter.add_batch(new_selected_index)

                    assert(splitter.current_iter == (i + 1))
                    assert(splitter.selected_at(i + 1).sum() == ((i + 1) * batch_size))
                    assert(splitter.batch_at(i + 1).sum() == batch_size)

                    config = dict(
                        seed=seed,
                        method=name,
                        n_iter=i,
                        dataset=dataset_id
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
                    # Metric skipped for dataset 43551 because of invalid shape error when calling trustscorer.score() (shape in axis 1: 0.)

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


                    # ================================================================================

            # #Saving indexes for reproducibility
            # """
            # Data types :
            # "one per class" = 0
            # "random" = 1
            # "test" = 2
            # """

            # try:
            #     df_to_save = pd.read_csv('results_{}/indexes.csv'.format(dataset_id)) 
            #     #First unique indexes
            #     df = pd.DataFrame([{'seed': int(seed), 'method': int(name_index), 'type': 0, 'index':index} for index in one_per_class])
            #     df_to_save = pd.concat([df_to_save, df], ignore_index=True)
            # except:
            #     #First unique indexes
            #     df_to_save = pd.DataFrame([{'seed': seed, 'method': name_index, 'type': 0, 'index':index} for index in one_per_class])
        
            # # Randomly selected samples
            # df = pd.DataFrame([{'seed': seed, 'method': name_index, 'type': 1, 'index':index} for index in first_index])
            # df_to_save = pd.concat([df_to_save, df], ignore_index=True)
            # # Test indexes
            # df = pd.DataFrame([{'seed': seed, 'method': name_index, 'type': 2, 'index':index} for index, is_in_test_set in enumerate(splitter.test) if is_in_test_set])
            # df_to_save = pd.concat([df_to_save, df], ignore_index=True)
            # # Train indexes
            # # for i in range(n_iter):
            # #     df = pd.DataFrame([{'seed': seed, 'method': name, 'n_iter': i+1, 'dataset':dataset_id, 'type': "train", 'index':index} for index, is_in_train_set in enumerate(splitter.batch_at(i + 1)) if is_in_train_set])
            # #     df_to_save = pd.concat([df_to_save, df], ignore_index=True)
            
            # df_to_save.to_csv('results_{}/indexes.csv'.format(dataset_id), index=False)


            log_folder = Path('logs')
            log_folder.mkdir(exist_ok=True)
            with open(log_folder / '{}-{}-{}.log'.format(name, seed, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
                f.write(tee.read())
        
        #Saving indexes for reproducibility
        """
        Data types :
        "one per class" = 0
        "random" = 1
        "test" = 2
        """

        try:
            df_to_save = pd.read_csv('user_results/results_{}/indexes.csv'.format(dataset_id)) 
            #First unique indexes
            df = pd.DataFrame([{'seed': int(seed), 'type': 0, 'index':index} for index in one_per_class])
            df_to_save = pd.concat([df_to_save, df], ignore_index=True)
        except:
            #First unique indexes
            df_to_save = pd.DataFrame([{'seed': seed, 'type': 0, 'index':index} for index in one_per_class])

        # Randomly selected samples
        df = pd.DataFrame([{'seed': seed, 'type': 1, 'index':index} for index in first_index])
        df_to_save = pd.concat([df_to_save, df], ignore_index=True)
        # Test indexes
        df = pd.DataFrame([{'seed': seed, 'type': 2, 'index':index} for index, is_in_test_set in enumerate(splitter.test) if is_in_test_set])
        df_to_save = pd.concat([df_to_save, df], ignore_index=True)
        
        df_to_save.to_csv('user_results/results_{}/indexes.csv'.format(dataset_id), index=False)



    plot_results(dataset_id, n_iter, n_seed)






if __name__ == '__main__':

    indexes = load_indexes(1461, 0, type="one_class")
    print(indexes)
