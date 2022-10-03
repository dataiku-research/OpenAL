from functools import partial
import os
import string
import json
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

from experiments.share_results import share_results

from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, pairwise_distances, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from cardinal.uncertainty import MarginSampler, ConfidenceSampler, EntropySampler, margin_score, confidence_score, entropy_score, _get_probability_classes
from cardinal.random import RandomSampler
from cardinal.clustering import KCentroidSampler, MiniBatchKMeansSampler, KCenterGreedy
from cardinal.utils import ActiveLearningSplitter


from bench.samplers import TwoStepIncrementalMiniBatchKMeansSampler, TwoStepMiniBatchKMeansSampler, AutoEmbedder, BatchBALDSampler 
from bench.trustscore import TrustScore
import bench.prose.datainsights as di

# Plots
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from bench.plotting import plot_confidence_interval # Upgraded version of cardinal plot_confidence_interval method
# from cardinal.plotting import plot_confidence_interval

import joblib
from tqdm import tqdm
# import line_profiler
# profile = line_profiler.LineProfiler()
# import time


# Setup matplotlib
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

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


def create_initial_conditions(initial_conditions_name, dataset_id, initial_labeled_size, n_folds):

    initial_conditions_folder = Path('./experiment_indices') / initial_conditions_name
    initial_conditions_folder.mkdir(parents=True, exist_ok=False)

    preproc = get_dataset(dataset_id)
    if len(preproc) == 3:
        X, y, _ = preproc
    else:
        X, y, _, _ = preproc

    initial_labeled_size = int(initial_labeled_size * X.shape[0]) if initial_labeled_size < 1 else initial_labeled_size

    experimental_parameters = {
        "dataset": dataset_id,
        "n_folds": n_folds,
        "initial_labeled_size": initial_labeled_size,
        "initial_conditions_name": initial_conditions_name,
    }

    with open(str(initial_conditions_folder / 'experimental_parameters.json'), 'w') as experimental_parameters_file:
        json.dump(experimental_parameters, experimental_parameters_file)

    for seed in range(n_folds):
        splitter = ActiveLearningSplitter.train_test_split(X.shape[0], test_size=.2, random_state=int(seed), stratify=y)
        splitter.initialize_with_random(
            n_init_samples=initial_labeled_size,
            at_least_one_of_each_class=y[splitter.train],
            random_state=int(seed))

        # Splitter has no save method for now, we do it manually
        np.savetxt(initial_conditions_folder / '{}.txt'.format(seed), splitter._mask)


def create_experiment(experiment_name, initial_conditions_name, batch_size, n_iter):
    experiment_folder = Path('./experiment_results') / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=False)

    experimental_parameters = {
        "batch_size": batch_size,
        "n_iter": n_iter,
        "initial_conditions": initial_conditions_name,
        "experiment_name": experiment_name,
    }

    with open(str(experiment_folder / 'experimental_parameters.json'), 'w') as experimental_parameters_file:
        json.dump(experimental_parameters, experimental_parameters_file)


def load_initial_conditions(initial_conditions_name):
    initial_conditions_folder = Path('./experiment_indices') / initial_conditions_name

    with open(str(initial_conditions_folder / 'experimental_parameters.json'), 'w') as experimental_parameters_file:
        experimental_parameters = json.load(experimental_parameters_file)
    
    return experimental_parameters


def load_experiment(experiment_name, initial_conditions):
    experiment_folder = Path('./experiment_results') / experiment_name

    with open(str(experiment_folder / 'experimental_parameters.json'), 'w') as experimental_parameters_file:
        experimental_parameters = json.load(experimental_parameters_file)

    if experimental_parameters["initial_conditions"] != initial_conditions['initial_conditions_name']:
        raise ValueError('Initial conditions and experiments do not match')

    experimental_parameters.update(initial_conditions)
    return experimental_parameters
    

def create_experiments():
    datasets_ids = [
        '1461', '1471', '1502', '1590', '40922', '41138', '42395',
        '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr',
        'mnist'
    ]

    for dataset_id in datasets_ids:
        batch_size_ratio = .005 if dataset_id == '1471' else .001
        create_initial_conditions(dataset_id, dataset_id, batch_size_ratio, 10)
        create_experiment(dataset_id, dataset_id, batch_size_ratio, 10)


def run_benchmark(
    sampler_name:string,
    new_sampler_generator, 
    datasets_ids:list=['1461', '1471', '1502', '1590', '40922', '41138',
                       '42395', '43439', '43551', '42803', '41162', 'cifar10',
                       'cifar10_simclr', 'mnist']):

    assert sampler_name is not None
    assert datasets_ids is not None
    assert type(datasets_ids) == list, f'{datasets_ids} is of type {type(datasets_ids)} instead of type "list"'
    if len(datasets_ids) == 0 : datasets_ids = ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist']
    for dataset_id in datasets_ids:
        assert (dataset_id is not None)
        assert type(dataset_id) == str, f'{dataset_id} is of type {type(dataset_id)} instead of type "str"'
        assert dataset_id in ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr','cifar100', 'cifar100_simclr', 'mnist'], f"{dataset_id} not in available datasets :\n- 1461 \n- 1471\n- 1502\n- 1590\n- 40922\n- 41138\n- 42395\n- 43439\n- 43551\n- 42803\n- 41162\n- cifar10\n- cifar10_simclr\n- cifar100\n- cifar100_simclr\n- mnist"


    """
    0.1% labelisation for all tasks expect for #1471 where we labelise 0.5% at each AL iteration
    Explanation :
        # 1471 is very small with a lot of features
        -> 0.01% of #1471 represents 8 samples : far too small for an AL experiment      
        -> need more samples to learn
    """

    two_step_beta = 10  # Beta parameter for wkmeans sampler

    # Samplers should reming commented as they have already been ran and saved in the benchmark results
    methods = {
        'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=params['seed']),
        'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'confidence': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'entropy': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'kmeans': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=params['seed']), batch_size=params['batch_size']),
        'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=params['train_dataset']), batch_size=params['batch_size']),
    }

    for dataset_id in datasets_ids:
        initial_conditions = load_initial_conditions(dataset_id)
        experimental_parameters = load_experiment(dataset_id)
        run(experimental_parameters)


# @profile
def run(experimental_parameters, methods):

    experiment_name = experimental_parameters['experiment_name']
    initial_conditions_name = experimental_parameters['initial_conditions']
    batch_size = experimental_parameters['batch_size']
    n_iter = experimental_parameters['n_iter']
    dataset_id = experimental_parameters['dataset_id']
    n_folds = experimental_parameters['n_folds']
    n_initial_labeled = experimental_parameters['n_initial_labeled']

    experiment_folder = Path('./experiment_results') / experiment_name

    print(f'\n--- Running experiment {experiment_name} ---\n')

    db = CsvDb(experiment_folder / 'db')

    preproc = get_dataset(dataset_id)
    if len(preproc) == 3:
        X, y, best_model = preproc
    else:
        X, y, transformer, best_model = preproc
        X = transformer.fit_transform(X)

    get_clf = lambda seed: best_model(seed=seed)
    fit_clf = lambda clf, X, y: clf.fit(X, y)

    n_classes = len(np.unique(y))
    batch_size = int(batch_size * X.shape[0]) if batch_size < 1 else batch_size



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

    def run_AL_experiment(seed):
        """Processes one AL experiment (corresponding to one fold)"""
        print('Iteration {}'.format(seed))

        for name_index, name in enumerate(methods):
            # print(name)
            iter_pbar = tqdm(np.arange(n_iter), desc=f"\tProcessing {name}")

            # Check if it has been computer already
            config = dict(
                seed=seed,
                method=name,
                n_iter=n_iter - 1,
                dataset=dataset_id
            )

            if db.get(config, 'accuracy'):
                continue

            # Capture the output for logging
            with Tee() as tee:

                # Reset the splitter with initial conditions
                mask = np.loadtxt(initial_conditions_folder / '{}.txt'.format(seed))
                splitter = ActiveLearningSplitter.from_mask(mask)

                # Check that there is one sample from each class in the initialization
                assert (np.unique(y[splitter.selected]).shape[0] == np.unique(y).shape[0])
                assert(splitter.current_iter == 0)

                method = methods[name]
                classifier = get_clf(seed)
                previous_predicted = None
                previous_knn_predicted = None
                previous_min_dist_per_class = None

                for i in iter_pbar:

                    fit_clf(classifier, X[splitter.selected], y[splitter.selected])
                    predicted = _get_probability_classes(classifier, X)
            
                    ################################################
                    #  Parameters passed to the sampler generator  #
                    ################################################
                    sampler_params = dict(batch_size=args['batch_size'], clf=classifier, seed=int(seed), iteration=i, train_dataset=X[splitter.train])
                    """
                    train_dataset : labelised + unlabelised samples (all non test samples)
                    """

                    sampler = method(sampler_params)
                    sampler.fit(X[splitter.selected], y[splitter.selected])

                    new_selected_index = sampler.select_samples(X[splitter.non_selected])
                                    
                    splitter.add_batch(new_selected_index)

                    assert(splitter.current_iter == (i+1))
                    assert(splitter.selected_at(i).sum() == ((i + 1) * batch_size))
                    assert(splitter.batch_at(i).sum() == batch_size)

                    config = dict(
                        seed=seed,
                        method=name,
                        n_iter=i,
                        dataset=dataset_id
                    )

                    selected = splitter.selected_at(i)
                    batch = splitter.batch_at(i)


                    # Metrics
                    # =======

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

                    uniques, counts = np.unique(y, return_counts=True)
                    if n_classes == 2:
                        pos_label = uniques[np.argmin(counts)]
                        labels = [pos_label]
                        average_f_score = 'binary'
                    elif n_classes >= 2:
                        labels = []
                        pos_label = None
                        sum = np.sum(counts)
                        for label_id in uniques:
                            if (counts[label_id] / sum) <= (0.2 / n_classes):   # if class ratio is under 0.2 / nb_classes in multi-class
                                labels.append(label_id)
                        average_f_score = 'micro'
                    
                    assert y[splitter.test].shape == predicted_test.shape, f'{y[splitter.test].shape} , {predicted_test.shape}'
                    assert y[selected].shape == predicted_selected.shape
                    assert y[batch].shape == predicted_batch.shape

                    # Precision / Recall / F1 score

                    db.upsert('f_score_test', config, f1_score(y[splitter.test], predicted_test, labels=labels, pos_label=pos_label, average=average_f_score))
                    db.upsert('f_score_selected', config, f1_score(y[selected], predicted_selected, labels=labels, pos_label=pos_label, average=average_f_score))
                    db.upsert('f_score_batch', config, f1_score(y[batch], predicted_batch, labels=labels, pos_label=pos_label, average=average_f_score))


                    # ROC AUC score
                    # db.upsert('ROC_AUC_score_test', config, roc_auc_score(y[splitter.test], predicted_test, labels=labels, average='micro', multi_class='ovr'))
                    # db.upsert('ROC_AUC_score_selected', config, roc_auc_score(y[selected], predicted_selected, labels=labels, average='micro', multi_class='ovr'))
                    # # db.upsert('ROC_AUC_score_batch', config, roc_auc_score(y[batch], predicted_batch, labels=labels, average='micro', multi_class='ovr'))

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

                    distance_matrix = pairwise_distances(X[selected], X[splitter.test])
                    min_dist_per_class = get_min_dist_per_class(distance_matrix, predicted_selected)

                    if previous_min_dist_per_class is not None:
                        # pre_selected = splitter.selected_at(i-1)
                        # post_selected = splitter.selected_at(i+1)

                        # post_distance_matrix = pairwise_distances(X[post_selected], X[splitter.test])
                        # post_min_dist_per_class = get_min_dist_per_class(post_distance_matrix, predicted_selected)
                        
                        db.upsert('hard_exploration', config, np.mean(np.argmin(min_dist_per_class, axis=1) == np.argmin(previous_min_dist_per_class, axis=1)).item())
                        # db.upsert('soft_exploration', config, np.mean(np.abs(min_dist_per_class - post_min_dist_per_class)).item())
                        db.upsert('top_exploration', config, np.mean(np.min(previous_min_dist_per_class, axis=1) - np.min(min_dist_per_class, axis=1)).item())


                        # Batch Distance
                        # post_closest = post_distance_matrix.min(axis=1)
                        # closest = distance_matrix.min(axis=1)
                        
                        # db.upsert('post_closest', config, np.mean(post_closest))
                        # db.upsert('this_closest', config, np.mean(closest))
                        # db.upsert('diff_closest', config, np.mean(closest) - np.mean(post_closest))
                    
                    previous_min_dist_per_class = min_dist_per_class

                    # ================================================================================

                    # Trustscore 
                    # Metric skipped for dataset 43551 because of invalid shape error when calling trustscorer.score() (shape in axis 1: 0.)

                    # trustscorer = TrustScore()
                    # score = np.nan
                    # trustscorer.fit(X[selected], y[selected], classes=n_classes)
                    # max_k = np.unique(y[selected], return_counts=True)[1].min() - 1
                    # score = trustscorer.score(X[batch], predicted[batch], k=min(max_k, 10))[0].mean()
                    # db.upsert('batch_trustscore', config, score)

                    # np.random.seed(int(seed))
                    # idx_sel = np.random.choice(splitter.test.sum())

                    # score = trustscorer.score(X[splitter.test], predicted[splitter.test], k=min(max_k, 10))[0].mean()
                    # db.upsert('test_trustscore', config, score)
                    # score = trustscorer.score(X[selected], predicted[selected], k=min(max_k, 10))[0].mean()
                    # db.upsert('self_trustscore', config, score)

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
                        # print('Assertions learning failed')
                        score = 0
                        db.upsert('test_violation', config,  score)
                        db.upsert('self_violation', config, score)
                        db.upsert('batch_violation', config, score)



                    # ================================================================================

                    # Agreement
                    knn = KNeighborsClassifier()
                    knn.fit(X[selected], y[selected])
                    knn_predicted = knn.predict(X)
                    db.upsert('agreement_test', config,  np.mean(knn_predicted[splitter.test] == predicted_test))
                    db.upsert('agreement_selected', config,  np.mean(knn_predicted[selected] == predicted_selected))
                    db.upsert('agreement_batch', config,  np.mean(knn_predicted[batch] == predicted_batch))

                    # Contradictions agreement
                    if previous_knn_predicted is not None:
                        db.upsert('contradiction_knn_test', config, np.mean(previous_knn_predicted[splitter.test] != knn_predicted[splitter.test]))
                        db.upsert('contradiction_knn_selected', config, np.mean(previous_knn_predicted[selected] != knn_predicted[selected]))
                        db.upsert('contradiction_knn_batch', config, np.mean(previous_knn_predicted[batch] != knn_predicted[batch]))

                    
                    # ================================================================================
                    
                    previous_predicted = predicted
                    previous_knn_predicted = knn_predicted

            log_folder = experiment_folder / 'logs'
            log_folder.mkdir(exist_ok=True)
            with open(log_folder / '{}-{}-{}.log'.format(name, seed, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
                f.write(tee.read())
        

    # start = time.time()
    for seed in range(n_folds):
        run_AL_experiment(seed)
    # joblib.Parallel(n_jobs=5, prefer='processes')(joblib.delayed(run_AL_experiment)(seed) for seed in range(args['n_seed']) )
    # t_elapsed = time.time() - start
    # print(t_elapsed)

    # Plots results from saved csv
    plot_results(dataset_id, n_iter=n_iter, n_seed=n_folds, save_folder=experiment_folder / 'plots')

    # Propose to merge current sampler results to benchmark resutls
    share_results(dataset_id)


def plot_results(dataset_id, n_iter, n_seed, save_folder, show=False):
    """Plots experiments results for the current dataset (both user sampler results and previously saved benchmark sampler results)"""

    metrics = [
        ('Accuracy','accuracy_test.csv'),
        ('F-Score','f_score_test.csv'),
        # ('ROC-AUC-Score','ROC_AUC_score_test.csv'),
        ('Contradictions', 'contradiction_test.csv'),
        ('Agreement','agreement_test.csv'),
        # ('Trustscore','test_trustscore.csv'),
        ('Violation','test_violation.csv'),
        ('Hard-Exploration','hard_exploration.csv'),
        ('Top-Exploration','top_exploration.csv'),
        # ('Closest','this_closest.csv') 
    ]
    multiclass_tasks = ['42803', 'cifar10', 'cifar10_simclr', 'cifar100', 'cifar100_simclr', 'mnist']
    is_biclass_task = dataset_id not in multiclass_tasks

    plt.rc('font', size=20)          # controls default text sizes
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('lines', linewidth=3)

    for i, (metric_name, filename) in enumerate(metrics):
        plt.figure(i, figsize=(15,10))

        # Plot saved samplers results from the benchmark and then new sampler results

        df_bench = pd.read_csv(f'experiments/results_{dataset_id}/db/{filename}')   # Benchmark samplers results
        bench_method_names = np.unique(df_bench["method"].values)

        df = pd.read_csv(f'{save_folder}/results_{dataset_id}/db/{filename}')       # New sampler results
        method_names = np.unique(df["method"].values)

        colors = {method_name:key for method_name, (key, _) in zip(np.concatenate([bench_method_names,method_names]), cycle(mcolors.TABLEAU_COLORS.items()))}


        for sampler_name in bench_method_names:
            if is_biclass_task and sampler_name in ['confidence', 'margin']:   # We will plot with entropy metrics (indentical in biclass)
                continue

            all_metric = []
            for seed in range(n_seed):
                metric = df_bench.loc[(df_bench["method"] == sampler_name) & (df_bench["seed"]== seed)]['value'].values
                all_metric.append(metric)
            
            x_data = np.arange(n_iter-len(all_metric[0]), n_iter)
            if is_biclass_task and sampler_name == 'entropy':
                plot_confidence_interval(x_data, all_metric, label='{}'.format('uncertainty'.capitalize()), color=colors[sampler_name])
            else:
                plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name.capitalize()), color=colors[sampler_name])

        for sampler_name in method_names: #Loop in case their are several samplers tested here
            all_metric = []
            for seed in range(n_seed):
                metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
                all_metric.append(metric)
                
            x_data = np.arange(n_iter-len(all_metric[0]), n_iter)
            plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name.capitalize()), color=colors[sampler_name])


        plt.xlabel('AL iteration')
        plt.ylabel(metric_name)
        plt.title('{} metric'.format(metric_name))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_folder}/results_{dataset_id}/plot-' + metric_name + '.pdf')
        plt.savefig(f'{save_folder}/results_{dataset_id}/plot-' + metric_name + '.png')

    if show:
        plt.show()

    for i in range(len(metrics)):
        plt.figure(i).clear()    