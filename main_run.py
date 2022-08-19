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
from cardinal.plotting import plot_confidence_interval

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


def run_benchmark(new_sampler_generator, 
                datasets_ids:list=['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist'], 
                sampler_name:string = 'my_custom_sampler'):

    assert sampler_name is not None
    assert datasets_ids is not None
    assert type(datasets_ids) == list, f'{datasets_ids} is of type {type(datasets_ids)} instead of type "list"'
    if len(datasets_ids) == 0 : datasets_ids = ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist']
    for dataset_id in datasets_ids:
        assert (dataset_id is not None)
        assert type(dataset_id) == str, f'{dataset_id} is of type {type(dataset_id)} instead of type "str"'
        assert dataset_id in ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr', 'mnist'], f"{dataset_id} not in available datasets :\n- 1461 \n- 1471\n- 1502\n- 1590\n- 40922\n- 41138\n- 42395\n- 43439\n- 43551\n- 42803\n- 41162\n- cifar10\n- cifar10_simclr\n- mnist"

    for dataset_id in datasets_ids:
        run(dataset_id, new_sampler_generator, sampler_name)


# @profile
def run(dataset_id, new_sampler_generator, sampler_name):
    print(f'\n--- RUN DATASET {dataset_id} ---\n')

    save_folder = 'experiments'
    if not os.path.isdir(f'{save_folder}/results_{dataset_id}/'): os.makedirs(f'{save_folder}/results_{dataset_id}/')
    db = CsvDb(f'{save_folder}/results_{dataset_id}/db')

    # X, y, transformer, best_model = get_openml(dataset_id)
    preproc = get_dataset(dataset_id)
    if len(preproc) == 3:
        X, y, best_model = preproc
    else:
        X, y, transformer, best_model = preproc
        X = transformer.fit_transform(X)

    get_clf = lambda seed: best_model(seed=seed)
    fit_clf = lambda clf, X, y: clf.fit(X, y)

    n_classes = len(np.unique(y))


    args = {
        "n_seed" : 10, 
        'n_iter' : 10, 
        'batch_size' : int(.001 * X.shape[0])   # int(.005 * X.shape[0]) -> 5% labelized
        }

    two_step_beta = 10

    # k_start = False
    start_size = args['batch_size']
    # oracle_error = False


    # model_cache = dict()

    methods = {
        # 'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=params['seed']),
        # 'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        # 'confidence': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        # 'entropy': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        # 'kmeans': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=params['seed']), batch_size=params['batch_size']),
        # 'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        # 'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(two_step_beta, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        # 'batchbald': lambda params: BatchBALDSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),     #TODO : check whitch one to use
        # 'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=X[splitter.train]), batch_size=params['batch_size']),
    }

    # Add new sampler method in the evaluated methods
    # methods[sampler_name] = new_sampler_generator


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
        print('Iteration {}'.format(seed))

        for name_index, name in enumerate(methods):
            # print(name)
            iter_pbar = tqdm(np.arange(args['n_iter']), desc=f"\tProcessing {name}")

            # Check if it has been computer already
            config = dict(
                seed=seed,
                method=name,
                n_iter=args['n_iter'] - 1,
                dataset=dataset_id
            )

            
            # Capture the output for logging
            with Tee() as tee:

                # Train test split 

                splitter = ActiveLearningSplitter.train_test_split(X.shape[0], test_size=.2, random_state=int(seed), stratify=y)
                # test_indexes = load_indexes(dataset_id, seed, type='test')
                # mask = np.full(X.shape[0], -1, dtype=np.int8)
                # mask[test_indexes] = -2
                # # TODO : revoir creation avec mask (wrt current_iter parameter )
                # splitter = ActiveLearningSplitter.from_mask(mask)   # [INFO] We instanciate the ActiveLearningSplitter with test sample indexes that have been registered and used in the previous benchmark (instead of using seeds)


                #Initialisation

                splitter.initialize_with_random(n_init_samples=start_size, at_least_one_of_each_class=y[splitter.train], random_state=int(seed))    #Seed very important for same initialisation between samplers
                first_index = np.where(splitter.selected == True)[0]
                # first_index = load_indexes(dataset_id, seed, type='random')     # [INFO] We select the same first samples indexes (randomly chosen for initialisation) that have been registered and used in the previous benchmark (instead of using seeds)
                # splitter.add_batch(first_index, partial=True)

                assert(np.unique(y[first_index]).shape[0] == np.unique(y).shape[0]), f'{np.unique(y[first_index]).shape[0]} != {np.unique(y).shape[0]}'


                method = methods[name]
                classifier = get_clf(seed)
                previous_predicted = None
                previous_knn_predicted = None
                previous_min_dist_per_class = None
                assert(splitter.selected.sum() == start_size), f"{splitter.selected.sum()}  {start_size}"
                assert(splitter.selected_at(0).sum() == start_size)
                assert(splitter.current_iter == 0)

                for i in iter_pbar:

                    fit_clf(classifier, X[splitter.selected], y[splitter.selected])
                    predicted = _get_probability_classes(classifier, X)
            
                    DYNAMIC_PARAMS = dict(batch_size=args['batch_size'], clf=classifier, seed=int(seed), iteration=i)  # HERE : PARAMETERS GIVEN TO THE SAMPLER AT EACH ITERATION             #iter=i + 1, splitter=splitter
                    # X_test = X[splitter.test]
                    # params['X_test'] = X_test

                    sampler = method(DYNAMIC_PARAMS)
                    sampler.fit(X[splitter.selected], y[splitter.selected])

                    new_selected_index = sampler.select_samples(X[splitter.non_selected])
                                    
                    splitter.add_batch(new_selected_index)

                    assert(splitter.current_iter == (i+1))
                    assert(splitter.selected_at(i).sum() == ((i+1) * args['batch_size'])) #+1 for initialisation batch
                    assert(splitter.batch_at(i).sum() == args['batch_size'])

                    config = dict(
                        seed=seed,
                        method=name,
                        n_iter=i,
                        dataset=dataset_id
                    )

                    selected = splitter.selected_at(i)
                    batch = splitter.batch_at(i)


                    # ================================================================================

                    # Performance

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
                        sum = np.sum(counts)
                        for label_id in uniques:
                            if (counts[label_id] / sum) <= (0.2 / n_classes):   # if class ratio is under 10% in bi-class      #TODO : define threshold
                                labels.append(label_id)
                        average_f_score = 'micro'

                    # Precision / Recall / F1 score

                    db.upsert('f_score_test', config, f1_score(y[splitter.test], predicted_test, labels=labels, pos_label=pos_label, average=average_f_score))
                    db.upsert('f_score_selected', config, f1_score(y[selected], predicted_selected, labels=labels, pos_label=pos_label, average=average_f_score))
                    db.upsert('f_score_batch', config, f1_score(y[batch], predicted_batch, labels=labels, pos_label=pos_label, average=average_f_score))


                    # ROC AUC score
                    db.upsert('ROC_AUC_score_test', config, roc_auc_score(y[splitter.test], predicted_test, labels=labels, average='micro', multi_class='ovr'))
                    db.upsert('ROC_AUC_score_selected', config, roc_auc_score(y[selected], predicted_selected, labels=labels, average='micro', multi_class='ovr'))
                    db.upsert('ROC_AUC_score_batch', config, roc_auc_score(y[batch], predicted_batch, labels=labels, average='micro', multi_class='ovr'))

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

                        # Save error 
                        db.upsert('FAIL_violation', config,  score)



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


        # Randomly selected samples
        if save_folder == 'experiments':
            try:
                df_to_save = pd.read_csv(f'{save_folder}/results_{dataset_id}/indexes.csv') 
                df = pd.DataFrame([{'seed': int(seed), 'type': 1, 'index':index} for index in first_index])
                df_to_save = pd.concat([df_to_save, df], ignore_index=True)
            except:
                df_to_save = pd.DataFrame([{'seed': seed, 'type': 1, 'index':index} for index in first_index])
            # Test indexes
            df = pd.DataFrame([{'seed': seed, 'type': 2, 'index':index} for index in np.where(splitter.test==True)[0]])
            df_to_save = pd.concat([df_to_save, df], ignore_index=True)
            df_to_save.to_csv(f'{save_folder}/results_{dataset_id}/indexes.csv', index=False)

    # start = time.time()
    for seed in range(args['n_seed']):
        run_AL_experiment(seed)
    # joblib.Parallel(n_jobs=5, prefer='processes')(joblib.delayed(run_AL_experiment)(seed) for seed in range(args['n_seed']) )
    # t_elapsed = time.time() - start
    # print(t_elapsed)


    # Plots results from saved csv
    plot_results(dataset_id, n_iter=args['n_iter'], n_seed=args['n_seed'], save_folder=save_folder)

    # Propose to merge current sampler results to benchmark resutls
    # share_results(dataset_id)


def plot_results(dataset_id, n_iter, n_seed, save_folder, show=False):

    # x_data = np.arange(n_iter)
    metrics = [
        ('Accuracy','accuracy_test.csv'),
        ('F-Score','f_score_test.csv'),
        ('ROC-AUC-Score','ROC_AUC_score_test.csv'),
        ('Contradictions', 'contradiction_test.csv'),
        ('Agreement','agreement_test.csv'),
        # ('Trustscore','test_trustscore.csv'),
        ('Violation','test_violation.csv'),
        ('Hard-Exploration','hard_exploration.csv'),
        ('Top-Exploration','top_exploration.csv'),
        # ('Closest','this_closest.csv') 
    ]

    for i, (metric_name, filename) in enumerate(metrics):

        # Plot new sampler results

        df = pd.read_csv(f'{save_folder}/results_{dataset_id}/db/{filename}')
        method_names = np.unique(df["method"].values)
        for sampler_name in method_names: #Loop in case their are several samplers tested here
            all_metric = []
            for seed in range(n_seed):
                metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
                all_metric.append(metric)
                
            plt.figure(i, figsize=(15,10))
            x_data = np.arange(n_iter-len(all_metric[0]), n_iter)
            plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name))
        

        # Plot other samplers results from the benchmark
        #TODO : uncomment for user use
        # plot_benchmark_sampler_results(i, dataset_id, filename, x_data, n_seed)
        # df = pd.read_csv(f'experiments/results_{dataset_id}/db/{filename}')
        # method_names = np.unique(df["method"].values)

        # for sampler_name in method_names:
        #     all_metric = []
        #     for seed in range(n_seed):
        #         metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
        #         all_metric.append(metric)
            
        #     plt.figure(i, figsize=(15,10))
        #     plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name))


        plt.xlabel('AL iteration')
        plt.ylabel(metric_name)
        plt.title('{} metric'.format(metric_name))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_folder}/results_{dataset_id}/plot-'+metric_name+'.png')

    if show:
        plt.show()

    for i in range(len(metrics)): plt.figure(i).clear()    


def load_indexes(dataset_id, seed, type):
    """
    Instead of only seeding the random initialisation, we use the indexes that have been saved during the previous benchmark run.
    Hence, all the samplers will have the same random initialisation samples
    """
    # if type == 'one_class':
    #     type = 0
    if type == 'random':
        type = 1
    elif type == 'test':
        type = 2 
    else:
        exit(f'[ERROR] Can’t load indexes from type {type}')

    df = pd.read_csv(f'experiments/results_{dataset_id}/indexes.csv')

    # return df.loc[(df["seed"] == seed) & (df["type"]== type)]['value'].values   # TODO When column name will be updated
    return df.loc[(df["seed"] == seed) & (df["type"]== type)]['index'].values   
