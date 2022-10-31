import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

from cardinal.utils import ActiveLearningSplitter
from cardinal.uncertainty import _get_probability_classes
from sklearn.metrics import accuracy_score, pairwise_distances, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from .prose import datainsights as di
from .utils import Tee, get_method_db
from .plotting import plot_results
from .data import get_dataset


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
        "n_classes": np.unique(y).shape[0]
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

    with open(str(initial_conditions_folder / 'experimental_parameters.json'), 'r') as experimental_parameters_file:
        experimental_parameters = json.load(experimental_parameters_file)
    
    return experimental_parameters


def load_experiment(experiment_name, initial_conditions):
    experiment_folder = Path('./experiment_results') / experiment_name

    with open(str(experiment_folder / 'experimental_parameters.json'), 'r') as experimental_parameters_file:
        experimental_parameters = json.load(experimental_parameters_file)

    if experimental_parameters["initial_conditions"] != initial_conditions['initial_conditions_name']:
        raise ValueError('Initial conditions and experiments do not match')

    experimental_parameters.update(initial_conditions)
    return experimental_parameters
    

def create_anonymous_2022_experiments():
    datasets_ids = [
        '1461', '1471', '1502', '1590', '40922', '41138', '42395',
        '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr',
        'mnist'
    ]

    for dataset_id in datasets_ids:
        batch_size_ratio = .005 if dataset_id == '1471' else .001
        create_initial_conditions(dataset_id, dataset_id, batch_size_ratio, 10)
        create_experiment(dataset_id, dataset_id, batch_size_ratio, 10)

    return datasets_ids


def get_min_dist_per_class(dist, labels, n_classes):
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


def run(experimental_parameters, methods):

    experiment_name = experimental_parameters['experiment_name']
    initial_conditions_name = experimental_parameters['initial_conditions']
    batch_size = experimental_parameters['batch_size']
    n_iter = experimental_parameters['n_iter']
    dataset_id = experimental_parameters['dataset']
    n_folds = experimental_parameters['n_folds']
    n_classes = experimental_parameters['n_classes']

    experiment_folder = Path('./experiment_results') / experiment_name
    initial_conditions_folder = Path('./experiment_indices') / initial_conditions_name

    print(f'\n--- Running experiment {experiment_name} ---\n')

    preproc = get_dataset(dataset_id)
    if len(preproc) == 3:
        X, y, get_best_model = preproc
    else:
        X, y, transformer, get_best_model = preproc
        X = transformer.fit_transform(X)

    n_classes = len(np.unique(y))
    batch_size = int(batch_size * X.shape[0]) if batch_size < 1 else batch_size

    # start = time.time()
    for seed in range(n_folds):
        print('Iteration {}'.format(seed))

        for name, get_sampler in methods.items():
            # print(name)
            iter_pbar = tqdm(np.arange(n_iter), desc=f"\tProcessing {name}")
            db = get_method_db(experimental_parameters, name)

            # Check if it has been computer already
            config = dict(
                seed=seed,
                method=name,
                n_iter=n_iter - 1,
                dataset=dataset_id
            )

            if db.get('accuracy_test', config) is not None:
                print('Already computed. Skipping.')
                continue

            # Capture the output for logging
            with Tee() as tee:

                # Reset the splitter with initial conditions
                mask = np.loadtxt(initial_conditions_folder / '{}.txt'.format(seed))
                splitter = ActiveLearningSplitter.from_mask(mask)

                # Check that there is one sample from each class in the initialization
                assert (np.unique(y[splitter.selected]).shape[0] == np.unique(y).shape[0])
                assert(splitter.current_iter == 0)

                classifier = get_best_model(seed)
                previous_predicted = None
                previous_knn_predicted = None
                previous_min_dist_per_class = None

                for i in iter_pbar:

                    classifier.fit(X[splitter.selected], y[splitter.selected])
                    predicted = _get_probability_classes(classifier, X)
            
                    sampler_params = dict(batch_size=batch_size, clf=classifier, seed=int(seed), iteration=i, train_dataset=X[splitter.train])
                    sampler = get_sampler(sampler_params)
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
                    min_dist_per_class = get_min_dist_per_class(distance_matrix, predicted_selected, n_classes)

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
    # joblib.Parallel(n_jobs=5, prefer='processes')(joblib.delayed(run_AL_experiment)(seed) for seed in range(args['n_seed']) )
    # t_elapsed = time.time() - start
    # print(t_elapsed)

    # Plots results from saved csv
    plot_results(experimental_parameters, methods)

    # Propose to merge current sampler results to benchmark resutls
    # share_results(dataset_id)
