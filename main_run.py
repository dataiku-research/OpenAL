import string
import sys
from sys import exit

sys.path.append('..')
sys.path.append('.')

import itertools
from pathlib import Path

from experiments.share_results import share_results
from bench.experiment import load_experiment, load_initial_conditions, create_experiment, create_initial_conditions




import bench.prose.datainsights as di

# Plots
from itertools import cycle
from bench.plotting import plot_confidence_interval # Upgraded version of cardinal plot_confidence_interval method
# from cardinal.plotting import plot_confidence_interval

import joblib
from tqdm import tqdm

# Setup matplotlib

cwd = Path.cwd()



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
        assert dataset_id in ['1461', '1471', '1502', '1590', '40922', '41138', '42395', '43439', '43551', '42803', '41162', 'cifar10', 'cifar10_simclr','cifar100', 'mnist'], f"{dataset_id} not in available datasets :\n- 1461 \n- 1471\n- 1502\n- 1590\n- 40922\n- 41138\n- 42395\n- 43439\n- 43551\n- 42803\n- 41162\n- cifar10\n- cifar10_simclr\n- cifar100\n- mnist"

    # Samplers should reming commented as they have already been ran and saved in the benchmark results
    methods = {
        'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=params['seed']),
        'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'confidence': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'entropy': lambda params: EntropySampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'kmeans': lambda params: KCentroidSampler(MiniBatchKMeans(n_clusters=params['batch_size'], n_init=1, random_state=params['seed']), batch_size=params['batch_size']),
        'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(10, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(10, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=params['seed']),
        'kcenter': lambda params: KCenterGreedy(AutoEmbedder(params['clf'], X=params['train_dataset']), batch_size=params['batch_size']),
    }

    for dataset_id in datasets_ids:
        initial_conditions = load_initial_conditions(dataset_id)
        experimental_parameters = load_experiment(dataset_id)
        run(experimental_parameters)


