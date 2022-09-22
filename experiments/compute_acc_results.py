"""
Script to compute benchmark results in a latex format

How to use:
You just have to specify the dataset ids you want to process and from which you want to get the results in the dataset_ids list below
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from cardinal.plotting import plot_confidence_interval
import os

dataset_ids = []    #[1461, 1471, 1502, 1590, 40922, 41138, 41162, 42395, 42803, 43439, 43551, 'cifar10', 'cifar10_simclr', 'mnist]
assert len(dataset_ids) > 0

n_iter = 10
n_seed=10

datasets_samplers_performances = {}      #shape (n_dataset, n_samplers)
for dataset_id in dataset_ids:
    datasets_samplers_performances[f'{dataset_id}'] = {} 

for dataset_id in dataset_ids:
    result_dir = f'results_{dataset_id}/'

    df = pd.read_csv(result_dir + 'db/accuracy_test.csv')

    method_names = np.unique(df["method"].values)
    with open(result_dir + 'summary_acc_results', 'w') as f:
        f.write(f"\n----- Dataset {dataset_id} ----- \n\n")
        for sampler_name in method_names:
            sampler_seeds_mean_acc = [np.mean(df.loc[(df["method"] == sampler_name)  & (df["seed"]== seed)]['value'].values) for seed in range(n_seed)] # mean over iterations
            sampler_mean_acc = round(np.mean(sampler_seeds_mean_acc), 3)
            sampler_mean_std = round(np.std(sampler_seeds_mean_acc), 3)
            datasets_samplers_performances[f'{dataset_id}'][f'{sampler_name}'] = (sampler_mean_acc, sampler_mean_std)
            f.write(f"\t{sampler_name} - {sampler_mean_acc*100} \pm({sampler_mean_std*100})\n")
    f.close()


#Global writing
df = pd.read_csv(f'results_{dataset_ids[0]}/db/accuracy_test.csv')
method_names = np.unique(df["method"].values)
with open('BENCH_SUMMARY_acc_results', 'w') as f:

    txt = "dataset id"
    for sampler_name in method_names:
        txt += f' & {sampler_name}'
    f.write(txt + '\\\\' + '\n')

    f.write('\midrule \n')

    # for sampler_name in method_names:
    for dataset_id in dataset_ids:
        dataset_perfs = [datasets_samplers_performances[f'{dataset_id}'][f'{sampler_name}'] 
                            if f'{sampler_name}' in datasets_samplers_performances[f'{dataset_id}'].keys() 
                            else (None, None)
                                for sampler_name in method_names]
        
        txt = f"{dataset_id}"
        for sampler_ac, sampler_std in dataset_perfs:
            if sampler_ac is not None:
                txt += f' & {np.round(sampler_ac*100, 1)} ($\pm${np.round(sampler_std*100, 1)})'
            else:
                txt += f' &  '
        f.write(txt + '\\\\'+'\n')

f.close()
