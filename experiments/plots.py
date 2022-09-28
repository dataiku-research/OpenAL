"""
Script to plot again benchmark dataset results

How to use:
- Specify the dataset ids you want to process and from which you want to plot the results in the dataset_ids list below
- Run the script from the experiment folder
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
import pandas as pd
from cardinal.plotting import plot_confidence_interval
import os


PLOT_TYPE =  "results"  #   "results", "variance"
save_folder = 'experiments'
dataset_ids = []    #[1461, 1471, 1502, 1590, 40922, 41138, 41162, 42395, 42803, 43439, 43551, 'cifar10', 'cifar10_simclr', 'mnist]
assert len(dataset_ids) > 0



n_iter = 10
n_seed=10
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

plt.rc('font', size=20)          # controls default text sizes
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('lines', linewidth=3)


names = {
    'confidence': 'Confidence',
    'entropy': 'Entropy',
    'margin' : 'Margin',
    'iwkmeans': 'IWKMeans',
    'wkmeans': 'WKmeans',
    'kcenter': 'KCenter',
    'random': 'Random',
    'kmeans': 'KMeans',
    'uncertainty', 'Uncertainty'
}



# PLOT RESULTS
if PLOT_TYPE =='results':
    for dataset_id in dataset_ids:
        is_biclass_task = dataset_id not in multiclass_tasks
        for i, (metric_name, filename) in enumerate(metrics):

            # Plot new sampler results

            df = pd.read_csv(f'results_{dataset_id}/db/{filename}')
            method_names = np.unique(df["method"].values)
            colors = {method_name:key for method_name, (key, _) in zip(method_names, cycle(mcolors.TABLEAU_COLORS.items()))}

            for sampler_name in method_names:
                if is_biclass_task and sampler_name in ['confidence', 'margin']:   # We will plot with entropy metrics (indentical in biclass)
                    continue

                all_metric = []
                for seed in range(n_seed):
                    metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
                    all_metric.append(metric)
                    
                plt.figure(i, figsize=(15,10))
                x_data = np.arange(n_iter-len(all_metric[0]), n_iter)
                if is_biclass_task and sampler_name == 'entropy':
                    plot_confidence_interval(x_data, all_metric, label=names.get('uncertainty'), color=colors[sampler_name])
                else:
                    plot_confidence_interval(x_data, all_metric, label=names.get(sampler_name, sampler_name.capitalize()), color=colors[sampler_name])
                

            plt.xlabel('AL iteration')
            plt.ylabel(metric_name)
            plt.title('{} metric'.format(metric_name))
            plt.grid()
            plt.legend()
            plt.tight_layout()
            save_dir_path = f'results_{dataset_id}/paper-plots/'
            if not os.path.isdir(save_dir_path):
                os.makedirs(save_dir_path)
            plt.savefig(save_dir_path+ f'plot-'+metric_name+'.png')
            plt.savefig(save_dir_path+ f'plot-'+metric_name+'.pdf')

        # plt.show()    
        for i in range(len(metrics)): plt.figure(i).clear()        


if PLOT_TYPE == "variance":
    metric_name, filename = metrics[0]  #variance on accuracy

    df = pd.read_csv('results_{}/'.format(dataset_ids[0])+filename)
    barWidth = (1-0.2)/2
    iters_to_plot = [2, 8] # 'all' if we want to compute over all iterations

    for iter in iters_to_plot:
        inter_method = []
        intra_method = []

        # set height of bar
        for dataset_id in dataset_ids:
            df = pd.read_csv('results_{}/'.format(dataset_id)+filename)
            if iter == 'all':
                continue
            else:
                df = df[df['n_iter'] == iter]

            inter_method.append(df.groupby(['seed']).std().mean()['value'])
            intra_method.append(df.groupby(['method']).std().mean()['value'])

        
        # Set position of bar on X axis
        x1 = np.arange(len(dataset_ids))
        x2 = [x + barWidth for x in x1]
        
        # Make the plot
        fig = plt.figure(figsize = (10, 5))
        plt.bar(x1, intra_method, width = barWidth, label ='intra method')
        plt.bar(x2, inter_method, width = barWidth, label ='inter method')
        
        # Adding Xticks
        plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
        plt.ylabel('Variance', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth/2 for r in range(len(dataset_ids))], dataset_ids)
        plt.ylim(0,0.025)
        plt.legend()
        plt.title("Variance metrics on datasets at AL iteration n°{}".format(iter), fontweight ='bold', fontsize = 15)
        plt.show()
