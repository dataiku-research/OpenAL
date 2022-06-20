import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from cardinal.plotting import plot_confidence_interval


PLOT_TYPE =  "variance"  #   "results", "variance", "correlations"



n_iter = 10
x_data = np.arange(n_iter)
metrics = [
    ('Accuracy','accuracy_test.csv'),
    ('Agreement','agreement_test.csv'),
    ('Trustscore','test_trustscore.csv'),
    ('Violation','test_violation.csv'),
    ('Exploration','soft_exploration.csv'), #TODO important file to plot
    ('Closest','this_closest.csv')  #TODO important file to plot
]

# PLOT RESULTS
if PLOT_TYPE == "results":
    # dataset_id = 43551
    dataset_ids = [1461, 1471, 1502, 1590, 40922, 41138, 42395, 43439, 43551, 42803, 41162]
    for dataset_id in dataset_ids:
        for i, (metric_name, filename) in enumerate(metrics):
            try:
                df = pd.read_csv('results_{}/'.format(dataset_id)+filename)
                method_names = np.unique(df["method"].values)

                for sampler_name in method_names:
                    # if sampler_name in ['entropy', 'margin']:
                        all_metric = []
                        for seed in range(10):
                            metric = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value'].values
                            all_metric.append(metric)
                        
                        plt.figure(i, figsize=(15,10))
                        plot_confidence_interval(x_data, all_metric, label='{}'.format(sampler_name))

                plt.xlabel('AL iteration')
                plt.ylabel(metric_name)
                plt.title('{} metric'.format(metric_name))
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'results_{dataset_id}/plot-'+metric_name+'.png')
                plt.clf()
                
            except:
                print("[ERROR] Problem occured when trying to plot {} metric values".format(metric_name))

        plt.show()



if PLOT_TYPE == "variance":
    dataset_ids = [1461, 1471, 1502, 1590, 40922, 41138, 42395, 43439, 43551, 42803, 41162]
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