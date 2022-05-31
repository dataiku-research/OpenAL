import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from cardinal.plotting import plot_confidence_interval



# parser = argparse.ArgumentParser()
# parser.add_argument('dataset_id', type=int, help='Dataset to process')
# args = parser.parse_args()
# dataset_id = args.dataset_id
# del args

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