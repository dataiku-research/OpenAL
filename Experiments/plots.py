import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from cardinal.plotting import plot_confidence_interval



# parser = argparse.ArgumentParser()
# parser.add_argument('dataset_id', type=int, help='Dataset to process')
# args = parser.parse_args()
# dataset_id = args.dataset_id
# del args

dataset_id = 1471
n_iter = 10

#Load data
df = pd.read_csv('results_{}/accuracy_test.csv'.format(dataset_id))
method_names = np.unique(df["method"].values)

for sampler_name in method_names:
    all_accuracies = []
    for seed in range(10):
        accuracies = df.loc[(df["method"] == sampler_name) & (df["seed"]== seed)]['value']
        all_accuracies.append(accuracies)


    # Plot
    x_data = np.arange(n_iter)
    plot_confidence_interval(x_data, all_accuracies, label='{}'.format(sampler_name))


plt.xlabel('AL iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

plt.savefig(f'results_{dataset_id}/accuracy-plot.png')
plt.show()