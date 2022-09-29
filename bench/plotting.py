from pathlib import Path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline

from .utils import get_method_db, REFERENCE_METHODS


def plot_confidence_interval(*args, label=None, q_inf=0.1, q_sup=0.9, alpha=.3, smoothing=None, dots=False, color=None, marker=None, markersize=None, markevery=None):
    y_data = np.asarray(args[-1])
    if len(args) == 1:
        # We only have the data. We create x-axis values.
        x_data = np.arange(y_data.shape[0])
    else:
        x_data = np.asarray(args[0])

    avg = np.mean(y_data, axis=0)
    q10 = np.quantile(y_data, q_inf, axis=0)
    q90 = np.quantile(y_data, q_sup, axis=0)

    if smoothing is not None:
        x_plot = np.linspace(x_data.min(), x_data.max(), x_data.shape[0] * smoothing) 
        avg_plot = make_interp_spline(x_data, avg, k=2)(x_plot)
        q10_plot = make_interp_spline(x_data, q10, k=2)(x_plot)
        q90_plot = make_interp_spline(x_data, q90, k=2)(x_plot)
    else:
        x_plot = x_data
        avg_plot = avg
        q10_plot = q10
        q90_plot = q90
    
    line_kwargs = {}
    if color is not None:
        line_kwargs['color'] = color
    if marker is not None:
        line_kwargs['marker'] = marker
    if markersize is not None:
        line_kwargs['markersize'] = markersize
    if markevery is not None:
        line_kwargs['markevery'] = markevery
    
    line = plt.plot(x_plot, avg_plot, label=label, **line_kwargs)
    color = line[0].get_c()

    if dots:
        plt.scatter(x_data, avg, c=color)

    plt.fill_between(x_plot, q90_plot, q10_plot, color=color, alpha=alpha)


def plot_results(experimental_parameters, methods, include_reference_methods=True, show=False):
    """Plots experiments results for the current dataset (both user sampler results and previously saved benchmark sampler results)"""

    experiment_name = experimental_parameters['experiment_name']
    initial_conditions_name = experimental_parameters['initial_conditions']
    batch_size = experimental_parameters['batch_size']
    n_iter = experimental_parameters['n_iter']
    dataset_id = experimental_parameters['dataset']
    n_folds = experimental_parameters['n_folds']
    n_initial_labeled = experimental_parameters['initial_labeled_size']
    n_classes = experimental_parameters['n_classes']

    experiment_folder = Path('./experiment_results') / experiment_name
    plot_folder = experiment_folder / 'plots'
    plot_folder.mkdir(exist_ok=True)

    if include_reference_methods:
        methods = list(methods) + list(REFERENCE_METHODS)
        methods = list(set(methods))

    dbs = [get_method_db(experimental_parameters, method) for method in methods]

    metrics = {
        'accuracy_test': 'Accuracy',
        # 'f_score_test': 'F-Score',
        'contradiction_test': 'Contradictions',
        'agreement_test': 'Agreement',
        'test_violation': 'Violation',
        'hard_exploration': 'Hard-Exploration',
        'top_exploration': 'Top-Exploration',
    }

    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=20)
    plt.rc('lines', linewidth=3)

    for metric_id, metric_name in metrics.items():
        plt.figure(figsize=(15,10))
        colors = {method_name:key for method_name, (key, _) in zip(methods, cycle(mcolors.TABLEAU_COLORS.items()))}

        for db, method in zip(dbs, methods):

            # All 3 methods are equivalent in binary classification, we show ony one.
            if n_classes == 2 and method in ['confidence', 'entropy']:
                continue
            
            df = db.get_dataframe(metric_id)
            if df is None:
                print('Missing metric', metric_id)
                continue
            df = df.reset_index()
            df = df[df["method"] == method]
            if df.shape[0] == 0:
                continue
            y_data = [df[df["seed"] == i]['value'].values for i in range(n_folds)]
            x_data = np.arange(n_iter - y_data[0].shape[0], n_iter)
            # if n_classes == 2 and method == 'margin':
            #     method = 'uncertainty'
            plot_confidence_interval(x_data, y_data, label='{}'.format(method.capitalize()), color=colors[method])

        plt.xlabel('Iteration')
        plt.ylabel(metric_name)
        plt.title('{} metric'.format(metric_name))
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_folder / 'plot-{}.pdf'.format(metric_name))
        plt.savefig(plot_folder / 'plot-{}.png'.format(metric_name))

    if show:
        plt.show()

    plt.close('all')  