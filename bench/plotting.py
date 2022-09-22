import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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
