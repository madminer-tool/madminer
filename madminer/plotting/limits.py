import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_pvalue_limits(
    p_values,
    best_fits,
    labels,
    grid_ranges,
    grid_resolutions,
    levels=[0.32],
    single_plot=True,
    show_index=None,
    xlabel=r"$\theta_0$",
    ylabel=r"$\theta_1$",
    p_val_min=0.001,
    p_val_max=1,
):

    """
    Function that plots the limits obtained from the AsymptoticLimits, Likelihood,
    FisherInformation and Information Geometry class. Note that only 2 dimensional
    grids are supported.

    Parameters
    ----------
    p_values : list of ndarray or dict
        List/dictionary of p-values with shape (nmethods, ngridpoints)

    best_fits : list of int or dict
        List/dictionary of best fit points for each method with shape (nmethods)

    labels : list of string or None
        List/dictionary of best labels for each method with shape (nmethods).
        If None, it is assumed that dictionaries are provided and all entries
        will be used.

    grid_ranges : list of (tuple of float) or None, optional
        Specifies the boundaries of the parameter grid on which the p-values
        are evaluated. It should be `[(min, max), (min, max), ..., (min, max)]`,
        where the list goes over all parameters and `min` and `max` are
        float. If None, thetas_eval has to be given. Default: None.

    grid_resolutions : int or list of int, optional
        Resolution of the parameter space grid on which the p-values are
        evaluated. If int, the resolution is the same along every dimension
        of the hypercube. If list of int, the individual entries specify the number of
        points along each parameter individually. Doesn't have any effect if
        grid_ranges is None. Default value: 25.

    levels : list of float, optional
        list of p-values used to draw contour lines. Default: [0.32]

    single_plot : bool, optional
        If True, only one summary plot is shown which contains confidence contours and
        best fit points for all methods, and the p-value grid for a selected method
        (if show_index is not None). If False, additional plots with the p-value grid,
        confidence contours and best fit points for all methods are provided. Default: True

    show_index : int, optional
        If None, no p-value grid is shown in summary plot. If show_index=n, the p-value
        grid of the nth method is shown in the summary plot. Default is None.

    xlabel,ylabel : string, optional
        Labels for the x and y axis. Default:  xlabel=r'$\theta_0$' and ylabel=r'$\theta_1$'.

    p_val_min,p_val_max : float, optional
        Plot range for p-values. Default: p_val_min=0.001 and p_val_max=1.

    """

    # Convert dict in array,if necessary
    if labels is None:
        if isinstance(p_values, dict) is False:
            raise ValueError("p_values should be a dictionary")
        if isinstance(best_fits, dict) is False:
            raise ValueError("best_fits should be a dictionary")
        labels = p_values.keys()
    if isinstance(p_values, dict):
        p_values = [p_values[label] for label in labels]
    if isinstance(best_fits, dict):
        best_fits = [best_fits[label] for label in labels]

    # Check input
    if len(p_values) != len(best_fits):
        raise ValueError("Length of arrays  p_values and best_fits should be the same")
    if len(p_values) != len(labels):
        raise ValueError("Length of arrays  p_values and labels should be the same")

    # Create theta grid
    if isinstance(grid_resolutions, int):
        grid_resolutions = [grid_resolutions for _ in range(grid_ranges)]
    if len(grid_resolutions) != 2:
        raise ValueError("Dimension of grid should be 2!")
    if len(grid_ranges) != 2:
        raise ValueError("Dimension of grid should be 2!")
    theta_each = []
    for resolution, (theta_min, theta_max) in zip(grid_resolutions, grid_ranges):
        theta_each.append(np.linspace(theta_min, theta_max, resolution))
    theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
    theta_grid_each = [theta.flatten() for theta in theta_grid_each]
    theta_grid = np.vstack(theta_grid_each).T

    # edges and centers
    xbin_size = (grid_ranges[0][1] - grid_ranges[0][0]) / (grid_resolutions[0] - 1)
    xedges = np.linspace(grid_ranges[0][0] - xbin_size / 2, grid_ranges[0][1] + xbin_size / 2, grid_resolutions[0] + 1)
    xcenters = np.linspace(grid_ranges[0][0], grid_ranges[0][1], grid_resolutions[0])
    ybin_size = (grid_ranges[1][1] - grid_ranges[1][0]) / (grid_resolutions[1] - 1)
    yedges = np.linspace(grid_ranges[1][0] - ybin_size / 2, grid_ranges[1][1] + ybin_size / 2, grid_resolutions[1] + 1)
    ycenters = np.linspace(grid_ranges[1][0], grid_ranges[1][1], grid_resolutions[1])

    # Preparing plot
    if single_plot is True:
        n_rows, n_cols = 1, 1
    else:
        n_cols = 3
        n_rows = (len(p_values) + n_cols) // n_cols
    fig = plt.figure(figsize=(6.0 * n_cols, 5.0 * n_rows))

    # plot summary plot
    ax = plt.subplot(n_rows, n_cols, 1)
    if show_index is not None:
        pcm = ax.pcolormesh(
            xedges,
            yedges,
            p_values[show_index].reshape((grid_resolutions[0], grid_resolutions[1])).T,
            norm=matplotlib.colors.LogNorm(vmin=p_val_min, vmax=p_val_max),
            cmap="Greys_r",
        )
        cbar = fig.colorbar(pcm, ax=ax, extend="both")
        cbar.set_label(f"Expected p-value ({labels[show_index]})")
    for i, panel in enumerate(p_values):
        ax.contour(
            xcenters,
            ycenters,
            panel.reshape((grid_resolutions[0], grid_resolutions[1])).T,
            levels=levels,
            colors=f"C{i}",
        )
        ax.scatter(
            theta_grid[best_fits[i]][0],
            theta_grid[best_fits[i]][1],
            s=80.0,
            color=f"C{i}",
            marker="*",
            label=labels[i],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # individual summary plot
    if single_plot is not True:
        for i, panel in enumerate(p_values):
            ax = plt.subplot(n_rows, n_cols, i + 2)
            pcm = ax.pcolormesh(
                xedges,
                yedges,
                panel.reshape((grid_resolutions[0], grid_resolutions[1])).T,
                norm=matplotlib.colors.LogNorm(vmin=p_val_min, vmax=p_val_max),
                cmap="Greys_r",
            )
            cbar = fig.colorbar(pcm, ax=ax, extend="both")
            cbar.set_label(f"Expected p-value ({labels[i]})")
            ax.contour(
                xcenters,
                ycenters,
                panel.reshape((grid_resolutions[0], grid_resolutions[1])).T,
                levels=levels,
                colors=f"C{i}",
            )
            ax.scatter(
                theta_grid[best_fits[i]][0],
                theta_grid[best_fits[i]][1],
                s=80.0,
                color=f"C{i}",
                marker="*",
                label=labels[i],
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    # finish plot
    plt.tight_layout()
    plt.show()
