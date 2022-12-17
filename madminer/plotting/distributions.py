import logging

import matplotlib
import numpy as np

from matplotlib import pyplot as plt

from ..sampling import SampleAugmenter
from ..utils.morphing import NuisanceMorpher
from ..utils.various import shuffle
from ..utils.various import sanitize_array
from ..utils.various import mdot
from ..utils.various import weighted_quantile

logger = logging.getLogger(__name__)


def plot_distributions(
    filename,
    observables=None,
    parameter_points=None,
    uncertainties="nuisance",
    nuisance_parameters=None,
    draw_nuisance_toys=None,
    normalize=False,
    log=False,
    observable_labels=None,
    n_bins=50,
    line_labels=None,
    colors=None,
    linestyles=None,
    linewidths=1.5,
    toy_linewidths=0.5,
    alpha=0.15,
    toy_alpha=0.75,
    n_events=None,
    n_toys=100,
    n_cols=3,
    quantiles_for_range=(0.025, 0.975),
    sample_only_from_closest_benchmark=True,
):
    """
    Plots one-dimensional histograms of observables in a MadMiner file for a given set of benchmarks.

    Parameters
    ----------
    filename : str
        Filename of a MadMiner HDF5 file.

    observables : list of str or None, optional
        Which observables to plot, given by a list of their names. If None, all observables in the file
        are plotted. Default value: None.

    parameter_points : list of (str or ndarray) or None, optional
        Which parameter points to use for histogramming the data. Given by a list, each element can either be the name
        of a benchmark in the MadMiner file, or an ndarray specifying any parameter point in a morphing setup. If None,
        all physics (non-nuisance) benchmarks defined in the MadMiner file are plotted. Default value: None.

    uncertainties : {"nuisance", "none"}, optional
        Defines how uncertainty bands are drawn. With "nuisance", the variation in cross section from all nuisance
        parameters is added in quadrature. With "none", no error bands are drawn.

    nuisance_parameters : None or list of int, optional
        If uncertainties is "nuisance", this can restrict which nuisance parameters are used to draw the uncertainty
        bands. Each entry of this list is the index of one nuisance parameter (same order as in the MadMiner file).

    draw_nuisance_toys : None or int, optional
        If not None and uncertainties is "nuisance", sets the number of nuisance toy distributions that are drawn
        (in addition to the error bands).

    normalize : bool, optional
        Whether the distribution is normalized to the total cross section. Default value: False.

    log : bool, optional
        Whether to draw the y axes on a logarithmic scale. Default value: False.

    observable_labels : None or list of (str or None), optional
        x-axis labels naming the observables. If None, the observable names from the MadMiner file are used. Default
        value: None.

    n_bins : int, optional
        Number of histogram bins. Default value: 50.

    line_labels : None or list of (str or None), optional
        Labels for the different parameter points. If None and if parameter_points is None, the benchmark names from
        the MadMiner file are used. Default value: None.

    colors : None or str or list of str, optional
        Matplotlib line (and error band) colors for the distributions. If None, uses default colors. Default value:
        None.

    linestyles : None or str or list of str, optional
        Matplotlib line styles for the distributions. If None, uses default linestyles. Default value: None.

    linewidths : float or list of float, optional
        Line widths for the contours. Default value: 1.5.

    toy_linewidths : float or list of float or None, optional
        Line widths for the toy replicas, if uncertainties is "nuisance" and draw_nuisance_toys is not None. If None,
        linewidths is used. Default value: 1.

    alpha : float, optional
        alpha value for the uncertainty bands. Default value: 0.25.

    toy_alpha : float, optional
        alpha value for the toy replicas, if uncertainties is "nuisance" and draw_nuisance_toys is not None. Default
        value: 0.75.

    n_events : None or int, optional
        If not None, sets the number of events from the MadMiner file that will be analyzed and plotted. Default value:
        None.

    n_toys : int, optional
        Number of toy nuisance parameter vectors used to estimate the systematic uncertainties. Default value: 100.

    n_cols : int, optional
        Number of columns of subfigures in the plot. Default value: 3.

    quantiles_for_range : tuple of two float, optional
        Tuple `(min_quantile, max_quantile)` that defines how the observable range is determined for each panel.
        Default: (0.025, 0.075).

    sample_only_from_closest_benchmark : bool, optional
        If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    # Load data
    sa = SampleAugmenter(filename, include_nuisance_parameters=True)
    if uncertainties == "nuisance":
        nuisance_morpher = NuisanceMorpher(
            sa.nuisance_parameters, list(sa.benchmarks.keys()), reference_benchmark=sa.reference_benchmark
        )

    # Default settings
    if parameter_points is None:
        parameter_points = []

        for key, is_nuisance in zip(sa.benchmarks, sa.benchmark_nuisance_flags):
            if not is_nuisance:
                parameter_points.append(key)

    if line_labels is None:
        line_labels = parameter_points

    n_parameter_points = len(parameter_points)

    if colors is None:
        colors = [f"C{i}" for i in range(10)] * (n_parameter_points // 10 + 1)
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_parameter_points)]

    if linestyles is None:
        linestyles = ["solid", "dashed", "dotted", "dashdot"] * (n_parameter_points // 4 + 1)
    elif not isinstance(linestyles, list):
        linestyles = [linestyles for _ in range(n_parameter_points)]

    if not isinstance(linewidths, list):
        linewidths = [linewidths for _ in range(n_parameter_points)]

    if toy_linewidths is None:
        toy_linewidths = linewidths
    if not isinstance(toy_linewidths, list):
        toy_linewidths = [toy_linewidths for _ in range(n_parameter_points)]

    # Observables
    observable_indices = []
    if observables is None:
        observable_indices = list(range(len(sa.observables)))
    else:
        all_observables = list(sa.observables.keys())
        for obs in observables:
            try:
                observable_indices.append(all_observables.index(str(obs)))
            except ValueError:
                logging.warning("Ignoring unknown observable %s", obs)

    logger.debug("Observable indices: %s", observable_indices)

    n_observables = len(observable_indices)

    if observable_labels is None:
        all_observables = list(sa.observables.keys())
        observable_labels = [all_observables[obs] for obs in observable_indices]

    # Parse thetas
    theta_values = [sa._get_theta_value(theta) for theta in parameter_points]
    theta_matrices = [sa._get_theta_benchmark_matrix(theta) for theta in parameter_points]
    logger.debug("Calculated %s theta matrices", len(theta_matrices))

    # Get event data (observations and weights)
    all_x, all_weights_benchmarks = sa.weighted_events(generated_close_to=None)
    logger.debug("Loaded raw data with shapes %s, %s", all_x.shape, all_weights_benchmarks.shape)

    indiv_x, indiv_weights_benchmarks = [], []
    if sample_only_from_closest_benchmark:
        for theta in theta_values:
            this_x, this_weights = sa.weighted_events(generated_close_to=theta)
            indiv_x.append(this_x)
            indiv_weights_benchmarks.append(this_weights)

    # Remove negative weights
    sane_event_filter = np.all(all_weights_benchmarks >= 0.0, axis=1)

    n_events_before = all_weights_benchmarks.shape[0]
    all_x = all_x[sane_event_filter]
    all_weights_benchmarks = all_weights_benchmarks[sane_event_filter]
    n_events_removed = n_events_before - all_weights_benchmarks.shape[0]

    if int(np.sum(sane_event_filter, dtype=int)) < len(sane_event_filter):
        logger.warning("Removed %s / %s events with negative weights", n_events_removed, n_events_before)

    for i, (x, weights) in enumerate(zip(indiv_x, indiv_weights_benchmarks)):
        sane_event_filter = np.all(weights >= 0.0, axis=1)
        indiv_x[i] = x[sane_event_filter]
        indiv_weights_benchmarks[i] = weights[sane_event_filter]

    # Shuffle events
    all_x, all_weights_benchmarks = shuffle(all_x, all_weights_benchmarks)

    for i, (x, weights) in enumerate(zip(indiv_x, indiv_weights_benchmarks)):
        indiv_x[i], indiv_weights_benchmarks[i] = shuffle(x, weights)

    # Only analyze n_events
    if n_events is not None and n_events < all_x.shape[0]:
        logger.debug("Only analyzing first %s / %s events", n_events, all_x.shape[0])

        all_x = all_x[:n_events]
        all_weights_benchmarks = all_weights_benchmarks[:n_events]

        for i, (x, weights) in enumerate(zip(indiv_x, indiv_weights_benchmarks)):
            indiv_x[i] = x[:n_events]
            indiv_weights_benchmarks[i] = weights[:n_events]

    if uncertainties != "nuisance":
        n_toys = 0

    n_nuisance_toys_drawn = 0
    if draw_nuisance_toys is not None:
        n_nuisance_toys_drawn = draw_nuisance_toys

    # Nuisance parameters
    nuisance_toy_factors = []

    if uncertainties == "nuisance":
        n_nuisance_params = sa.n_nuisance_parameters

        if not n_nuisance_params > 0:
            raise RuntimeError("Cannot draw systematic uncertainties -- no nuisance parameters found!")

        logger.debug("Drawing nuisance toys")

        nuisance_toys = np.random.normal(loc=0.0, scale=1.0, size=n_nuisance_params * n_toys)
        nuisance_toys = nuisance_toys.reshape(n_toys, n_nuisance_params)

        # Restrict nuisance parameters
        if nuisance_parameters is not None:
            for i in range(n_nuisance_params):
                if i not in nuisance_parameters:
                    nuisance_toys[:, i] = 0.0

        logger.debug("Drew %s toy values for nuisance parameters", n_toys * n_nuisance_params)

        nuisance_toy_factors = np.array(
            [
                nuisance_morpher.calculate_nuisance_factors(nuisance_toy, all_weights_benchmarks)
                for nuisance_toy in nuisance_toys
            ]
        )  # Shape (n_toys, n_events)

        nuisance_toy_factors = sanitize_array(nuisance_toy_factors, min_value=1.0e-2, max_value=100.0)
        # Shape (n_toys, n_events)

    # Preparing plot
    n_rows = (n_observables + n_cols - 1) // n_cols
    n_events_for_range = 10000 if n_events is None else min(10000, n_events)

    fig = plt.figure(figsize=(4.0 * n_cols, 4.0 * n_rows))

    for i_panel, (i_obs, xlabel) in enumerate(zip(observable_indices, observable_labels)):
        logger.debug("Plotting panel %s: observable %s, label %s", i_panel, i_obs, xlabel)

        # Figure out x range
        xmins, xmaxs = [], []
        for theta_matrix in theta_matrices:
            x_small = all_x[:n_events_for_range]
            weights_small = mdot(theta_matrix, all_weights_benchmarks[:n_events_for_range])

            xmin = weighted_quantile(x_small[:, i_obs], quantiles_for_range[0], weights_small)
            xmax = weighted_quantile(x_small[:, i_obs], quantiles_for_range[1], weights_small)
            xwidth = xmax - xmin
            xmin -= xwidth * 0.1
            xmax += xwidth * 0.1

            xmin = max(xmin, np.min(all_x[:, i_obs]))
            xmax = min(xmax, np.max(all_x[:, i_obs]))

            xmins.append(xmin)
            xmaxs.append(xmax)

        xmin = min(xmins)
        xmax = max(xmaxs)
        x_range = (xmin, xmax)

        logger.debug("Ranges for observable %s: min = %s, max = %s", xlabel, xmins, xmaxs)

        # Subfigure
        ax = plt.subplot(n_rows, n_cols, i_panel + 1)

        # Calculate histograms
        bin_edges = None
        histos = []
        histos_up = []
        histos_down = []
        histos_toys = []

        for i_theta, theta_matrix in enumerate(theta_matrices):
            theta_weights = mdot(theta_matrix, all_weights_benchmarks)  # Shape (n_events,)

            if sample_only_from_closest_benchmark:
                indiv_theta_weights = mdot(theta_matrix, indiv_weights_benchmarks[i_theta])  # Shape (n_events,)
                histo, bin_edges = np.histogram(
                    indiv_x[i_theta][:, i_obs],
                    bins=n_bins,
                    range=x_range,
                    weights=indiv_theta_weights,
                    density=normalize,
                )
            else:
                histo, bin_edges = np.histogram(
                    all_x[:, i_obs], bins=n_bins, range=x_range, weights=theta_weights, density=normalize
                )
            histos.append(histo)

            if uncertainties == "nuisance":
                histos_toys_this_theta = []
                for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors):
                    toy_histo, _ = np.histogram(
                        all_x[:, i_obs],
                        bins=n_bins,
                        range=x_range,
                        weights=theta_weights * nuisance_toy_factors_this_toy,
                        density=normalize,
                    )
                    histos_toys_this_theta.append(toy_histo)

                histos_up.append(np.percentile(histos_toys_this_theta, 84.0, axis=0))
                histos_down.append(np.percentile(histos_toys_this_theta, 16.0, axis=0))
                histos_toys.append(histos_toys_this_theta[:n_nuisance_toys_drawn])

        # Draw error bands
        if uncertainties == "nuisance":
            for histo_up, histo_down, lw, color, label, ls in zip(
                histos_up, histos_down, linewidths, colors, line_labels, linestyles
            ):
                bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
                histo_down_ = np.repeat(histo_down, 2)
                histo_up_ = np.repeat(histo_up, 2)

                plt.fill_between(bin_edges_, histo_down_, histo_up_, facecolor=color, edgecolor="none", alpha=alpha)

            # Draw some toys
            for histo_toys, lw, color, ls in zip(histos_toys, toy_linewidths, colors, linestyles):
                for k in range(n_nuisance_toys_drawn):
                    bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
                    histo_ = np.repeat(histo_toys[k], 2)

                    plt.plot(bin_edges_, histo_, color=color, alpha=toy_alpha, lw=lw, ls=ls)

        # Draw central lines
        for histo, lw, color, label, ls in zip(histos, linewidths, colors, line_labels, linestyles):
            bin_edges_ = np.repeat(bin_edges, 2)[1:-1]
            histo_ = np.repeat(histo, 2)

            plt.plot(bin_edges_, histo_, color=color, lw=lw, ls=ls, label=label, alpha=1.0)

        plt.legend()

        plt.xlabel(xlabel)
        if normalize:
            plt.ylabel("Normalized distribution")
        else:
            plt.ylabel(r"$\frac{d\sigma}{dx}$ [pb / bin]")

        plt.xlim(x_range[0], x_range[1])
        if log:
            ax.set_yscale("log", nonposy="clip")
        else:
            plt.ylim(0.0, None)

    plt.tight_layout()

    return fig


def plot_histograms(
    histos,
    observed=None,
    observed_weights=None,
    xrange=None,
    yrange=None,
    zrange=None,
    log=False,
    histo_labels=None,
    observed_label="Data",
    xlabel=None,
    ylabel=None,
    zlabel=None,
    colors=None,
    linestyles=None,
    linewidths=1.5,
    markercolor="black",
    markersize=20.0,
    cmap="viridis",
    n_cols=2,
):
    # TODO: docstring

    # Basic setup
    n_histos = len(histos)
    dim = len(histos[0].edges)
    assert dim in [1, 2], f"Only 1- or 2-dimensional histograms are supported, but found {dim} dimensions"

    # Defaults
    if colors is None:
        colors = [f"C{i}" for i in range(10)] * (n_histos // 10 + 1)
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_histos)]
    if linestyles is None:
        linestyles = ["solid"] * n_histos
    elif not isinstance(linestyles, list):
        linestyles = [linestyles for _ in range(n_histos)]
    if not isinstance(linewidths, list):
        linewidths = [linewidths for _ in range(n_histos)]
    if histo_labels is None:
        histo_labels = [f"Histogram {i+1}" for i in range(n_histos)]

    # 1D plot
    if dim == 1:

        def _plot_histo(edges, histo, color=None, label=None, lw=1.5, ls="-"):
            edges_ = np.copy(edges)
            edges_ = np.repeat(edges_, 2)[1:-1]
            histo_ = np.repeat(histo, 2)
            plt.plot(edges_, histo_, color=color, lw=1.5, ls="-", label=label)

        fig = plt.figure(figsize=(5, 5))
        for histo, label, ls, lw, c in zip(histos, histo_labels, linestyles, linewidths, colors):
            _plot_histo(histo.edges[0], histo.histo, label=label, ls=ls, lw=lw, color=c)

        # Prepare observed data
        if observed is not None:
            observed = np.asarray(observed).squeeze()
            if len(observed.shape) > 1:
                observed = observed[0]
            obs_counts, obs_edges = np.histogram(observed, histos[0].edges[0], weights=observed_weights)
            obs_middles = 0.5 * (obs_edges[:-1] + obs_edges[1:])
            obs_counts /= (obs_edges[1:] - obs_edges[:-1]) * np.sum(obs_counts)
            plt.scatter(obs_middles, obs_counts, color=markercolor, marker="o", s=markersize, label=observed_label)

        plt.legend()
        if log:
            plt.yscale("log")
        if xrange is not None:
            plt.xlim(*xrange)
        if yrange is not None:
            plt.ylim(*yrange)
        else:
            plt.ylim(0.0, None)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("Likelihood")

    # 2D plot
    else:
        n_rows = (n_histos - 1) // n_cols + 1
        fig = plt.figure(figsize=(n_cols * 5.0, n_rows * 4.0))

        if observed is None:
            observed = [None for _ in histos]
        elif isinstance(observed, np.ndarray) and len(observed.squeeze().shape) <= 2:
            observed = [observed for _ in histos]

        for panel, (obs, histo, label) in enumerate(zip(observed, histos, histo_labels)):
            ax = plt.subplot(n_rows, n_cols, panel + 1)
            z = histo.histo.T
            if zrange is None:
                zrange = (np.min(z), np.max(z))
            z = np.clip(z, zrange[0] + 1.0e-12, zrange[1] - 1.0e-12)

            pcm = ax.pcolormesh(
                histo.edges[0],
                histo.edges[1],
                z,
                cmap=cmap,
                norm=matplotlib.colors.LogNorm(*zrange) if log else matplotlib.colors.Normalize(*zrange),
            )
            cbar = fig.colorbar(pcm, ax=ax, extend="both")

            # Prepare observed data
            if obs is not None:
                plt.scatter(
                    obs[:, 0],
                    obs[:, 1],
                    color=markercolor,
                    marker="o",
                    s=markersize
                    if observed_weights is None
                    else markersize * observed_weights / np.mean(observed_weights),
                    label=observed_label,
                )

            plt.title(label, fontsize=11.0)
            if xrange is not None:
                plt.xlim(*xrange)
            if yrange is not None:
                plt.ylim(*yrange)
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            if zlabel is not None:
                cbar.set_label(zlabel)
            else:
                cbar.set_label("Likelihood")

    plt.tight_layout()
    return fig
