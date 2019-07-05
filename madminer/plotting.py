from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import gridspec
import logging

from madminer.sampling import SampleAugmenter
from madminer.utils.morphing import NuisanceMorpher
from madminer.utils.various import weighted_quantile, sanitize_array, shuffle, mdot

logger = logging.getLogger(__name__)


def plot_uncertainty(
    filename,
    theta,
    observable,
    obs_label,
    obs_range,
    n_bins=50,
    nuisance_parameters=None,
    n_events=None,
    n_toys=100,
    linecolor="black",
    bandcolor1="#CC002E",
    bandcolor2="orange",
    ratio_range=(0.8, 1.2),
):
    """
    Plots absolute and relative uncertainty bands in a histogram of one observable in a MadMiner file.

    Parameters
    ----------
    filename : str
        Filename of a MadMiner HDF5 file.

    theta : ndarray, optional
        Which parameter points to use for histogramming the data.

    observable : str
        Which observable to plot, given by its name in the MadMiner file.

    obs_label : str
        x-axis label naming the observable.

    obs_range : tuple of two float
        Range to be plotted for the observable.

    n_bins : int
        Number of bins. Default value: 50.

    nuisance_parameters : None or list of int, optional
        This can restrict which nuisance parameters are used to draw the uncertainty
        bands. Each entry of this list is the index of one nuisance parameter (same order as in the MadMiner file).

    n_events : None or int, optional
        If not None, sets the number of events from the MadMiner file that will be analyzed and plotted. Default value:
        None.

    n_toys : int, optional
        Number of toy nuisance parameter vectors used to estimate the systematic uncertainties. Default value: 100.

    linecolor : str, optional
        Line color for central prediction. Default value: "black".

    bandcolor1 : str, optional
        Error band color for 1 sigma uncertainty. Default value: "#CC002E".

    bandcolor2 : str, optional
        Error band color for 2 sigma uncertainty. Default value: "orange".

    ratio_range : tuple of two floar
        y-axis range for the plots of the ratio to the central prediction. Default value: (0.8, 1.2).

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    # Load data
    sa = SampleAugmenter(filename, include_nuisance_parameters=True)
    nuisance_morpher = NuisanceMorpher(
        sa.nuisance_parameters, list(sa.benchmarks.keys()), reference_benchmark=sa.reference_benchmark
    )

    # Observable index
    obs_idx = list(sa.observables.keys()).index(observable)

    # Get event data (observations and weights)
    x, weights_benchmarks = sa.weighted_events()
    x = x[:, obs_idx]

    # Theta matrix
    theta_matrix = sa._get_theta_benchmark_matrix(theta)
    weights = mdot(theta_matrix, weights_benchmarks)

    # Remove negative weights
    x = x[weights >= 0.0]
    weights_benchmarks = weights_benchmarks[weights >= 0.0]
    weights = weights[weights >= 0.0]

    # Shuffle events
    x, weights, weights_benchmarks = shuffle(x, weights, weights_benchmarks)

    # Only analyze n_events
    if n_events is not None and n_events < x.shape[0]:
        x = x[:n_events]
        weights_benchmarks = weights_benchmarks[:n_events]
        weights = weights[:n_events]

    # Nuisance parameters
    n_nuisance_params = sa.n_nuisance_parameters

    nuisance_toys = np.random.normal(loc=0.0, scale=1.0, size=n_nuisance_params * n_toys)
    nuisance_toys = nuisance_toys.reshape(n_toys, n_nuisance_params)

    # Restrict nuisance parameters
    if nuisance_parameters is not None:
        for i in range(n_nuisance_params):
            if i not in nuisance_parameters:
                nuisance_toys[:, i] = 0.0

    nuisance_toy_factors = np.array(
        [
            nuisance_morpher.calculate_nuisance_factors(nuisance_toy, weights_benchmarks)
            for nuisance_toy in nuisance_toys
        ]
    )  # Shape (n_toys, n_events)

    nuisance_toy_factors = sanitize_array(nuisance_toy_factors, min_value=1.0e-2, max_value=100.0)
    # Shape (n_toys, n_events)

    # Calculate histogram for central prediction, not normalized
    histo, bin_edges = np.histogram(x, bins=n_bins, range=obs_range, weights=weights, density=False)

    # Calculate toy histograms, not normalized
    histos_toys_this_theta = []
    for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors):
        toy_histo, _ = np.histogram(
            x, bins=n_bins, range=obs_range, weights=weights * nuisance_toy_factors_this_toy, density=False
        )
        histos_toys_this_theta.append(toy_histo)

    histo_plus2sigma = np.percentile(histos_toys_this_theta, 97.5, axis=0)
    histo_plus1sigma = np.percentile(histos_toys_this_theta, 84.0, axis=0)
    histo_minus1sigma = np.percentile(histos_toys_this_theta, 16.0, axis=0)
    histo_minus2sigma = np.percentile(histos_toys_this_theta, 2.5, axis=0)

    # Calculate histogram for central prediction,  normalized
    histo_norm, bin_edges_norm = np.histogram(x, bins=n_bins, range=obs_range, weights=weights, density=True)

    # Calculate toy histograms, normalized
    histos_toys_this_theta = []
    for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors):
        toy_histo, _ = np.histogram(
            x, bins=n_bins, range=obs_range, weights=weights * nuisance_toy_factors_this_toy, density=True
        )
        histos_toys_this_theta.append(toy_histo)

    histo_plus2sigma_norm = np.percentile(histos_toys_this_theta, 97.5, axis=0)
    histo_plus1sigma_norm = np.percentile(histos_toys_this_theta, 84.0, axis=0)
    histo_minus1sigma_norm = np.percentile(histos_toys_this_theta, 16.0, axis=0)
    histo_minus2sigma_norm = np.percentile(histos_toys_this_theta, 2.5, axis=0)

    # Prepare plotting
    def plot_mc(edges, histo_central, histo_m2, histo_m1, histo_p1, histo_p2, relative=False):
        bin_edges_ = np.repeat(edges, 2)[1:-1]
        histo_ = np.repeat(histo_central, 2)
        histo_m2_ = np.repeat(histo_m2, 2)
        histo_m1_ = np.repeat(histo_m1, 2)
        histo_p1_ = np.repeat(histo_p1, 2)
        histo_p2_ = np.repeat(histo_p2, 2)

        if relative:
            histo_m2_ /= histo_
            histo_m1_ /= histo_
            histo_p1_ /= histo_
            histo_p2_ /= histo_
            histo_ /= histo_

        plt.fill_between(bin_edges_, histo_m2_, histo_p2_, facecolor=bandcolor2, edgecolor="none")
        plt.fill_between(bin_edges_, histo_m1_, histo_p1_, facecolor=bandcolor1, edgecolor="none")
        plt.plot(bin_edges_, histo_, color=linecolor, lw=1.5, ls="-")

    # Make plot
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # MC, absolute residuals
    ax = plt.subplot(gs[2])
    plot_mc(bin_edges, histo, histo_minus2sigma, histo_minus1sigma, histo_plus1sigma, histo_plus2sigma, relative=True)
    plt.xlabel(obs_label)
    plt.ylabel(r"Relative to central pred.")
    plt.xlim(obs_range[0], obs_range[1])
    plt.ylim(ratio_range[0], ratio_range[1])

    # MC, absolute
    ax = plt.subplot(gs[0], sharex=ax)
    plot_mc(bin_edges, histo, histo_minus2sigma, histo_minus1sigma, histo_plus1sigma, histo_plus2sigma)
    plt.ylabel(r"Differential cross section [pb/bin]")
    plt.ylim(0.0, None)
    plt.setp(ax.get_xticklabels(), visible=False)

    # MC, relative residuals
    ax = plt.subplot(gs[3])
    plot_mc(
        bin_edges_norm,
        histo_norm,
        histo_minus2sigma_norm,
        histo_minus1sigma_norm,
        histo_plus1sigma_norm,
        histo_plus2sigma_norm,
        relative=True,
    )
    plt.xlabel(r"$p_{T,\gamma}$ [GeV]")
    plt.ylabel(r"Relative to central pred.")
    plt.xlim(obs_range[0], obs_range[1])
    plt.ylim(ratio_range[0], ratio_range[1])

    # MC, relative
    ax = plt.subplot(gs[1], sharex=ax)
    plot_mc(
        bin_edges_norm,
        histo_norm,
        histo_minus2sigma_norm,
        histo_minus1sigma_norm,
        histo_plus1sigma_norm,
        histo_plus2sigma_norm,
    )
    plt.ylabel(r"Normalized distribution")
    plt.ylim(0.0, None)
    plt.setp(ax.get_xticklabels(), visible=False)

    # Return
    plt.tight_layout()
    return fig


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
        Whether to draw the y axes on a logarithmic scale. Defaul value: False.

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

        for key, is_nuisance in zip(sa.benchmarks, sa.benchmark_is_nuisance):
            if not is_nuisance:
                parameter_points.append(key)

        if line_labels is None:
            line_labels = parameter_points

    n_parameter_points = len(parameter_points)

    if colors is None:
        colors = ["C" + str(i) for i in range(10)] * (n_parameter_points // 10 + 1)
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

    if int(np.sum(sane_event_filter, dtype=np.int)) < len(sane_event_filter):
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


def plot_2d_morphing_basis(
    morpher,
    xlabel=r"$\theta_0$",
    ylabel=r"$\theta_1$",
    xrange=(-1.0, 1.0),
    yrange=(-1.0, 1.0),
    crange=(1.0, 100.0),
    resolution=100,
):
    """
    Visualizes a morphing basis and morphing errors for problems with a two-dimensional parameter space.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    xlabel : str, optional
        Label for the x axis. Default value: r'$\theta_0$'.

    ylabel : str, optional
        Label for the y axis. Default value: r'$\theta_1$'.

    xrange : tuple of float, optional
        Range `(min, max)` for the x axis. Default value: (-1., 1.).

    yrange : tuple of float, optional
        Range `(min, max)` for the y axis. Default value: (-1., 1.).

    crange : tuple of float, optional
        Range `(min, max)` for the color map. Default value: (1., 100.).

    resolution : int, optional
        Number of points per axis for the rendering of the squared morphing weights. Default value: 100.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    basis = morpher.basis

    assert basis is not None, "No basis defined"
    assert basis.shape[1] == 2, "Only 2d problems can be plotted with this function"

    xi, yi = (np.linspace(xrange[0], xrange[1], resolution), np.linspace(yrange[0], yrange[1], resolution))
    xx, yy = np.meshgrid(xi, yi)
    xx, yy = xx.reshape((-1, 1)), yy.reshape((-1, 1))
    theta_test = np.hstack([xx, yy])

    squared_weights = []
    for theta in theta_test:
        wi = morpher.calculate_morphing_weights(theta, None)
        squared_weights.append(np.sum(wi * wi) ** 0.5)
    squared_weights = np.array(squared_weights).reshape((resolution, resolution))

    fig = plt.figure(figsize=(6.5, 5))
    ax = plt.gca()

    pcm = ax.pcolormesh(
        xi, yi, squared_weights, norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]), cmap="viridis_r"
    )
    cbar = fig.colorbar(pcm, ax=ax, extend="both")

    plt.scatter(basis[:, 0], basis[:, 1], s=50.0, c="black")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar.set_label(r"$\sqrt{\sum w_i^2}$")
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])

    plt.tight_layout()

    return fig


def plot_nd_morphing_basis_scatter(morpher, crange=(1.0, 100.0), n_test_thetas=1000):
    """
    Visualizes a morphing basis and morphing errors with scatter plots between each pair of operators.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    crange : tuple of float, optional
        Range `(min, max)` for the color map. Default value: (1. 100.).

    n_test_thetas : int, optional
        Number of random points evaluated. Default value: 1000.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    basis = morpher.basis

    assert basis is not None, "No basis defined"

    n_parameters = basis.shape[1]

    #  Get squared weights
    thetas, squared_weights = morpher.evaluate_morphing(n_test_thetas=n_test_thetas, return_weights_and_thetas=True)

    # Plot
    fig = plt.figure(figsize=((n_parameters - 1) * 5.0, (n_parameters - 1) * 4.0))

    for iy in range(1, n_parameters):
        for ix in range(0, iy):
            i_panel = 1 + (iy - 1) * (n_parameters - 1) + ix
            ax = plt.subplot(n_parameters - 1, n_parameters - 1, i_panel)

            sc = plt.scatter(
                thetas[:, ix],
                thetas[:, iy],
                c=squared_weights,
                s=20.0,
                norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                cmap="viridis_r",
            )
            cbar = fig.colorbar(sc, ax=ax, extend="both")

            plt.scatter(basis[:, ix], basis[:, iy], s=100.0, lw=1.0, edgecolor="black", c="white")

            plt.xlabel(r"$\theta_" + str(ix) + "$")
            plt.ylabel(r"$\theta_" + str(iy) + "$")
            cbar.set_label(r"$\sqrt{\sum w_i^2}$")

    plt.tight_layout()

    return fig


def plot_nd_morphing_basis_slices(morpher, crange=(1.0, 100.0), resolution=50):
    """
    Visualizes a morphing basis and morphing errors with two-dimensional slices through parameter space.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    crange : tuple of float, optional
        Range `(min, max)` for the color map.

    resolution : int, optional
        Number of points per panel and axis for the rendering of the squared morphing weights. Default value: 50.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    basis = morpher.basis

    assert basis is not None, "No basis defined"

    n_parameters = basis.shape[1]

    # Plot
    fig = plt.figure(figsize=((n_parameters - 1) * 5.0, (n_parameters - 1) * 4.0))

    for iy in range(1, n_parameters):
        for ix in range(0, iy):
            i_panel = 1 + (iy - 1) * (n_parameters - 1) + ix
            ax = plt.subplot(n_parameters - 1, n_parameters - 1, i_panel)

            # Grid
            xrange = morpher.parameter_range[ix]
            yrange = morpher.parameter_range[iy]
            xi = np.linspace(xrange[0], xrange[1], resolution)
            yi = np.linspace(yrange[0], yrange[1], resolution)
            xx, yy = np.meshgrid(xi, yi)
            xx, yy = xx.flatten(), yy.flatten()

            theta_test = np.zeros((resolution ** 2, n_parameters))
            theta_test[:, ix] = xx
            theta_test[:, iy] = yy

            # Get squared weights
            squared_weights = []
            for theta in theta_test:
                wi = morpher.calculate_morphing_weights(theta, None)
                squared_weights.append(np.sum(wi * wi) ** 0.5)
            squared_weights = np.array(squared_weights).reshape((resolution, resolution))

            pcm = ax.pcolormesh(
                xi,
                yi,
                squared_weights,
                norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                cmap="viridis_r",
            )
            cbar = fig.colorbar(pcm, ax=ax, extend="both")

            plt.scatter(basis[:, ix], basis[:, iy], s=100.0, lw=1.0, edgecolor="black", c="white")

            plt.xlabel(r"$\theta_" + str(ix) + "$")
            plt.ylabel(r"$\theta_" + str(iy) + "$")
            cbar.set_label(r"$\sqrt{\sum w_i^2}$")

    plt.tight_layout()

    return fig


def plot_fisher_information_contours_2d(
    fisher_information_matrices,
    fisher_information_covariances=None,
    reference_thetas=None,
    contour_distance=1.0,
    xlabel=r"$\theta_0$",
    ylabel=r"$\theta_1$",
    xrange=(-1.0, 1.0),
    yrange=(-1.0, 1.0),
    labels=None,
    inline_labels=None,
    resolution=500,
    colors=None,
    linestyles=None,
    linewidths=1.5,
    alphas=1.0,
    alphas_uncertainties=0.25,
    ax=None,
):
    """
    Visualizes 2x2 Fisher information matrices as contours of constant Fisher distance from a reference point `theta0`.

    The local (tangent-space) approximation is used: distances `d(theta)` are given by
    `d(theta)^2 = (theta - theta0)_i I_ij (theta - theta0)_j`, summing over `i` and `j`.

    Parameters
    ----------
    fisher_information_matrices : list of ndarray
        Fisher information matrices, each with shape (2,2).

    fisher_information_covariances : None or list of (ndarray or None), optional
        Covariance matrices for the Fisher information matrices. Has to have the same length as
        fisher_information_matrices, and each entry has to be None (no uncertainty) or a tensor with shape
        (2,2,2,2). Default value: None.

    reference_thetas : None or list of (ndarray or None), optional
        Reference points from which the distances are calculated. If None, the origin (0,0) is used. Default value:
        None.

    contour_distance : float, optional.
        Distance threshold at which the contours are drawn. Default value: 1.

    xlabel : str, optional
        Label for the x axis. Default value: r'$\theta_0$'.

    ylabel : str, optional
        Label for the y axis. Default value: r'$\theta_1$'.

    xrange : tuple of float, optional
        Range `(min, max)` for the x axis. Default value: (-1., 1.).

    yrange : tuple of float, optional
        Range `(min, max)` for the y axis. Default value: (-1., 1.).

    labels : None or list of (str or None), optional
        Legend labels for the contours. Default value: None.

    inline_labels : None or list of (str or None), optional
        Inline labels for the contours. Default value: None.

    resolution : int
        Number of points per axis for the calculation of the distances. Default value: 500.

    colors : None or str or list of str, optional
        Matplotlib line (and error band) colors for the contours. If None, uses default colors. Default value: None.

    linestyles : None or str or list of str, optional
        Matploitlib line styles for the contours. If None, uses default linestyles. Default value: None.

    linewidths : float or list of float, optional
        Line widths for the contours. Default value: 1.5.

    alphas : float or list of float, optional
        Opacities for the contours. Default value: 1.

    alphas_uncertainties : float or list of float, optional
        Opacities for the error bands. Default value: 0.25.

    ax: axes or None, optional
        Predefined axes as part of figure instead of standalone figure. Default: None
    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    # Input data
    fisher_information_matrices = np.asarray(fisher_information_matrices)

    n_matrices = fisher_information_matrices.shape[0]

    if fisher_information_matrices.shape != (n_matrices, 2, 2):
        raise RuntimeError(
            "Fisher information matrices have shape {}, not (n, 2,2)!".format(fisher_information_matrices.shape)
        )

    if fisher_information_covariances is None:
        fisher_information_covariances = [None for _ in range(n_matrices)]

    if reference_thetas is None:
        reference_thetas = [None for _ in range(n_matrices)]

    d2_threshold = contour_distance ** 2.0

    # Line formatting
    if colors is None:
        colors = ["C" + str(i) for i in range(10)] * (n_matrices // 10 + 1)
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_matrices)]

    if linestyles is None:
        linestyles = ["solid", "dashed", "dotted", "dashdot"] * (n_matrices // 4 + 1)
    elif not isinstance(linestyles, list):
        linestyles = [linestyles for _ in range(n_matrices)]

    if not isinstance(linewidths, list):
        linewidths = [linewidths for _ in range(n_matrices)]

    if not isinstance(alphas, list):
        alphas = [alphas for _ in range(n_matrices)]

    if not isinstance(alphas_uncertainties, list):
        alphas_uncertainties = [alphas_uncertainties for _ in range(n_matrices)]

    # Grid
    xi = np.linspace(xrange[0], xrange[1], resolution)
    yi = np.linspace(yrange[0], yrange[1], resolution)
    xx, yy = np.meshgrid(xi, yi, indexing="xy")
    xx, yy = xx.flatten(), yy.flatten()
    thetas = np.vstack((xx, yy)).T

    # Theta from reference thetas
    d_thetas = []
    for reference_theta in reference_thetas:
        if reference_theta is None:
            d_thetas.append(thetas)
        else:
            d_thetas.append(thetas - reference_theta)
    d_thetas = np.array(d_thetas)  # Shape (n_matrices, n_thetas, n_parameters)

    # Calculate Fisher distances
    fisher_distances_squared = np.einsum("mni,mij,mnj->mn", d_thetas, fisher_information_matrices, d_thetas)
    fisher_distances_squared = fisher_distances_squared.reshape((n_matrices, resolution, resolution))

    # Calculate uncertainties of Fisher distances
    fisher_distances_squared_uncertainties = []
    for d_theta, inf_cov in zip(d_thetas, fisher_information_covariances):
        if inf_cov is None:
            fisher_distances_squared_uncertainties.append(None)
            continue

        var = np.einsum("ni,nj,ijkl,nk,nl->n", d_theta, d_theta, inf_cov, d_theta, d_theta)

        uncertainties = (var ** 0.5).reshape((resolution, resolution))
        fisher_distances_squared_uncertainties.append(uncertainties)

        logger.debug("Std: %s", uncertainties)

    # Plot results
    do_fig = False
    if ax is None:
        do_fig = True
        fig = plt.figure(figsize=(5.0, 5.0))
        ax = plt.gca()

    # Error bands
    for i in range(n_matrices):
        if fisher_information_covariances[i] is not None:
            d2_up = fisher_distances_squared[i] + fisher_distances_squared_uncertainties[i]
            d2_down = fisher_distances_squared[i] - fisher_distances_squared_uncertainties[i]
            band = (d2_up > d2_threshold) * (d2_down < d2_threshold) + (d2_up < d2_threshold) * (d2_down > d2_threshold)

            plt.contourf(xi, yi, band, [0.5, 2.5], colors=colors[i], alpha=alphas_uncertainties[i])

    # Predictions
    for i in range(n_matrices):
        cs = ax.contour(
            xi,
            yi,
            fisher_distances_squared[i],
            np.array([d2_threshold]),
            colors=colors[i],
            linestyles=linestyles[i],
            linewidths=linewidths[i],
            alpha=alphas[i],
            label=None if labels is None else labels[i],
        )

        if inline_labels is not None and inline_labels[i] is not None and len(inline_labels[i]) > 0:
            ax.clabel(cs, cs.levels, inline=True, fontsize=12, fmt={d2_threshold: inline_labels[i]})

    # Legend and decorations
    if labels is not None:
        ax.legend()

    if do_fig:
        plt.axes().set_xlim(xrange)
        plt.axes().set_ylim(yrange)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        return fig
    else:
        return ax


def plot_fisherinfo_barplot(
    fisher_information_matrices, labels, determinant_indices=None, eigenvalue_colors=None, bar_colors=None
):
    """

    Parameters
    ----------
    fisher_information_matrices : list of ndarray
        Fisher information matrices

    labels : list of str
        Labels for the x axis

    determinant_indices : list of int or None, optional
        If not None, the determinants will be based only on the indices given here. Default value: None.

    eigenvalue_colors : None or list of str
        Colors for the eigenvalue decomposition. If None, default colors are used. Default value: None.

    bar_colors : None or list of str
        Colors for the determinant bars. If None, default colors are used. Default value: None.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    # Prepare data
    if determinant_indices is None:
        matrices_for_determinants = fisher_information_matrices
    else:
        matrices_for_determinants = [m[determinant_indices, determinant_indices] for m in fisher_information_matrices]

    size_upper = len(fisher_information_matrices[1])
    size_lower = len(matrices_for_determinants[1])
    exponent_lower = 1.0 / float(size_lower)

    determinants = [np.linalg.det(m) ** exponent_lower for m in matrices_for_determinants]

    assert len(determinants) == len(labels)
    n_entries = len(determinants)

    # Calculate eigenvalues + eigenvalue composition
    eigenvalues = []
    eigenvalues_dominant_components = []
    eigenvalues_composition = []

    for m in fisher_information_matrices:
        v, w = np.linalg.eig(m)
        w = np.transpose(w)
        v, w = zip(*sorted(zip(v, w), key=lambda x: x[0], reverse=True))
        temp = []
        temp_dominant_components = []
        temp_composition = []
        for vi, wi in zip(v, w):
            temp.append(vi)
            temp_dominant_components.append(np.argmax(np.absolute(wi)))
            temp_composition.append(wi * wi / (sum(wi * wi)))

        eigenvalues.append(temp)
        eigenvalues_dominant_components.append(temp_dominant_components)
        eigenvalues_composition.append(temp_composition)

    # x positioning
    base_xvalues = np.linspace(0.0, float(n_entries) - 1.0, n_entries)
    base_xmin = base_xvalues[0]
    base_xmax = base_xvalues[n_entries - 1] + 1.0
    xmin_eigenvalues = base_xvalues + 0.08
    xmax_eigenvalues = base_xvalues + 0.92
    xpos_ticks = base_xvalues + 0.5
    xpos_lower = base_xvalues + 0.5
    width_lower = 0.8

    # Colors
    if bar_colors is None:
        bar_colors = ["0.5" for _ in range(n_entries)]
        bar_colors_light = ["0.9" for _ in range(n_entries)]
    else:
        bar_colors_light = bar_colors

    if eigenvalue_colors is None:
        eigenvalue_colors = ["C{}".format(str(i)) for i in range(10)]
    eigenvalue_linewidth = 1.5

    # Upper plot
    fig = plt.figure(figsize=(10.0, 7.0))
    ax1 = plt.subplot(211)

    # Plot eigenvalues
    for i in range(n_entries):
        for eigenvalue, composition in zip(eigenvalues[i], eigenvalues_composition[i]):
            # Gap sizing
            n_gaps = -1
            minimal_fraction_for_plot = 0.01
            for fraction in composition:
                if fraction >= minimal_fraction_for_plot:
                    n_gaps += 1
            gap_fraction = 0.04
            gap_correction_factor = 1.0 - n_gaps * gap_fraction

            fraction_finished = 0.0

            for component in range(len(composition)):
                fraction = composition[component]

                if fraction >= minimal_fraction_for_plot:
                    plt.hlines(
                        [eigenvalue],
                        xmin_eigenvalues[i] + fraction_finished * (xmax_eigenvalues[i] - xmin_eigenvalues[i]),
                        xmin_eigenvalues[i]
                        + (fraction_finished + gap_correction_factor * fraction)
                        * (xmax_eigenvalues[i] - xmin_eigenvalues[i]),
                        eigenvalue_colors[component],
                        linestyles="solid",
                        linewidth=eigenvalue_linewidth,
                    )
                    fraction_finished += gap_correction_factor * fraction + gap_fraction

    ax1.set_yscale("log")
    ax1.set_xlim([base_xmin - 0.2, base_xmax + 0.2])
    y_max = max([max(ev) for ev in eigenvalues])
    ax1.set_ylim(0.0001 * y_max, 2.0 * y_max)

    ax1.set_xticks(xpos_ticks)
    ax1.set_xticklabels(["" for _ in labels], rotation=40, ha="right")
    ax1.set_ylabel(r"$I_{ij}$ eigenvalues")

    # Lower plot
    ax3 = plt.subplot(212)

    bar_plot = ax3.bar(xpos_lower, determinants, width=width_lower, log=False)

    for i in range(n_entries):
        bar_plot[i].set_color(bar_colors_light[i])
        bar_plot[i].set_edgecolor(bar_colors[i])

    ax3.set_xlim([base_xmin - 0.2, base_xmax + 0.2])
    ax3.set_ylim([0.0, max(determinants) * 1.05])

    ax3.set_xticks(xpos_ticks)
    ax3.set_xticklabels(labels, rotation=40, ha="right")
    ax3.set_ylabel(r"$(\det \ I_{ij})^{1/" + str(size_lower) + r"}$")

    plt.tight_layout()
    return fig


def plot_distribution_of_information(
    xbins,
    xsecs,
    fisher_information_matrices,
    fisher_information_matrices_aux=None,
    xlabel=None,
    xmin=None,
    xmax=None,
    log_xsec=False,
    norm_xsec=True,
    epsilon=1.0e-9,
    figsize=(5.4, 4.5),
    fontsize=None,
):
    """
    Plots the distribution of the cross section together with the distribution of the Fisher information.

    Parameters
    ----------
    xbins : list of float
        Bin boundaries.

    xsecs : list of float
        Cross sections (in pb) per bin.

    fisher_information_matrices : list of ndarray
        Fisher information matrices for each bin.

    fisher_information_matrices_aux : list of ndarray or None, optional
        Additional Fisher information matrices for each bin (will be plotted with a dashed line).

    xlabel : str or None, optional
        Label for the x axis.

    xmin : float or None, optional
        Minimum value for the x axis.

    xmax : float or None, optional
        Maximum value for the x axis.

    log_xsec : bool, optional
        Whether to plot the cross section on a logarithmic y axis.

    norm_xsec : bool, optional
        Whether the cross sections are normalized to 1.

    epsilon : float, optional
        Numerical factor.
        
    figsize : tuple of float, optional
        Figure size, default: (5.4, 4.5)
        
    fontsize: float, optional
        Fontsize, default None

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    # prepare Plot
    if fontsize is not None:
        matplotlib.rcParams.update({"font.size": fontsize})

    # Prepare data
    n_entries = len(fisher_information_matrices)
    size = len(fisher_information_matrices[1])
    exponent = 1.0 / float(size)

    determinants = [np.nan_to_num(np.linalg.det(m) ** exponent) for m in fisher_information_matrices]

    if fisher_information_matrices_aux is not None:
        determinants_aux = [np.nan_to_num(np.linalg.det(m) ** exponent) for m in fisher_information_matrices_aux]

    if xlabel is None:
        xlabel = ""

    # Normalize xsecs
    if norm_xsec:
        norm = 1.0 / max(sum([xs for xs in xsecs]), epsilon)
    else:
        norm = 1.0
    xsec_norm = [norm * xs for xs in xsecs]

    # Get xvals from xbins
    xvals = [(xbins[i] + xbins[i + 1]) / 2 for i in range(0, len(xbins) - 1)]
    xvals = [xbins[0] - epsilon] + xvals + [xbins[len(xbins) - 1] + epsilon]
    assert len(xvals) == n_entries

    # Plotting options
    xs_color = "black"
    xs_linestyle = "-"
    xs_linewidth = 1.5

    det_color = "red"
    det_linestyle = "-"
    det_linewidth = 1.5
    det_fill_alpha = 0.1

    det_aux_color = "red"
    det_aux_linestyle = "--"
    det_aux_linewidth = 1.5

    # xsec plot
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(111)
    # fig.subplots_adjust(left=0.1667, right=0.8333, bottom=0.17, top=0.97)

    if log_xsec:
        ax1.set_yscale("log")

    ax1.hist(
        xvals,
        weights=xsec_norm,
        bins=xbins,
        range=(xmin, xmax),
        histtype="step",
        color=xs_color,
        linewidth=xs_linewidth,
        linestyle=xs_linestyle,
    )

    if norm_xsec:
        ax1.set_ylabel(r"Normalized distribution", color=xs_color)
    else:
        ax1.set_ylabel(r"$\sigma$ [pb/bin]")
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([0.0, max(xsec_norm) * 1.05])
    ax1.set_xlabel(xlabel)
    for tl in ax1.get_yticklabels():
        tl.set_color(xs_color)

    # det plot
    ax2 = ax1.twinx()

    if fisher_information_matrices_aux is not None:
        ax2.hist(
            xvals,
            weights=determinants_aux,
            bins=xbins,
            range=(xmin, xmax),
            histtype="step",
            color=det_aux_color,
            linewidth=det_aux_linewidth,
            linestyle=det_aux_linestyle,
        )

    ax2.hist(
        xvals,
        weights=determinants,
        bins=xbins,
        range=(xmin, xmax),
        histtype="stepfilled",
        alpha=det_fill_alpha,
        color=det_color,
        linewidth=0.0,
    )

    ax2.hist(
        xvals,
        weights=determinants,
        bins=xbins,
        range=(xmin, xmax),
        histtype="step",
        color=det_color,
        linewidth=det_linewidth,
        linestyle=det_linestyle,
    )

    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([0.0, max(determinants) * 1.1])
    ax2.set_ylabel(r"$(\det \; I_{ij})^{1/" + str(size) + "}$", color=det_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(det_color)

    return fig
