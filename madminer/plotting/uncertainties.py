import logging
import numpy as np
from matplotlib import pyplot as plt, gridspec

from ..sampling import SampleAugmenter
from ..utils.morphing import NuisanceMorpher
from ..utils.various import mdot, shuffle, sanitize_array

logger = logging.getLogger(__name__)


def plot_uncertainty(
    filename,
    theta,
    observable,
    obs_label,
    obs_range,
    n_bins=50,
    systematics=None,
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

    systematics : None or list of str, optional
        This can restrict which nuisance parameters are used to draw the uncertainty
        bands. Each entry of this list is the name of a systematic uncertainty (see `MadMiner.add_systematics()`).

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
    if systematics is not None:
        nuisance_parameters = []
        for npar, (npar_syst, _, _) in sa.nuisance_parameters.items():
            if npar_syst in systematics:
                nuisance_parameters.append(npar)

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


def plot_systematics(
    filename,
    theta,
    observable,
    obs_label,
    obs_range,
    n_bins=50,
    n_events=None,
    n_toys=100,
    linecolor="black",
    bandcolors=None,
    band_alpha=0.2,
    ratio_range=(0.8, 1.2),
):
    """
    Plots absolute and relative uncertainty bands for all systematic uncertainties in a histogram of one observable in
    a MadMiner file.

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

    n_events : None or int, optional
        If not None, sets the number of events from the MadMiner file that will be analyzed and plotted. Default value:
        None.

    n_toys : int, optional
        Number of toy nuisance parameter vectors used to estimate the systematic uncertainties. Default value: 100.

    linecolor : str, optional
        Line color for central prediction. Default value: "black".

    bandcolors : None or list of str, optional
        Error band colors. Default value: None.

    ratio_range : tuple of two float
        y-axis range for the plots of the ratio to the central prediction. Default value: (0.8, 1.2).

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    # Colors
    if bandcolors is None:
        bandcolors = [f"C{i}" for i in range(10)]

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

    # Systematics
    n_systematics = len(sa.systematics) + 1
    labels = list(sa.systematics.keys()) + ["combined"]

    # Nuisance parameters
    n_nuisance_params = sa.n_nuisance_parameters

    nuisance_toys = np.random.normal(loc=0.0, scale=1.0, size=n_systematics * n_nuisance_params * n_toys)
    nuisance_toys = nuisance_toys.reshape(n_systematics, n_toys, n_nuisance_params)

    # Restrict nuisance parameters
    all_nuisance_parameters = list(sa.nuisance_parameters.keys())
    for i_syst, syst_name in enumerate(sa.systematics.keys()):
        n_used = n_nuisance_params
        used_nuisance_parameters = []
        for npar, (npar_syst, _, _) in sa.nuisance_parameters.items():
            if npar_syst == syst_name:
                used_nuisance_parameters.append(npar)

        for i in range(n_nuisance_params):
            if all_nuisance_parameters[i] not in used_nuisance_parameters:
                nuisance_toys[i_syst, :, i] = 0.0
                n_used -= 1
        logger.debug("Systematics %s based on %s nuisance parameters", syst_name, n_used)

    nuisance_toys = nuisance_toys.reshape(n_systematics * n_toys, n_nuisance_params)

    nuisance_toy_factors = np.array(
        [
            nuisance_morpher.calculate_nuisance_factors(nuisance_toy, weights_benchmarks)
            for nuisance_toy in nuisance_toys
        ]
    )  # Shape (n_systematics*n_toys, n_events)

    nuisance_toy_factors = sanitize_array(nuisance_toy_factors, min_value=1.0e-2, max_value=100.0)
    # Shape (n_systematics*n_toys, n_events)

    # Calculate histogram for central prediction, not normalized
    histo, bin_edges = np.histogram(x, bins=n_bins, range=obs_range, weights=weights, density=False)

    # Calculate toy histograms, not normalized
    histos_toys_this_theta = []
    for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors):
        toy_histo, _ = np.histogram(
            x, bins=n_bins, range=obs_range, weights=weights * nuisance_toy_factors_this_toy, density=False
        )
        histos_toys_this_theta.append(toy_histo)

    histos_toys_this_theta = np.array(histos_toys_this_theta)
    histos_toys_this_theta = histos_toys_this_theta.reshape(n_systematics, n_toys, -1)
    # Shape (n_systematics, n_toys, v)

    histo_plus1sigma = np.percentile(histos_toys_this_theta, 84.0, axis=1)
    histo_minus1sigma = np.percentile(histos_toys_this_theta, 16.0, axis=1)

    # Calculate histogram for central prediction,  normalized
    histo_norm, bin_edges_norm = np.histogram(x, bins=n_bins, range=obs_range, weights=weights, density=True)

    # Calculate toy histograms, normalized
    histos_toys_this_theta = []
    for i_toy, nuisance_toy_factors_this_toy in enumerate(nuisance_toy_factors):
        toy_histo, _ = np.histogram(
            x, bins=n_bins, range=obs_range, weights=weights * nuisance_toy_factors_this_toy, density=True
        )
        histos_toys_this_theta.append(toy_histo)
    histos_toys_this_theta = np.array(histos_toys_this_theta)
    histos_toys_this_theta = histos_toys_this_theta.reshape(n_systematics, n_toys, -1)
    # Shape (n_systematics, n_toys, n_bins)

    histo_plus1sigma_norm = np.percentile(histos_toys_this_theta, 84.0, axis=1)
    histo_minus1sigma_norm = np.percentile(histos_toys_this_theta, 16.0, axis=1)

    # Prepare plotting
    def plot_mc(edges, histo_central, histos_minus, histos_plus, relative=False):
        bin_edges_ = np.repeat(edges, 2)[1:-1]
        histo_central_ = np.repeat(histo_central, 2)
        histos_minus_ = [np.repeat(h, 2) for h in histos_minus]
        histos_plus_ = [np.repeat(h, 2) for h in histos_plus]

        if relative:
            histos_minus_ = [h / histo_central_ for h in histos_minus_]
            histos_plus_ = [h / histo_central_ for h in histos_plus_]
            histo_central_ /= histo_central_

        for i, (histo_minus_, histo_plus_) in enumerate(zip(histos_minus_, histos_plus_)):
            plt.fill_between(
                bin_edges_,
                histo_minus_,
                histo_plus_,
                facecolor=bandcolors[i % len(bandcolors)],
                alpha=band_alpha,
                edgecolor="none",
                label=labels[i],
            )
            plt.plot(bin_edges_, histo_minus_, color=bandcolors[i % len(bandcolors)], lw=1.0, ls="-")
            plt.plot(bin_edges_, histo_plus_, color=bandcolors[i % len(bandcolors)], lw=1.0, ls="-")
        plt.plot(bin_edges_, histo_central_, color=linecolor, lw=1.5, ls="-")

    # Make plot
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # MC, absolute residuals
    ax = plt.subplot(gs[2])
    plot_mc(bin_edges, histo, histo_minus1sigma, histo_plus1sigma, relative=True)
    plt.xlabel(obs_label)
    plt.ylabel(r"Relative to central pred.")
    plt.xlim(obs_range[0], obs_range[1])
    plt.ylim(ratio_range[0], ratio_range[1])

    # MC, absolute
    ax = plt.subplot(gs[0], sharex=ax)
    plot_mc(bin_edges, histo, histo_minus1sigma, histo_plus1sigma)
    plt.legend()
    plt.ylabel(r"Differential cross section [pb/bin]")
    plt.ylim(0.0, None)
    plt.setp(ax.get_xticklabels(), visible=False)

    # MC, relative residuals
    ax = plt.subplot(gs[3])
    plot_mc(bin_edges_norm, histo_norm, histo_minus1sigma_norm, histo_plus1sigma_norm, relative=True)
    plt.xlabel(obs_label)
    plt.ylabel(r"Relative to central pred.")
    plt.xlim(obs_range[0], obs_range[1])
    plt.ylim(ratio_range[0], ratio_range[1])

    # MC, relative
    ax = plt.subplot(gs[1], sharex=ax)
    plot_mc(bin_edges_norm, histo_norm, histo_minus1sigma_norm, histo_plus1sigma_norm)
    plt.legend()
    plt.ylabel(r"Normalized distribution")
    plt.ylim(0.0, None)
    plt.setp(ax.get_xticklabels(), visible=False)

    # Return
    plt.tight_layout()
    return fig
